// Command omni runs the Janus-Pro-7B unified multimodal engine on a GPU.
//
// Build: go build -o omni .
// Run:   KERNEL_SET_LIB=/path/to/libkernel_set.so ./omni --model /path/to/janus-pro-7b
//
// The engine uses kernel-set CUDA kernels for all compute operations and
// loads safetensors weights directly.
package main

import (
	"flag"
	"fmt"
	"image"
	"image/png"
	"os"
	"path/filepath"
	"strings"
	"time"
)

var (
	debugLogits = flag.Bool("debug-logits", false, "dump first-token logits to /tmp/go_logits.bin")
)

var (
	mode         = flag.String("mode", "understand", "text | understand | generate")
	modelDir     = flag.String("model", "", "path to Janus-Pro-7b safetensors directory")
	imagePath    = flag.String("image", "", "input image path (for understanding)")
	prompt       = flag.String("prompt", "Describe the image in detail.", "text prompt")
	maxNewTokens = flag.Int("max-tokens", 128, "max tokens to generate")
	outImage     = flag.String("out", "output.png", "output image path (for generation)")
)

func main() {
	flag.Parse()

	// For generate mode, auto-calculate required tokens: 24x24 = 576 for 384x384.
	if *mode == "generate" && *maxNewTokens < 576 {
		*maxNewTokens = 576
	}

	if *modelDir == "" {
		fmt.Fprintln(os.Stderr, "usage: omni --model <dir> [--image <path>] [--prompt <text>]")
		os.Exit(1)
	}

	engine, err := NewJanusPro(*modelDir)
	if err != nil {
		fmt.Fprintf(os.Stderr, "load model: %v\n", err)
		os.Exit(1)
	}
	defer engine.Close()

	if err := initTokenizer(); err != nil {
		fmt.Fprintf(os.Stderr, "init tokenizer: %v\n", err)
		os.Exit(1)
	}
	defer closeTokenizer()

	switch *mode {
	case "text":
		if *prompt == "" {
			fmt.Fprintln(os.Stderr, "--prompt required for text mode")
			os.Exit(1)
		}
		start := time.Now()
		text, err := engine.TextOnly(*prompt, *maxNewTokens, debugLogitsPath())
		elapsed := time.Since(start)
		if err != nil {
			fmt.Fprintf(os.Stderr, "text: %v\n", err)
			os.Exit(1)
		}
		fmt.Println(text)
		fmt.Fprintf(os.Stderr, "[omni] %d tokens in %.1fs\n", *maxNewTokens, elapsed.Seconds())

	case "understand":
		if *imagePath == "" {
			fmt.Fprintln(os.Stderr, "--image required for understanding mode")
			os.Exit(1)
		}
		img, err := loadImage(*imagePath)
		if err != nil {
			fmt.Fprintf(os.Stderr, "load image: %v\n", err)
			os.Exit(1)
		}
		start := time.Now()
		text, err := engine.Understand(img, *prompt, *maxNewTokens, debugLogitsPath())
		elapsed := time.Since(start)
		if err != nil {
			fmt.Fprintf(os.Stderr, "understand: %v\n", err)
			os.Exit(1)
		}
		fmt.Println(text)
		fmt.Fprintf(os.Stderr, "[omni] %d tokens in %.1fs\n", *maxNewTokens, elapsed.Seconds())

	case "generate":
		start := time.Now()
		outImg, err := engine.Generate(*prompt, *maxNewTokens)
		elapsed := time.Since(start)
		if err != nil {
			fmt.Fprintf(os.Stderr, "generate: %v\n", err)
			os.Exit(1)
		}
		f, err := os.Create(*outImage)
		if err != nil {
			fmt.Fprintf(os.Stderr, "create output: %v\n", err)
			os.Exit(1)
		}
		if err := png.Encode(f, outImg); err != nil {
			f.Close()
			fmt.Fprintf(os.Stderr, "encode png: %v\n", err)
			os.Exit(1)
		}
		f.Close()
		fmt.Printf("wrote %s\n", *outImage)
		fmt.Fprintf(os.Stderr, "[omni] %d steps in %.1fs\n", *maxNewTokens, elapsed.Seconds())

	default:
		fmt.Fprintf(os.Stderr, "unknown mode: %s\n", *mode)
		os.Exit(1)
	}
}

func debugLogitsPath() string {
	if *debugLogits {
		return "/tmp/go_logits.bin"
	}
	return ""
}

func loadImage(path string) (*image.RGBA, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	img, _, err := image.Decode(f)
	if err != nil {
		return nil, err
	}
	// Bilinear resize to 384x384 (matches PIL default)
	return bilinearResize(img, 384, 384), nil
}

// bilinearResize implements bilinear interpolation resizing.
func bilinearResize(src image.Image, dstW, dstH int) *image.RGBA {
	srcW := src.Bounds().Dx()
	srcH := src.Bounds().Dy()
	dst := image.NewRGBA(image.Rect(0, 0, dstW, dstH))
	xScale := float64(srcW) / float64(dstW)
	yScale := float64(srcH) / float64(dstH)
	for y := 0; y < dstH; y++ {
		srcY := float64(y)*yScale
		srcY0 := int(srcY)
		srcY1 := srcY0 + 1
		if srcY1 >= srcH {
			srcY1 = srcH - 1
		}
		yFrac := srcY - float64(srcY0)
		for x := 0; x < dstW; x++ {
			srcX := float64(x)*xScale
			srcX0 := int(srcX)
			srcX1 := srcX0 + 1
			if srcX1 >= srcW {
				srcX1 = srcW - 1
			}
			xFrac := srcX - float64(srcX0)
			// Sample 4 corners
			r00, g00, b00, _ := src.At(srcX0, srcY0).RGBA()
			r01, g01, b01, _ := src.At(srcX1, srcY0).RGBA()
			r10, g10, b10, _ := src.At(srcX0, srcY1).RGBA()
			r11, g11, b11, _ := src.At(srcX1, srcY1).RGBA()
			// Bilinear interpolation
			r := uint8(bilinear(float64(r00>>8), float64(r01>>8), float64(r10>>8), float64(r11>>8), xFrac, yFrac))
			g := uint8(bilinear(float64(g00>>8), float64(g01>>8), float64(g10>>8), float64(g11>>8), xFrac, yFrac))
			b := uint8(bilinear(float64(b00>>8), float64(b01>>8), float64(b10>>8), float64(b11>>8), xFrac, yFrac))
			off := dst.PixOffset(x, y)
			s := dst.Pix[off : off+4 : off+4]
			s[0] = r
			s[1] = g
			s[2] = b
			s[3] = 255
		}
	}
	return dst
}

func bilinear(c00, c01, c10, c11, xFrac, yFrac float64) float64 {
	return c00*(1-xFrac)*(1-yFrac) + c01*xFrac*(1-yFrac) + c10*(1-xFrac)*yFrac + c11*xFrac*yFrac
}

// safetensors helper: locate all .safetensors files in a directory.
func findSafetensors(dir string) ([]string, error) {
	entries, err := os.ReadDir(dir)
	if err != nil {
		return nil, err
	}
	var paths []string
	for _, e := range entries {
		if !e.IsDir() && strings.HasSuffix(e.Name(), ".safetensors") {
			paths = append(paths, filepath.Join(dir, e.Name()))
		}
	}
	return paths, nil
}