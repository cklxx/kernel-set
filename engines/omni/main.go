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

func main() {
	modelDir := flag.String("model", "", "path to Janus-Pro-7b safetensors directory")
	imagePath := flag.String("image", "", "input image path (for understanding)")
	prompt := flag.String("prompt", "Describe this image in detail.", "text prompt")
	maxNewTokens := flag.Int("max-tokens", 128, "max tokens to generate")
	mode := flag.String("mode", "understand", "understand | generate")
	outImage := flag.String("out", "output.png", "output image path (for generation)")
	flag.Parse()

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
		text, err := engine.Understand(img, *prompt, *maxNewTokens)
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
	b := img.Bounds()
	rgba := image.NewRGBA(b)
	for y := b.Min.Y; y < b.Max.Y; y++ {
		for x := b.Min.X; x < b.Max.X; x++ {
			rgba.Set(x, y, img.At(x, y))
		}
	}
	return rgba, nil
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