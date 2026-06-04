package kernelset

import (
	"os"
	"path/filepath"
	"runtime"
)

// sharedLibNames returns the platform-specific file names of the kernel-set
// shared library, most-specific first.
func sharedLibNames() []string {
	switch runtime.GOOS {
	case "windows":
		return []string{"kernel_set.dll", "libkernel_set.dll"}
	case "darwin":
		return []string{"libkernel_set.dylib"}
	default:
		return []string{"libkernel_set.so"}
	}
}

// standardLibDirs returns the directories LibPath searches, in priority order.
func standardLibDirs() []string {
	var dirs []string

	// Explicit per-loader search paths first.
	switch runtime.GOOS {
	case "windows":
		dirs = append(dirs, splitPathList(os.Getenv("PATH"))...)
	case "darwin":
		dirs = append(dirs, splitPathList(os.Getenv("DYLD_LIBRARY_PATH"))...)
		dirs = append(dirs, splitPathList(os.Getenv("DYLD_FALLBACK_LIBRARY_PATH"))...)
	default:
		dirs = append(dirs, splitPathList(os.Getenv("LD_LIBRARY_PATH"))...)
	}

	// Common install prefixes.
	dirs = append(dirs,
		"/usr/local/lib",
		"/usr/lib",
		"/usr/lib64",
		"/opt/kernel-set/lib",
	)
	return dirs
}

func splitPathList(v string) []string {
	if v == "" {
		return nil
	}
	return filepath.SplitList(v)
}

// LibPath reports the resolved on-disk path of the kernel-set shared library, if
// it can be found. Discovery order:
//
//  1. $KERNEL_SET_LIB — if it names a file, that file; if it names a directory,
//     a known library name inside it.
//  2. The platform loader search paths (LD_LIBRARY_PATH / DYLD_LIBRARY_PATH /
//     PATH) and common install prefixes (/usr/local/lib, /usr/lib, ...).
//
// The library is linked at build time via cgo's -lkernel_set, so LibPath is a
// diagnostic helper (for logging / "where will this load from?") rather than a
// loader — the dynamic linker still performs the actual resolution at run time.
// It returns ("", false) if nothing matching is found.
func LibPath() (string, bool) {
	names := sharedLibNames()

	if env := os.Getenv("KERNEL_SET_LIB"); env != "" {
		if info, err := os.Stat(env); err == nil {
			if info.IsDir() {
				for _, name := range names {
					p := filepath.Join(env, name)
					if fileExists(p) {
						return p, true
					}
				}
			} else {
				return env, true
			}
		}
	}

	for _, dir := range standardLibDirs() {
		if dir == "" {
			continue
		}
		for _, name := range names {
			p := filepath.Join(dir, name)
			if fileExists(p) {
				return p, true
			}
		}
	}
	return "", false
}

func fileExists(p string) bool {
	info, err := os.Stat(p)
	return err == nil && !info.IsDir()
}
