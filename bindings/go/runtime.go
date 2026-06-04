package kernelset

/*
#include "kernel_set/kernel_set.h"
*/
import "C"

import "unsafe"

// ---- Library / build introspection -------------------------------------

// Version returns the kernel-set semantic version string (ks_version).
func Version() string {
	return C.GoString(C.ks_version())
}

// BackendName returns the backend this build targets: "cuda", "hip", or "none"
// (ks_backend_name).
func BackendName() string {
	return C.GoString(C.ks_backend_name())
}

// ---- Device properties -------------------------------------------------

// DeviceProperties mirrors ks_device_properties_t.
type DeviceProperties struct {
	Name                    string
	ComputeMajor            int
	ComputeMinor            int
	MultiprocessorCount     int
	MaxThreadsPerBlock      int
	MaxSharedMemoryPerBlock int // bytes
	WarpSize                int
	TotalGlobalMemory       uint64 // bytes
	SupportsBF16            bool
	SupportsFP8             bool
	SupportsTF32            bool
}

// ---- Device queries ----------------------------------------------------

// DeviceCount returns the number of visible GPU devices (ks_device_count).
func DeviceCount() (int, error) {
	var n C.int
	st := Status(C.ks_device_count(&n))
	if err := statusError(st, "ks_device_count"); err != nil {
		return 0, err
	}
	return int(n), nil
}

// SetDevice selects the active GPU device (ks_set_device).
func SetDevice(device int) error {
	return statusError(Status(C.ks_set_device(C.int(device))), "ks_set_device")
}

// GetDevice returns the active GPU device index (ks_get_device).
func GetDevice() (int, error) {
	var d C.int
	st := Status(C.ks_get_device(&d))
	if err := statusError(st, "ks_get_device"); err != nil {
		return 0, err
	}
	return int(d), nil
}

// GetDeviceProperties queries the capabilities of a device
// (ks_get_device_properties).
func GetDeviceProperties(device int) (DeviceProperties, error) {
	var p C.ks_device_properties_t
	st := Status(C.ks_get_device_properties(C.int(device), &p))
	if err := statusError(st, "ks_get_device_properties"); err != nil {
		return DeviceProperties{}, err
	}
	return DeviceProperties{
		Name:                    C.GoString(&p.name[0]),
		ComputeMajor:            int(p.compute_major),
		ComputeMinor:            int(p.compute_minor),
		MultiprocessorCount:     int(p.multiprocessor_count),
		MaxThreadsPerBlock:      int(p.max_threads_per_block),
		MaxSharedMemoryPerBlock: int(p.max_shared_memory_per_block),
		WarpSize:                int(p.warp_size),
		TotalGlobalMemory:       uint64(p.total_global_memory),
		SupportsBF16:            p.supports_bf16 != 0,
		SupportsFP8:             p.supports_fp8 != 0,
		SupportsTF32:            p.supports_tf32 != 0,
	}, nil
}

// ---- Device memory -----------------------------------------------------

// MallocDevice allocates `bytes` of device memory and returns the device
// pointer (ks_malloc_device). Free it with FreeDevice.
func MallocDevice(bytes uintptr) (unsafe.Pointer, error) {
	var ptr unsafe.Pointer
	st := Status(C.ks_malloc_device(&ptr, C.size_t(bytes)))
	if err := statusError(st, "ks_malloc_device"); err != nil {
		return nil, err
	}
	return ptr, nil
}

// FreeDevice releases device memory allocated by MallocDevice (ks_free_device).
func FreeDevice(ptr unsafe.Pointer) error {
	return statusError(Status(C.ks_free_device(ptr)), "ks_free_device")
}

// Memcpy copies `bytes` between host/device buffers in the given direction on a
// stream (ks_memcpy). dst and src are raw addresses appropriate for `kind`.
func Memcpy(dst, src unsafe.Pointer, bytes uintptr, kind MemcpyKind, stream Stream) error {
	st := Status(C.ks_memcpy(dst, src, C.size_t(bytes),
		C.ks_memcpy_kind_t(kind), stream.c()))
	return statusError(st, "ks_memcpy")
}

// MemsetDevice sets `bytes` of device memory to the byte `value` (ks_memset_device).
func MemsetDevice(dst unsafe.Pointer, value int, bytes uintptr, stream Stream) error {
	st := Status(C.ks_memset_device(dst, C.int(value), C.size_t(bytes), stream.c()))
	return statusError(st, "ks_memset_device")
}

// CopyToDevice is a convenience wrapper copying a Go byte slice to a device
// buffer (host-to-device). The slice must be at least as large as the
// destination usage; only len(src) bytes are copied.
func CopyToDevice(dst unsafe.Pointer, src []byte, stream Stream) error {
	if len(src) == 0 {
		return nil
	}
	return Memcpy(dst, unsafe.Pointer(&src[0]), uintptr(len(src)),
		MemcpyHostToDevice, stream)
}

// CopyFromDevice is a convenience wrapper copying a device buffer into a Go byte
// slice (device-to-host). len(dst) bytes are copied.
func CopyFromDevice(dst []byte, src unsafe.Pointer, stream Stream) error {
	if len(dst) == 0 {
		return nil
	}
	return Memcpy(unsafe.Pointer(&dst[0]), src, uintptr(len(dst)),
		MemcpyDeviceToHost, stream)
}
