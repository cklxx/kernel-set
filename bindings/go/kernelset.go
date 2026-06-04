// Package kernelset provides idiomatic Go bindings for kernel-set, a collection
// of high-performance CUDA/HIP kernels for LLM inference and training exposed
// through a stable, pure-C ABI (see include/kernel_set/*.h).
//
// The bindings use cgo to call directly into the shared library
// (libkernel_set.so / libkernel_set.dylib / kernel_set.dll). Every kernel takes
// device pointers (as unsafe.Pointer) and an optional Stream, and returns a Go
// error built from the C status/last-error strings.
//
// # Linking & discovery
//
// This file's cgo directives add the in-tree include dir (../../include) to the
// compiler search path and link against -lkernel_set. To build and run against
// a library installed elsewhere, set the standard cgo / loader environment
// variables before `go build` / `go test` / running your program:
//
//	# point the compiler at the headers (if not in the default tree):
//	export CGO_CFLAGS="-I/path/to/kernel-set/include"
//	# point the linker at the .so/.dylib at build time:
//	export CGO_LDFLAGS="-L/path/to/lib"
//	# point the dynamic loader at it at run time:
//	export LD_LIBRARY_PATH=/path/to/lib       # Linux
//	export DYLD_LIBRARY_PATH=/path/to/lib     # macOS
//
// kernel-set itself documents KERNEL_SET_LIB as the install location of the
// shared library; LibPath() reads it and other standard locations so callers can
// log/verify where the library is expected to live. See README.md for details.
package kernelset

/*
#cgo CFLAGS: -I${SRCDIR}/../../include
#cgo LDFLAGS: -lkernel_set

#include <stdlib.h>
#include <string.h>
#include "kernel_set/kernel_set.h"
*/
import "C"

import (
	"fmt"
	"unsafe"
)

// ---------------------------------------------------------------------------
// Enums mirroring the C ABI (include/kernel_set/types.h).
// ---------------------------------------------------------------------------

// Status mirrors ks_status_t. Every C entry point returns one of these; the Go
// wrappers convert non-success codes into errors (see statusError).
type Status int

const (
	StatusSuccess          Status = C.KS_SUCCESS
	StatusInvalidArgument  Status = C.KS_ERROR_INVALID_ARGUMENT
	StatusUnsupportedDtype Status = C.KS_ERROR_UNSUPPORTED_DTYPE
	StatusUnsupportedShape Status = C.KS_ERROR_UNSUPPORTED_SHAPE
	StatusCUDA             Status = C.KS_ERROR_CUDA
	StatusNotImplemented   Status = C.KS_ERROR_NOT_IMPLEMENTED
	StatusOutOfMemory      Status = C.KS_ERROR_OUT_OF_MEMORY
	StatusArchUnsupported  Status = C.KS_ERROR_ARCH_UNSUPPORTED
	StatusInternal         Status = C.KS_ERROR_INTERNAL
)

// String returns the human-readable name reported by ks_status_string.
func (s Status) String() string {
	return C.GoString(C.ks_status_string(C.ks_status_t(s)))
}

// Dtype mirrors ks_dtype_t — the numeric element type of a device buffer.
type Dtype int

const (
	F32    Dtype = C.KS_DTYPE_F32
	F16    Dtype = C.KS_DTYPE_F16
	BF16   Dtype = C.KS_DTYPE_BF16
	F8E4M3 Dtype = C.KS_DTYPE_F8E4M3
	F8E5M2 Dtype = C.KS_DTYPE_F8E5M2
	F64    Dtype = C.KS_DTYPE_F64
	I64    Dtype = C.KS_DTYPE_I64
	I32    Dtype = C.KS_DTYPE_I32
	I8     Dtype = C.KS_DTYPE_I8
	U8     Dtype = C.KS_DTYPE_U8
	I4     Dtype = C.KS_DTYPE_I4
)

// Name returns the short token for the dtype (e.g. "bf16") via ks_dtype_name.
func (d Dtype) Name() string {
	return C.GoString(C.ks_dtype_name(C.ks_dtype_t(d)))
}

// SizeBits returns the element size in bits via ks_dtype_size_bits.
func (d Dtype) SizeBits() int {
	return int(C.ks_dtype_size_bits(C.ks_dtype_t(d)))
}

// String reports the dtype's short token.
func (d Dtype) String() string { return d.Name() }

// Activation mirrors ks_activation_t for fused GEMM epilogues.
type Activation int

const (
	ActNone     Activation = C.KS_ACT_NONE
	ActReLU     Activation = C.KS_ACT_RELU
	ActGELU     Activation = C.KS_ACT_GELU
	ActGELUTanh Activation = C.KS_ACT_GELU_TANH
	ActSiLU     Activation = C.KS_ACT_SILU
)

// QuantMode mirrors ks_quant_mode_t — quantization granularity.
type QuantMode int

const (
	QuantPerTensor  QuantMode = C.KS_QUANT_PER_TENSOR
	QuantPerToken   QuantMode = C.KS_QUANT_PER_TOKEN
	QuantPerChannel QuantMode = C.KS_QUANT_PER_CHANNEL
	QuantGroupwise  QuantMode = C.KS_QUANT_GROUPWISE
)

// MemcpyKind mirrors ks_memcpy_kind_t for runtime memory transfers.
type MemcpyKind int

const (
	MemcpyHostToDevice   MemcpyKind = C.KS_MEMCPY_HOST_TO_DEVICE
	MemcpyDeviceToHost   MemcpyKind = C.KS_MEMCPY_DEVICE_TO_HOST
	MemcpyDeviceToDevice MemcpyKind = C.KS_MEMCPY_DEVICE_TO_DEVICE
)

// ---------------------------------------------------------------------------
// Stream
// ---------------------------------------------------------------------------

// Stream is an opaque GPU stream handle (cudaStream_t / hipStream_t). The zero
// value is the default stream, exactly like passing NULL to the C ABI. Wrap a
// stream you obtained from another library (e.g. a CUDA runtime call or a tensor
// framework) with StreamFromPtr.
type Stream struct {
	ptr unsafe.Pointer
}

// DefaultStream is the default GPU stream (NULL in the C ABI). It is the zero
// value of Stream, so you may also pass kernelset.Stream{}.
var DefaultStream = Stream{}

// StreamFromPtr wraps a raw stream handle (cudaStream_t / hipStream_t) obtained
// elsewhere — e.g. torch.cuda.current_stream().cuda_stream as a uintptr, or a
// pointer from the CUDA driver/runtime.
func StreamFromPtr(p unsafe.Pointer) Stream { return Stream{ptr: p} }

// StreamFromUintptr wraps a stream handle given as an integer address.
func StreamFromUintptr(addr uintptr) Stream { return Stream{ptr: unsafe.Pointer(addr)} }

// Ptr returns the raw stream handle.
func (s Stream) Ptr() unsafe.Pointer { return s.ptr }

// c returns the C stream handle.
func (s Stream) c() C.ks_stream_t { return C.ks_stream_t(s.ptr) }

// Create allocates a new GPU stream via ks_stream_create.
func (s *Stream) Create() error {
	var raw C.ks_stream_t
	st := Status(C.ks_stream_create(&raw))
	if err := statusError(st, "ks_stream_create"); err != nil {
		return err
	}
	s.ptr = unsafe.Pointer(raw)
	return nil
}

// NewStream creates and returns a fresh GPU stream.
func NewStream() (Stream, error) {
	var s Stream
	return s, s.Create()
}

// Destroy releases the stream via ks_stream_destroy.
func (s Stream) Destroy() error {
	return statusError(Status(C.ks_stream_destroy(s.c())), "ks_stream_destroy")
}

// Synchronize blocks until all work on the stream has completed.
func (s Stream) Synchronize() error {
	return statusError(Status(C.ks_stream_synchronize(s.c())), "ks_stream_synchronize")
}

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

// Error is the structured error returned by kernel wrappers on a non-success
// status. It carries the failing operation name, the Status code, and the
// thread-local backend message captured from ks_last_error_string.
type Error struct {
	Op      string // kernel-set C function that failed (e.g. "ks_rms_norm")
	Status  Status // status code returned by the C ABI
	Backend string // ks_last_error_string() message, if any
}

func (e *Error) Error() string {
	if e.Backend != "" {
		return fmt.Sprintf("kernelset: %s: %s (%s): %s",
			e.Op, e.Status.String(), statusCodeName(e.Status), e.Backend)
	}
	return fmt.Sprintf("kernelset: %s: %s (%s)",
		e.Op, e.Status.String(), statusCodeName(e.Status))
}

// Is allows errors.Is(err, someStatus-wrapped error) style matching by Status.
func (e *Error) Is(target error) bool {
	t, ok := target.(*Error)
	if !ok {
		return false
	}
	return e.Status == t.Status
}

// statusError converts a Status into a Go error (nil on success). On failure it
// captures the thread-local backend message; this must be called immediately
// after the C call that produced the status.
func statusError(s Status, op string) error {
	if s == StatusSuccess {
		return nil
	}
	backend := C.GoString(C.ks_last_error_string())
	return &Error{Op: op, Status: s, Backend: backend}
}

func statusCodeName(s Status) string {
	switch s {
	case StatusSuccess:
		return "KS_SUCCESS"
	case StatusInvalidArgument:
		return "KS_ERROR_INVALID_ARGUMENT"
	case StatusUnsupportedDtype:
		return "KS_ERROR_UNSUPPORTED_DTYPE"
	case StatusUnsupportedShape:
		return "KS_ERROR_UNSUPPORTED_SHAPE"
	case StatusCUDA:
		return "KS_ERROR_CUDA"
	case StatusNotImplemented:
		return "KS_ERROR_NOT_IMPLEMENTED"
	case StatusOutOfMemory:
		return "KS_ERROR_OUT_OF_MEMORY"
	case StatusArchUnsupported:
		return "KS_ERROR_ARCH_UNSUPPORTED"
	case StatusInternal:
		return "KS_ERROR_INTERNAL"
	default:
		return "KS_ERROR_UNKNOWN"
	}
}

// LastErrorString returns the thread-local backend error message for the last
// kernel-set call on this thread (ks_last_error_string). Usually you want the
// error returned by a wrapper instead; this is exposed for advanced diagnostics.
func LastErrorString() string {
	return C.GoString(C.ks_last_error_string())
}

// ---------------------------------------------------------------------------
// Pointer helpers
// ---------------------------------------------------------------------------

// boolToCInt converts a Go bool into the 0/1 int flag the C ABI expects.
func boolToCInt(b bool) C.int {
	if b {
		return 1
	}
	return 0
}
