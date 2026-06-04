package kernelset

/*
#include <stdlib.h>
#include "kernel_set/kernel_set.h"
*/
import "C"

import "unsafe"

// AdamW runs a fused AdamW (decoupled weight decay) step over n elements in
// place. step is the 1-based iteration for bias correction. masterParam may be
// nil; if set it holds the fp32 master copy that is updated and cast back into
// param. expAvg and expAvgSq are fp32 state device pointers. Wraps ks_adamw.
func AdamW(param, masterParam, grad, expAvg, expAvgSq unsafe.Pointer,
	lr, beta1, beta2, eps, weightDecay float32, step int64, gradScale float32,
	n int64, dtype Dtype, stream Stream) error {
	st := Status(C.ks_adamw(param, masterParam, grad,
		(*C.float)(expAvg), (*C.float)(expAvgSq),
		C.float(lr), C.float(beta1), C.float(beta2), C.float(eps),
		C.float(weightDecay), C.int64_t(step), C.float(gradScale),
		C.int64_t(n), C.ks_dtype_t(dtype), stream.c()))
	return statusError(st, "ks_adamw")
}

// SGDMomentum runs a fused SGD step with (optional Nesterov) momentum and weight
// decay over n elements in place. masterParam may be nil; momentum is a fp32
// state device pointer. Wraps ks_sgd_momentum.
func SGDMomentum(param, masterParam, grad, momentum unsafe.Pointer,
	lr, momentumFactor, weightDecay float32, nesterov bool, gradScale float32,
	n int64, dtype Dtype, stream Stream) error {
	st := Status(C.ks_sgd_momentum(param, masterParam, grad, (*C.float)(momentum),
		C.float(lr), C.float(momentumFactor), C.float(weightDecay),
		boolToCInt(nesterov), C.float(gradScale),
		C.int64_t(n), C.ks_dtype_t(dtype), stream.c()))
	return statusError(st, "ks_sgd_momentum")
}

// GlobalGradNorm computes the global L2 norm of a set of grad tensors (for
// gradient clipping). outNorm is a fp32 [1] device scalar = sqrt(sum ||g||^2).
// grads holds one device pointer per tensor and sizes holds their element
// counts; the two slices must be the same length. Wraps ks_global_grad_norm.
func GlobalGradNorm(outNorm unsafe.Pointer, grads []unsafe.Pointer, sizes []int64,
	dtype Dtype, stream Stream) error {
	n := len(grads)
	if n == 0 {
		// Nothing to reduce; ask the kernel anyway with a zero count so it can
		// validate/zero the output. Pass nil arrays.
		st := Status(C.ks_global_grad_norm((*C.float)(outNorm), nil, nil,
			C.int(0), C.ks_dtype_t(dtype), stream.c()))
		return statusError(st, "ks_global_grad_norm")
	}

	// Marshal the Go slices into C arrays. grads is `const void* const*`, sizes
	// is `const int64_t*`. We allocate C memory so the pointers passed to cgo do
	// not themselves contain Go pointers (cgo pointer-passing rules).
	cGrads := C.malloc(C.size_t(n) * C.size_t(unsafe.Sizeof(uintptr(0))))
	defer C.free(cGrads)
	cSizes := C.malloc(C.size_t(n) * C.size_t(unsafe.Sizeof(C.int64_t(0))))
	defer C.free(cSizes)

	gradSlice := unsafe.Slice((*unsafe.Pointer)(cGrads), n)
	sizeSlice := unsafe.Slice((*C.int64_t)(cSizes), n)
	for i := 0; i < n; i++ {
		gradSlice[i] = grads[i]
		var sz int64
		if i < len(sizes) {
			sz = sizes[i]
		}
		sizeSlice[i] = C.int64_t(sz)
	}

	st := Status(C.ks_global_grad_norm((*C.float)(outNorm),
		(*unsafe.Pointer)(cGrads), (*C.int64_t)(cSizes),
		C.int(n), C.ks_dtype_t(dtype), stream.c()))
	return statusError(st, "ks_global_grad_norm")
}
