/**
 * kernel-set — ergonomic, typed TypeScript API.
 *
 * Wraps the raw FFI declarations in ./lib.ts. Every wrapper:
 *   - accepts device pointers as `bigint | number` (raw addresses; see DevPtr),
 *   - accepts a stream as `Stream` (a pointer, `0`, or `null` => default stream),
 *   - checks the returned `ks_status_t` and throws `KernelSetError` on non-zero,
 *   - returns plain JS values where the C function uses out-parameters.
 *
 * The library performs NO copying of device data — pointers are addresses you
 * obtained from a tensor library (PyTorch `tensor.data_ptr()`, etc.) or from the
 * runtime helpers (`mallocDevice`). See README.md "Interop" for details.
 */
import * as raw from './lib';
import { koffi } from './lib';

// ---------------------------------------------------------------------------
// Core types
// ---------------------------------------------------------------------------

/**
 * A raw device pointer (GPU address) or a stream handle. We accept both
 * `bigint` (full 64-bit address; recommended) and `number` (safe for values
 * below 2^53). `null`/`0` means a null pointer / default stream where allowed.
 */
export type DevPtr = bigint | number;

/** A GPU stream handle. `0`, `null`, or `undefined` selects the default stream. */
export type Stream = DevPtr | null | undefined;

/** Normalize a pointer-ish value to what koffi expects for a `void *` arg. */
function ptr(p: DevPtr | null | undefined): bigint | number {
  if (p === null || p === undefined) return 0;
  return p;
}

/** Normalize a stream to a pointer value (0 == default stream). */
function stream(s: Stream): bigint | number {
  return ptr(s);
}

// ---------------------------------------------------------------------------
// Enums — mirror kernel_set/types.h and runtime.h
// ---------------------------------------------------------------------------

/** ks_status_t — return code of every entry point (0 == Success). */
export enum Status {
  Success = 0,
  InvalidArgument = 1,
  UnsupportedDtype = 2,
  UnsupportedShape = 3,
  Cuda = 4,
  NotImplemented = 5,
  OutOfMemory = 6,
  ArchUnsupported = 7,
  Internal = 8,
}

/** ks_dtype_t — numeric element types understood across the ABI. */
export enum Dtype {
  F32 = 0,
  F16 = 1,
  BF16 = 2,
  F8E4M3 = 3,
  F8E5M2 = 4,
  F64 = 5,
  I64 = 6,
  I32 = 7,
  I8 = 8,
  U8 = 9,
  I4 = 10,
}

/** ks_activation_t — fused activation selectors. */
export enum Activation {
  None = 0,
  Relu = 1,
  Gelu = 2,
  GeluTanh = 3,
  Silu = 4,
}

/** ks_quant_mode_t — quantization granularity. */
export enum QuantMode {
  PerTensor = 0,
  PerToken = 1,
  PerChannel = 2,
  Groupwise = 3,
}

/** ks_memcpy_kind_t — direction for `memcpy`. */
export enum MemcpyKind {
  HostToDevice = 0,
  DeviceToHost = 1,
  DeviceToDevice = 2,
}

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

/** Thrown when a kernel-set entry point returns a non-zero status. */
export class KernelSetError extends Error {
  /** The numeric ks_status_t value. */
  readonly status: Status;
  /** The C function name that failed. */
  readonly fn: string;
  /** The backend's thread-local last-error message, if any. */
  readonly backendMessage: string;

  constructor(status: Status, fn: string) {
    const statusName = raw.ks_status_string(status) ?? `status ${status}`;
    let backendMsg = '';
    try {
      backendMsg = raw.ks_last_error_string() ?? '';
    } catch {
      backendMsg = '';
    }
    const detail = backendMsg ? `: ${backendMsg}` : '';
    super(`${fn} failed (${statusName} [${status}])${detail}`);
    this.name = 'KernelSetError';
    this.status = status;
    this.fn = fn;
    this.backendMessage = backendMsg;
  }
}

/** Throw a KernelSetError unless `status` is Success. */
function check(status: number, fn: string): void {
  if (status !== Status.Success) {
    throw new KernelSetError(status as Status, fn);
  }
}

// ===========================================================================
// runtime.h
// ===========================================================================

/** Semantic version string of the loaded library, e.g. "0.1.0". */
export function version(): string {
  return raw.ks_version();
}

/** Human-readable name for a status code. */
export function statusString(status: Status): string {
  return raw.ks_status_string(status);
}

/** Element size in bits (e.g. I4 -> 4, F32 -> 32). */
export function dtypeSizeBits(dtype: Dtype): number {
  return raw.ks_dtype_size_bits(dtype);
}

/** Short token for a dtype, e.g. "bf16". */
export function dtypeName(dtype: Dtype): string {
  return raw.ks_dtype_name(dtype);
}

/** The backend this build targets: "cuda", "hip", or "none". */
export function backendName(): string {
  return raw.ks_backend_name();
}

/** The thread-local message describing the last backend runtime error. */
export function lastErrorString(): string {
  return raw.ks_last_error_string();
}

/** Number of visible GPU devices. */
export function deviceCount(): number {
  const out = [0];
  check(raw.ks_device_count(out), 'ks_device_count');
  return out[0];
}

/** Select the active device. */
export function setDevice(device: number): void {
  check(raw.ks_set_device(device), 'ks_set_device');
}

/** Get the active device index. */
export function getDevice(): number {
  const out = [0];
  check(raw.ks_get_device(out), 'ks_get_device');
  return out[0];
}

/** Decoded device properties (mirrors ks_device_properties_t). */
export interface DeviceProperties {
  name: string;
  computeMajor: number;
  computeMinor: number;
  multiprocessorCount: number;
  maxThreadsPerBlock: number;
  maxSharedMemoryPerBlock: number;
  warpSize: number;
  totalGlobalMemory: bigint;
  supportsBf16: boolean;
  supportsFp8: boolean;
  supportsTf32: boolean;
}

/** Query properties of the given device index. */
export function getDeviceProperties(device: number): DeviceProperties {
  const out: Record<string, unknown> = {};
  check(raw.ks_get_device_properties(device, out), 'ks_get_device_properties');
  // koffi decodes `char[256]` to a JS string array of code points; join + trim.
  const rawName = out.name as unknown;
  let name: string;
  if (typeof rawName === 'string') {
    name = rawName.replace(/\0.*$/, '');
  } else if (Array.isArray(rawName)) {
    name = String.fromCharCode(...(rawName as number[])).replace(/\0.*$/, '');
  } else {
    name = String(rawName ?? '');
  }
  const total = out.total_global_memory;
  return {
    name,
    computeMajor: Number(out.compute_major),
    computeMinor: Number(out.compute_minor),
    multiprocessorCount: Number(out.multiprocessor_count),
    maxThreadsPerBlock: Number(out.max_threads_per_block),
    maxSharedMemoryPerBlock: Number(out.max_shared_memory_per_block),
    warpSize: Number(out.warp_size),
    totalGlobalMemory: typeof total === 'bigint' ? total : BigInt(Number(total)),
    supportsBf16: Number(out.supports_bf16) !== 0,
    supportsFp8: Number(out.supports_fp8) !== 0,
    supportsTf32: Number(out.supports_tf32) !== 0,
  };
}

/** Create a new stream; returns its pointer as a bigint. */
export function streamCreate(): bigint {
  const out: [unknown] = [null];
  check(raw.ks_stream_create(out), 'ks_stream_create');
  return toBigIntPtr(out[0]);
}

/** Destroy a previously created stream. */
export function streamDestroy(s: DevPtr): void {
  check(raw.ks_stream_destroy(ptr(s)), 'ks_stream_destroy');
}

/** Block until all work on `s` (or the default stream) completes. */
export function streamSynchronize(s: Stream = 0): void {
  check(raw.ks_stream_synchronize(stream(s)), 'ks_stream_synchronize');
}

/** Allocate `bytes` of device memory; returns the address as a bigint. */
export function mallocDevice(bytes: bigint | number): bigint {
  const out: [unknown] = [null];
  check(raw.ks_malloc_device(out, bytes), 'ks_malloc_device');
  return toBigIntPtr(out[0]);
}

/** Free device memory previously returned by mallocDevice. */
export function freeDevice(p: DevPtr): void {
  check(raw.ks_free_device(ptr(p)), 'ks_free_device');
}

/**
 * Copy `bytes` between host/device buffers. For host endpoints you may pass a
 * Node Buffer/TypedArray (koffi marshals it); for device endpoints pass a
 * raw address (bigint/number).
 */
export function memcpy(
  dst: DevPtr | Buffer | ArrayBufferView,
  src: DevPtr | Buffer | ArrayBufferView,
  bytes: bigint | number,
  kind: MemcpyKind,
  s: Stream = 0,
): void {
  check(
    raw.ks_memcpy(dst as never, src as never, bytes, kind, stream(s)),
    'ks_memcpy',
  );
}

/** Fill `bytes` of device memory at `dst` with the byte `value`. */
export function memsetDevice(
  dst: DevPtr,
  value: number,
  bytes: bigint | number,
  s: Stream = 0,
): void {
  check(raw.ks_memset_device(ptr(dst), value, bytes, stream(s)), 'ks_memset_device');
}

/** Best-effort conversion of a koffi pointer output to a bigint address. */
function toBigIntPtr(p: unknown): bigint {
  if (typeof p === 'bigint') return p;
  if (typeof p === 'number') return BigInt(p);
  if (p === null || p === undefined) return 0n;
  // koffi may return an external pointer object; decode its address.
  try {
    const addr = koffi.address(p as never);
    return typeof addr === 'bigint' ? addr : BigInt(addr as number);
  } catch {
    return 0n;
  }
}

// ===========================================================================
// norm.h
// ===========================================================================

/** out = (x / rms(x)) * weight. */
export function rmsNorm(
  out: DevPtr,
  input: DevPtr,
  weight: DevPtr,
  rows: bigint | number,
  cols: bigint | number,
  eps: number,
  dtype: Dtype,
  s: Stream = 0,
): void {
  check(
    raw.ks_rms_norm(ptr(out), ptr(input), ptr(weight), rows, cols, eps, dtype, stream(s)),
    'ks_rms_norm',
  );
}

/** Fused add + RMSNorm. residualOut <- input + residual; out <- rmsnorm(.)*w. */
export function rmsNormResidual(
  out: DevPtr,
  residualOut: DevPtr,
  input: DevPtr,
  residual: DevPtr,
  weight: DevPtr,
  rows: bigint | number,
  cols: bigint | number,
  eps: number,
  dtype: Dtype,
  s: Stream = 0,
): void {
  check(
    raw.ks_rms_norm_residual(
      ptr(out), ptr(residualOut), ptr(input), ptr(residual), ptr(weight),
      rows, cols, eps, dtype, stream(s),
    ),
    'ks_rms_norm_residual',
  );
}

/** out = (x - mean)/sqrt(var+eps) * weight + bias. `bias` may be null. */
export function layerNorm(
  out: DevPtr,
  input: DevPtr,
  weight: DevPtr,
  bias: DevPtr | null,
  rows: bigint | number,
  cols: bigint | number,
  eps: number,
  dtype: Dtype,
  s: Stream = 0,
): void {
  check(
    raw.ks_layer_norm(ptr(out), ptr(input), ptr(weight), ptr(bias), rows, cols, eps, dtype, stream(s)),
    'ks_layer_norm',
  );
}

/** RMSNorm backward; gradWeightFp32 is a [cols] fp32 buffer. */
export function rmsNormBackward(
  gradInput: DevPtr,
  gradWeightFp32: DevPtr,
  gradOut: DevPtr,
  input: DevPtr,
  weight: DevPtr,
  rows: bigint | number,
  cols: bigint | number,
  eps: number,
  dtype: Dtype,
  s: Stream = 0,
): void {
  check(
    raw.ks_rms_norm_backward(
      ptr(gradInput), ptr(gradWeightFp32), ptr(gradOut), ptr(input), ptr(weight),
      rows, cols, eps, dtype, stream(s),
    ),
    'ks_rms_norm_backward',
  );
}

/** LayerNorm backward. gradBiasFp32 may be null if no bias. */
export function layerNormBackward(
  gradInput: DevPtr,
  gradWeightFp32: DevPtr,
  gradBiasFp32: DevPtr | null,
  gradOut: DevPtr,
  input: DevPtr,
  weight: DevPtr,
  rows: bigint | number,
  cols: bigint | number,
  eps: number,
  dtype: Dtype,
  s: Stream = 0,
): void {
  check(
    raw.ks_layer_norm_backward(
      ptr(gradInput), ptr(gradWeightFp32), ptr(gradBiasFp32), ptr(gradOut),
      ptr(input), ptr(weight), rows, cols, eps, dtype, stream(s),
    ),
    'ks_layer_norm_backward',
  );
}

// ===========================================================================
// activation.h
// ===========================================================================

/** out = silu(x). */
export function silu(out: DevPtr, input: DevPtr, n: bigint | number, dtype: Dtype, s: Stream = 0): void {
  check(raw.ks_silu(ptr(out), ptr(input), n, dtype, stream(s)), 'ks_silu');
}

/** out = gelu(x). `tanhApprox` selects the tanh approximation. */
export function gelu(
  out: DevPtr,
  input: DevPtr,
  n: bigint | number,
  tanhApprox: boolean,
  dtype: Dtype,
  s: Stream = 0,
): void {
  check(raw.ks_gelu(ptr(out), ptr(input), n, tanhApprox ? 1 : 0, dtype, stream(s)), 'ks_gelu');
}

/** out = relu(x). */
export function relu(out: DevPtr, input: DevPtr, n: bigint | number, dtype: Dtype, s: Stream = 0): void {
  check(raw.ks_relu(ptr(out), ptr(input), n, dtype, stream(s)), 'ks_relu');
}

/** out = silu(gate) * up. */
export function swiglu(
  out: DevPtr,
  gate: DevPtr,
  up: DevPtr,
  rows: bigint | number,
  inter: bigint | number,
  dtype: Dtype,
  s: Stream = 0,
): void {
  check(raw.ks_swiglu(ptr(out), ptr(gate), ptr(up), rows, inter, dtype, stream(s)), 'ks_swiglu');
}

/** Packed SwiGLU: input is [rows, 2*inter] (gate || up). */
export function swigluPacked(
  out: DevPtr,
  input: DevPtr,
  rows: bigint | number,
  inter: bigint | number,
  dtype: Dtype,
  s: Stream = 0,
): void {
  check(raw.ks_swiglu_packed(ptr(out), ptr(input), rows, inter, dtype, stream(s)), 'ks_swiglu_packed');
}

/** out = gelu(gate) * up (GeGLU). */
export function geglu(
  out: DevPtr,
  gate: DevPtr,
  up: DevPtr,
  rows: bigint | number,
  inter: bigint | number,
  tanhApprox: boolean,
  dtype: Dtype,
  s: Stream = 0,
): void {
  check(
    raw.ks_geglu(ptr(out), ptr(gate), ptr(up), rows, inter, tanhApprox ? 1 : 0, dtype, stream(s)),
    'ks_geglu',
  );
}

/** SwiGLU backward -> gradGate, gradUp. */
export function swigluBackward(
  gradGate: DevPtr,
  gradUp: DevPtr,
  gradOut: DevPtr,
  gate: DevPtr,
  up: DevPtr,
  rows: bigint | number,
  inter: bigint | number,
  dtype: Dtype,
  s: Stream = 0,
): void {
  check(
    raw.ks_swiglu_backward(ptr(gradGate), ptr(gradUp), ptr(gradOut), ptr(gate), ptr(up), rows, inter, dtype, stream(s)),
    'ks_swiglu_backward',
  );
}

// ===========================================================================
// attention.h
// ===========================================================================

/** Variable-length packed prefill (FlashAttention-2 forward). lse may be null. */
export function flashAttnVarlen(args: {
  out: DevPtr;
  softmaxLse?: DevPtr | null;
  q: DevPtr;
  k: DevPtr;
  v: DevPtr;
  cuSeqlensQ: DevPtr;
  cuSeqlensK: DevPtr;
  batch: number;
  maxSeqlenQ: number;
  maxSeqlenK: number;
  numHeads: number;
  numKvHeads: number;
  headDim: number;
  softmaxScale?: number;
  causal?: boolean;
  dtype: Dtype;
  stream?: Stream;
}): void {
  check(
    raw.ks_flash_attn_varlen(
      ptr(args.out), ptr(args.softmaxLse ?? 0), ptr(args.q), ptr(args.k), ptr(args.v),
      ptr(args.cuSeqlensQ), ptr(args.cuSeqlensK), args.batch, args.maxSeqlenQ, args.maxSeqlenK,
      args.numHeads, args.numKvHeads, args.headDim, args.softmaxScale ?? -1,
      args.causal ? 1 : 0, args.dtype, stream(args.stream),
    ),
    'ks_flash_attn_varlen',
  );
}

/** Dense (non-varlen) prefill for a uniform [batch, seqlen] shape. */
export function flashAttn(args: {
  out: DevPtr;
  softmaxLse?: DevPtr | null;
  q: DevPtr;
  k: DevPtr;
  v: DevPtr;
  batch: number;
  seqlenQ: number;
  seqlenK: number;
  numHeads: number;
  numKvHeads: number;
  headDim: number;
  softmaxScale?: number;
  causal?: boolean;
  dtype: Dtype;
  stream?: Stream;
}): void {
  check(
    raw.ks_flash_attn(
      ptr(args.out), ptr(args.softmaxLse ?? 0), ptr(args.q), ptr(args.k), ptr(args.v),
      args.batch, args.seqlenQ, args.seqlenK, args.numHeads, args.numKvHeads, args.headDim,
      args.softmaxScale ?? -1, args.causal ? 1 : 0, args.dtype, stream(args.stream),
    ),
    'ks_flash_attn',
  );
}

/** Paged KV-cache decode (FlashDecoding). */
export function pagedAttnDecode(args: {
  out: DevPtr;
  q: DevPtr;
  kCache: DevPtr;
  vCache: DevPtr;
  blockTables: DevPtr;
  seqLens: DevPtr;
  numSeqs: number;
  numHeads: number;
  numKvHeads: number;
  headDim: number;
  blockSize: number;
  maxBlocksPerSeq: number;
  softmaxScale?: number;
  dtype: Dtype;
  stream?: Stream;
}): void {
  check(
    raw.ks_paged_attn_decode(
      ptr(args.out), ptr(args.q), ptr(args.kCache), ptr(args.vCache), ptr(args.blockTables),
      ptr(args.seqLens), args.numSeqs, args.numHeads, args.numKvHeads, args.headDim,
      args.blockSize, args.maxBlocksPerSeq, args.softmaxScale ?? -1, args.dtype, stream(args.stream),
    ),
    'ks_paged_attn_decode',
  );
}

/** Append new K/V into a paged cache at the given slots. */
export function reshapeAndCache(args: {
  kCache: DevPtr;
  vCache: DevPtr;
  key: DevPtr;
  value: DevPtr;
  slotMapping: DevPtr;
  numTokens: number;
  numKvHeads: number;
  headDim: number;
  blockSize: number;
  dtype: Dtype;
  stream?: Stream;
}): void {
  check(
    raw.ks_reshape_and_cache(
      ptr(args.kCache), ptr(args.vCache), ptr(args.key), ptr(args.value), ptr(args.slotMapping),
      args.numTokens, args.numKvHeads, args.headDim, args.blockSize, args.dtype, stream(args.stream),
    ),
    'ks_reshape_and_cache',
  );
}

/** DeepSeek Multi-head Latent Attention decode over a compressed KV cache. */
export function mlaDecode(args: {
  out: DevPtr;
  qNope: DevPtr;
  qPe: DevPtr;
  kvCache: DevPtr;
  blockTables: DevPtr;
  seqLens: DevPtr;
  numSeqs: number;
  numHeads: number;
  kvLoraRank: number;
  ropeDim: number;
  blockSize: number;
  maxBlocksPerSeq: number;
  softmaxScale?: number;
  dtype: Dtype;
  stream?: Stream;
}): void {
  check(
    raw.ks_mla_decode(
      ptr(args.out), ptr(args.qNope), ptr(args.qPe), ptr(args.kvCache), ptr(args.blockTables),
      ptr(args.seqLens), args.numSeqs, args.numHeads, args.kvLoraRank, args.ropeDim,
      args.blockSize, args.maxBlocksPerSeq, args.softmaxScale ?? -1, args.dtype, stream(args.stream),
    ),
    'ks_mla_decode',
  );
}

/** FlashAttention backward (training). */
export function flashAttnBackward(args: {
  gradQ: DevPtr;
  gradK: DevPtr;
  gradV: DevPtr;
  gradOut: DevPtr;
  q: DevPtr;
  k: DevPtr;
  v: DevPtr;
  out: DevPtr;
  softmaxLse: DevPtr;
  batch: number;
  seqlenQ: number;
  seqlenK: number;
  numHeads: number;
  numKvHeads: number;
  headDim: number;
  softmaxScale?: number;
  causal?: boolean;
  dtype: Dtype;
  stream?: Stream;
}): void {
  check(
    raw.ks_flash_attn_backward(
      ptr(args.gradQ), ptr(args.gradK), ptr(args.gradV), ptr(args.gradOut), ptr(args.q),
      ptr(args.k), ptr(args.v), ptr(args.out), ptr(args.softmaxLse), args.batch, args.seqlenQ,
      args.seqlenK, args.numHeads, args.numKvHeads, args.headDim, args.softmaxScale ?? -1,
      args.causal ? 1 : 0, args.dtype, stream(args.stream),
    ),
    'ks_flash_attn_backward',
  );
}

// ===========================================================================
// gemm.h
// ===========================================================================

/** C = alpha * op(A) @ op(B) + beta * C. */
export function gemm(args: {
  c: DevPtr;
  a: DevPtr;
  b: DevPtr;
  m: bigint | number;
  n: bigint | number;
  k: bigint | number;
  transA?: boolean;
  transB?: boolean;
  lda: bigint | number;
  ldb: bigint | number;
  ldc: bigint | number;
  alpha?: number;
  beta?: number;
  dtype: Dtype;
  stream?: Stream;
}): void {
  check(
    raw.ks_gemm(
      ptr(args.c), ptr(args.a), ptr(args.b), args.m, args.n, args.k,
      args.transA ? 1 : 0, args.transB ? 1 : 0, args.lda, args.ldb, args.ldc,
      args.alpha ?? 1, args.beta ?? 0, args.dtype, stream(args.stream),
    ),
    'ks_gemm',
  );
}

/** D = act(alpha * A @ B + bias). bias is [N] or null. */
export function gemmBiasAct(args: {
  d: DevPtr;
  a: DevPtr;
  b: DevPtr;
  bias?: DevPtr | null;
  m: bigint | number;
  n: bigint | number;
  k: bigint | number;
  alpha?: number;
  act?: Activation;
  dtype: Dtype;
  stream?: Stream;
}): void {
  check(
    raw.ks_gemm_bias_act(
      ptr(args.d), ptr(args.a), ptr(args.b), ptr(args.bias ?? 0), args.m, args.n, args.k,
      args.alpha ?? 1, args.act ?? Activation.None, args.dtype, stream(args.stream),
    ),
    'ks_gemm_bias_act',
  );
}

/** Batched / strided-batched GEMM. */
export function gemmBatched(args: {
  c: DevPtr;
  a: DevPtr;
  b: DevPtr;
  batch: bigint | number;
  m: bigint | number;
  n: bigint | number;
  k: bigint | number;
  transA?: boolean;
  transB?: boolean;
  strideA: bigint | number;
  strideB: bigint | number;
  strideC: bigint | number;
  alpha?: number;
  beta?: number;
  dtype: Dtype;
  stream?: Stream;
}): void {
  check(
    raw.ks_gemm_batched(
      ptr(args.c), ptr(args.a), ptr(args.b), args.batch, args.m, args.n, args.k,
      args.transA ? 1 : 0, args.transB ? 1 : 0, args.strideA, args.strideB, args.strideC,
      args.alpha ?? 1, args.beta ?? 0, args.dtype, stream(args.stream),
    ),
    'ks_gemm_batched',
  );
}

/** W8A8 GEMM: int8 A x int8 B -> outDtype C. Scales are fp32 device pointers. */
export function gemmW8a8(args: {
  c: DevPtr;
  aI8: DevPtr;
  bI8: DevPtr;
  aScale: DevPtr;
  bScale: DevPtr;
  bias?: DevPtr | null;
  m: bigint | number;
  n: bigint | number;
  k: bigint | number;
  aMode: QuantMode;
  bMode: QuantMode;
  outDtype: Dtype;
  stream?: Stream;
}): void {
  check(
    raw.ks_gemm_w8a8(
      ptr(args.c), ptr(args.aI8), ptr(args.bI8), ptr(args.aScale), ptr(args.bScale),
      ptr(args.bias ?? 0), args.m, args.n, args.k, args.aMode, args.bMode, args.outDtype, stream(args.stream),
    ),
    'ks_gemm_w8a8',
  );
}

/** W4A16 GEMM: fp16/bf16 A x int4 weights with group-wise scales/zeros. */
export function gemmW4a16(args: {
  c: DevPtr;
  a: DevPtr;
  bPacked: DevPtr;
  scales: DevPtr;
  zeros?: DevPtr | null;
  bias?: DevPtr | null;
  m: bigint | number;
  n: bigint | number;
  k: bigint | number;
  groupSize: number;
  dtype: Dtype;
  stream?: Stream;
}): void {
  check(
    raw.ks_gemm_w4a16(
      ptr(args.c), ptr(args.a), ptr(args.bPacked), ptr(args.scales), ptr(args.zeros ?? 0),
      ptr(args.bias ?? 0), args.m, args.n, args.k, args.groupSize, args.dtype, stream(args.stream),
    ),
    'ks_gemm_w4a16',
  );
}

// ===========================================================================
// moe.h
// ===========================================================================

/** Softmax over experts then top-k select. Weights fp32, indices int32. */
export function moeGateSoftmaxTopk(args: {
  outWeights: DevPtr;
  outIndices: DevPtr;
  logits: DevPtr;
  numTokens: bigint | number;
  numExperts: number;
  topK: number;
  renormalize?: boolean;
  dtype: Dtype;
  stream?: Stream;
}): void {
  check(
    raw.ks_moe_gate_softmax_topk(
      ptr(args.outWeights), ptr(args.outIndices), ptr(args.logits), args.numTokens,
      args.numExperts, args.topK, args.renormalize ? 1 : 0, args.dtype, stream(args.stream),
    ),
    'ks_moe_gate_softmax_topk',
  );
}

/** Sigmoid + group-limited top-k gating (DeepSeek-V3 style). */
export function moeGateSigmoidGroupTopk(args: {
  outWeights: DevPtr;
  outIndices: DevPtr;
  logits: DevPtr;
  correctionBias?: DevPtr | null;
  numTokens: bigint | number;
  numExperts: number;
  nGroup: number;
  topkGroup: number;
  topK: number;
  renormalize?: boolean;
  routedScalingFactor?: number;
  dtype: Dtype;
  stream?: Stream;
}): void {
  check(
    raw.ks_moe_gate_sigmoid_group_topk(
      ptr(args.outWeights), ptr(args.outIndices), ptr(args.logits), ptr(args.correctionBias ?? 0),
      args.numTokens, args.numExperts, args.nGroup, args.topkGroup, args.topK,
      args.renormalize ? 1 : 0, args.routedScalingFactor ?? 1, args.dtype, stream(args.stream),
    ),
    'ks_moe_gate_sigmoid_group_topk',
  );
}

/** Build the permutation that groups tokens by expert. */
export function moeComputePermutation(args: {
  sortedTokenIds: DevPtr;
  expertOffsets: DevPtr;
  topkIndices: DevPtr;
  numTokens: bigint | number;
  numExperts: number;
  topK: number;
  stream?: Stream;
}): void {
  check(
    raw.ks_moe_compute_permutation(
      ptr(args.sortedTokenIds), ptr(args.expertOffsets), ptr(args.topkIndices),
      args.numTokens, args.numExperts, args.topK, stream(args.stream),
    ),
    'ks_moe_compute_permutation',
  );
}

/** Gather rows into permuted layout following sortedTokenIds. */
export function moePermute(args: {
  permuted: DevPtr;
  input: DevPtr;
  sortedTokenIds: DevPtr;
  numTokens: bigint | number;
  topK: number;
  hidden: bigint | number;
  dtype: Dtype;
  stream?: Stream;
}): void {
  check(
    raw.ks_moe_permute(
      ptr(args.permuted), ptr(args.input), ptr(args.sortedTokenIds),
      args.numTokens, args.topK, args.hidden, args.dtype, stream(args.stream),
    ),
    'ks_moe_permute',
  );
}

/** Scatter-add expert outputs back, weighted by routing weights. */
export function moeUnpermute(args: {
  out: DevPtr;
  permuted: DevPtr;
  sortedTokenIds: DevPtr;
  routingWeights: DevPtr;
  numTokens: bigint | number;
  topK: number;
  hidden: bigint | number;
  dtype: Dtype;
  stream?: Stream;
}): void {
  check(
    raw.ks_moe_unpermute(
      ptr(args.out), ptr(args.permuted), ptr(args.sortedTokenIds), ptr(args.routingWeights),
      args.numTokens, args.topK, args.hidden, args.dtype, stream(args.stream),
    ),
    'ks_moe_unpermute',
  );
}

/** Grouped GEMM: per-expert C-block = A-block @ B_e. */
export function moeGroupedGemm(args: {
  c: DevPtr;
  a: DevPtr;
  b: DevPtr;
  expertOffsets: DevPtr;
  numExperts: number;
  totalRows: bigint | number;
  n: bigint | number;
  k: bigint | number;
  dtype: Dtype;
  stream?: Stream;
}): void {
  check(
    raw.ks_moe_grouped_gemm(
      ptr(args.c), ptr(args.a), ptr(args.b), ptr(args.expertOffsets),
      args.numExperts, args.totalRows, args.n, args.k, args.dtype, stream(args.stream),
    ),
    'ks_moe_grouped_gemm',
  );
}

// ===========================================================================
// rope.h
// ===========================================================================

/** In-place RoPE on q and k. k may be null to skip. */
export function ropeInplace(args: {
  q: DevPtr;
  k?: DevPtr | null;
  cos: DevPtr;
  sin: DevPtr;
  numTokens: bigint | number;
  numQHeads: number;
  numKvHeads: number;
  headDim: number;
  interleaved?: boolean;
  dtype: Dtype;
  stream?: Stream;
}): void {
  check(
    raw.ks_rope_inplace(
      ptr(args.q), ptr(args.k ?? 0), ptr(args.cos), ptr(args.sin), args.numTokens,
      args.numQHeads, args.numKvHeads, args.headDim, args.interleaved ? 1 : 0, args.dtype, stream(args.stream),
    ),
    'ks_rope_inplace',
  );
}

/** Out-of-place RoPE -> qOut/kOut. */
export function rope(args: {
  qOut: DevPtr;
  kOut?: DevPtr | null;
  q: DevPtr;
  k?: DevPtr | null;
  cos: DevPtr;
  sin: DevPtr;
  numTokens: bigint | number;
  numQHeads: number;
  numKvHeads: number;
  headDim: number;
  interleaved?: boolean;
  dtype: Dtype;
  stream?: Stream;
}): void {
  check(
    raw.ks_rope(
      ptr(args.qOut), ptr(args.kOut ?? 0), ptr(args.q), ptr(args.k ?? 0), ptr(args.cos), ptr(args.sin),
      args.numTokens, args.numQHeads, args.numKvHeads, args.headDim, args.interleaved ? 1 : 0, args.dtype, stream(args.stream),
    ),
    'ks_rope',
  );
}

/** Gathered RoPE: cos/sin caches indexed by `positions` (int32). */
export function ropeGather(args: {
  q: DevPtr;
  k?: DevPtr | null;
  cosCache: DevPtr;
  sinCache: DevPtr;
  positions: DevPtr;
  numTokens: bigint | number;
  numQHeads: number;
  numKvHeads: number;
  headDim: number;
  interleaved?: boolean;
  dtype: Dtype;
  stream?: Stream;
}): void {
  check(
    raw.ks_rope_gather(
      ptr(args.q), ptr(args.k ?? 0), ptr(args.cosCache), ptr(args.sinCache), ptr(args.positions),
      args.numTokens, args.numQHeads, args.numKvHeads, args.headDim, args.interleaved ? 1 : 0, args.dtype, stream(args.stream),
    ),
    'ks_rope_gather',
  );
}

/** RoPE backward (rotate gradients by the conjugate rotation). */
export function ropeBackward(args: {
  gradQ: DevPtr;
  gradK?: DevPtr | null;
  cos: DevPtr;
  sin: DevPtr;
  numTokens: bigint | number;
  numQHeads: number;
  numKvHeads: number;
  headDim: number;
  interleaved?: boolean;
  dtype: Dtype;
  stream?: Stream;
}): void {
  check(
    raw.ks_rope_backward(
      ptr(args.gradQ), ptr(args.gradK ?? 0), ptr(args.cos), ptr(args.sin), args.numTokens,
      args.numQHeads, args.numKvHeads, args.headDim, args.interleaved ? 1 : 0, args.dtype, stream(args.stream),
    ),
    'ks_rope_backward',
  );
}

// ===========================================================================
// quant.h
// ===========================================================================

/** Dynamic FP8 quantization. `scale` is an fp32 device buffer ([1] or [rows]). */
export function quantizeFp8(args: {
  out: DevPtr;
  scale: DevPtr;
  input: DevPtr;
  rows: bigint | number;
  cols: bigint | number;
  inDtype: Dtype;
  fp8Dtype: Dtype;
  mode: QuantMode;
  stream?: Stream;
}): void {
  check(
    raw.ks_quantize_fp8(
      ptr(args.out), ptr(args.scale), ptr(args.input), args.rows, args.cols,
      args.inDtype, args.fp8Dtype, args.mode, stream(args.stream),
    ),
    'ks_quantize_fp8',
  );
}

/** Dequantize FP8 -> outDtype using fp32 scale. */
export function dequantizeFp8(args: {
  out: DevPtr;
  input: DevPtr;
  scale: DevPtr;
  rows: bigint | number;
  cols: bigint | number;
  outDtype: Dtype;
  fp8Dtype: Dtype;
  mode: QuantMode;
  stream?: Stream;
}): void {
  check(
    raw.ks_dequantize_fp8(
      ptr(args.out), ptr(args.input), ptr(args.scale), args.rows, args.cols,
      args.outDtype, args.fp8Dtype, args.mode, stream(args.stream),
    ),
    'ks_dequantize_fp8',
  );
}

/** Dynamic symmetric INT8 quantization. */
export function quantizeInt8(args: {
  out: DevPtr;
  scale: DevPtr;
  input: DevPtr;
  rows: bigint | number;
  cols: bigint | number;
  inDtype: Dtype;
  mode: QuantMode;
  stream?: Stream;
}): void {
  check(
    raw.ks_quantize_int8(
      ptr(args.out), ptr(args.scale), ptr(args.input), args.rows, args.cols,
      args.inDtype, args.mode, stream(args.stream),
    ),
    'ks_quantize_int8',
  );
}

/** Dequantize INT8 -> outDtype using fp32 scale. */
export function dequantizeInt8(args: {
  out: DevPtr;
  input: DevPtr;
  scale: DevPtr;
  rows: bigint | number;
  cols: bigint | number;
  outDtype: Dtype;
  mode: QuantMode;
  stream?: Stream;
}): void {
  check(
    raw.ks_dequantize_int8(
      ptr(args.out), ptr(args.input), ptr(args.scale), args.rows, args.cols,
      args.outDtype, args.mode, stream(args.stream),
    ),
    'ks_dequantize_int8',
  );
}

/** Dequantize group-wise INT4 weights (AWQ/GPTQ). */
export function dequantizeInt4(args: {
  out: DevPtr;
  qweightPacked: DevPtr;
  scales: DevPtr;
  zeros?: DevPtr | null;
  k: bigint | number;
  n: bigint | number;
  groupSize: number;
  outDtype: Dtype;
  stream?: Stream;
}): void {
  check(
    raw.ks_dequantize_int4(
      ptr(args.out), ptr(args.qweightPacked), ptr(args.scales), ptr(args.zeros ?? 0),
      args.k, args.n, args.groupSize, args.outDtype, stream(args.stream),
    ),
    'ks_dequantize_int4',
  );
}

// ===========================================================================
// sampling.h
// ===========================================================================

/** out = softmax(input / temperature) along the last dim. */
export function softmax(
  out: DevPtr,
  input: DevPtr,
  rows: bigint | number,
  cols: bigint | number,
  temperature: number,
  dtype: Dtype,
  s: Stream = 0,
): void {
  check(raw.ks_softmax(ptr(out), ptr(input), rows, cols, temperature, dtype, stream(s)), 'ks_softmax');
}

/** log_softmax along the last dim. */
export function logSoftmax(
  out: DevPtr,
  input: DevPtr,
  rows: bigint | number,
  cols: bigint | number,
  dtype: Dtype,
  s: Stream = 0,
): void {
  check(raw.ks_log_softmax(ptr(out), ptr(input), rows, cols, dtype, stream(s)), 'ks_log_softmax');
}

/** Greedy argmax over the vocab into outTokens (int32). */
export function argmax(
  outTokens: DevPtr,
  logits: DevPtr,
  numSeqs: bigint | number,
  vocabSize: bigint | number,
  dtype: Dtype,
  s: Stream = 0,
): void {
  check(raw.ks_argmax(ptr(outTokens), ptr(logits), numSeqs, vocabSize, dtype, stream(s)), 'ks_argmax');
}

/** Combined temperature + top-k + top-p sampling. */
export function sample(args: {
  outTokens: DevPtr;
  outProbs?: DevPtr | null;
  logits: DevPtr;
  temperatures?: DevPtr | null;
  topKs?: DevPtr | null;
  topPs?: DevPtr | null;
  numSeqs: bigint | number;
  vocabSize: bigint | number;
  seed: bigint | number;
  philoxOffset: bigint | number;
  dtype: Dtype;
  stream?: Stream;
}): void {
  check(
    raw.ks_sample(
      ptr(args.outTokens), ptr(args.outProbs ?? 0), ptr(args.logits), ptr(args.temperatures ?? 0),
      ptr(args.topKs ?? 0), ptr(args.topPs ?? 0), args.numSeqs, args.vocabSize,
      args.seed, args.philoxOffset, args.dtype, stream(args.stream),
    ),
    'ks_sample',
  );
}

// ===========================================================================
// embedding.h
// ===========================================================================

/** out[i,:] = table[indices[i],:]. indicesI64 selects int64 indices. */
export function embeddingLookup(args: {
  out: DevPtr;
  table: DevPtr;
  indices: DevPtr;
  indicesI64?: boolean;
  numTokens: bigint | number;
  embedDim: bigint | number;
  dtype: Dtype;
  stream?: Stream;
}): void {
  check(
    raw.ks_embedding_lookup(
      ptr(args.out), ptr(args.table), ptr(args.indices), args.indicesI64 ? 1 : 0,
      args.numTokens, args.embedDim, args.dtype, stream(args.stream),
    ),
    'ks_embedding_lookup',
  );
}

/** Atomic scatter-add of grad_out into grad_table (fp32). */
export function embeddingBackward(args: {
  gradTableFp32: DevPtr;
  gradOut: DevPtr;
  indices: DevPtr;
  indicesI64?: boolean;
  numTokens: bigint | number;
  embedDim: bigint | number;
  dtype: Dtype;
  stream?: Stream;
}): void {
  check(
    raw.ks_embedding_backward(
      ptr(args.gradTableFp32), ptr(args.gradOut), ptr(args.indices), args.indicesI64 ? 1 : 0,
      args.numTokens, args.embedDim, args.dtype, stream(args.stream),
    ),
    'ks_embedding_backward',
  );
}

// ===========================================================================
// elementwise.h
// ===========================================================================

/** out = a + b. */
export function add(out: DevPtr, a: DevPtr, b: DevPtr, n: bigint | number, dtype: Dtype, s: Stream = 0): void {
  check(raw.ks_add(ptr(out), ptr(a), ptr(b), n, dtype, stream(s)), 'ks_add');
}

/** out = a * b (Hadamard). */
export function mul(out: DevPtr, a: DevPtr, b: DevPtr, n: bigint | number, dtype: Dtype, s: Stream = 0): void {
  check(raw.ks_mul(ptr(out), ptr(a), ptr(b), n, dtype, stream(s)), 'ks_mul');
}

/** In-place residual accumulation: residual <- residual + x. */
export function addResidual(residual: DevPtr, x: DevPtr, n: bigint | number, dtype: Dtype, s: Stream = 0): void {
  check(raw.ks_add_residual(ptr(residual), ptr(x), n, dtype, stream(s)), 'ks_add_residual');
}

/** out = x * scale (scalar). */
export function scale(out: DevPtr, x: DevPtr, scalar: number, n: bigint | number, dtype: Dtype, s: Stream = 0): void {
  check(raw.ks_scale(ptr(out), ptr(x), scalar, n, dtype, stream(s)), 'ks_scale');
}

/** out[i] = cast<dstDtype>(in[i]). */
export function cast(
  out: DevPtr,
  dstDtype: Dtype,
  input: DevPtr,
  srcDtype: Dtype,
  n: bigint | number,
  s: Stream = 0,
): void {
  check(raw.ks_cast(ptr(out), dstDtype, ptr(input), srcDtype, n, stream(s)), 'ks_cast');
}

/** out = a*alpha + b*beta (fused AXPBY). */
export function axpby(
  out: DevPtr,
  a: DevPtr,
  alpha: number,
  b: DevPtr,
  beta: number,
  n: bigint | number,
  dtype: Dtype,
  s: Stream = 0,
): void {
  check(raw.ks_axpby(ptr(out), ptr(a), alpha, ptr(b), beta, n, dtype, stream(s)), 'ks_axpby');
}

// ===========================================================================
// loss.h
// ===========================================================================

/** Fused cross-entropy forward+backward. gradLogits may alias logits. */
export function crossEntropy(args: {
  losses: DevPtr;
  gradLogits: DevPtr;
  logits: DevPtr;
  targets: DevPtr;
  targetsI64?: boolean;
  numTokens: bigint | number;
  vocab: bigint | number;
  ignoreIndex?: bigint | number;
  labelSmoothing?: number;
  dtype: Dtype;
  stream?: Stream;
}): void {
  check(
    raw.ks_cross_entropy(
      ptr(args.losses), ptr(args.gradLogits), ptr(args.logits), ptr(args.targets),
      args.targetsI64 ? 1 : 0, args.numTokens, args.vocab, args.ignoreIndex ?? -100,
      args.labelSmoothing ?? 0, args.dtype, stream(args.stream),
    ),
    'ks_cross_entropy',
  );
}

/** Fused-linear-cross-entropy (chunked; no full logits materialized). */
export function fusedLinearCrossEntropy(args: {
  losses: DevPtr;
  gradHidden: DevPtr;
  gradWeightFp32: DevPtr;
  hidden: DevPtr;
  weight: DevPtr;
  targets: DevPtr;
  targetsI64?: boolean;
  numTokens: bigint | number;
  hiddenDim: bigint | number;
  vocab: bigint | number;
  ignoreIndex?: bigint | number;
  labelSmoothing?: number;
  chunkSize?: number;
  dtype: Dtype;
  stream?: Stream;
}): void {
  check(
    raw.ks_fused_linear_cross_entropy(
      ptr(args.losses), ptr(args.gradHidden), ptr(args.gradWeightFp32), ptr(args.hidden),
      ptr(args.weight), ptr(args.targets), args.targetsI64 ? 1 : 0, args.numTokens,
      args.hiddenDim, args.vocab, args.ignoreIndex ?? -100, args.labelSmoothing ?? 0,
      args.chunkSize ?? 0, args.dtype, stream(args.stream),
    ),
    'ks_fused_linear_cross_entropy',
  );
}

// ===========================================================================
// optimizer.h
// ===========================================================================

/** Fused AdamW step. masterParam may be null. */
export function adamw(args: {
  param: DevPtr;
  masterParam?: DevPtr | null;
  grad: DevPtr;
  expAvg: DevPtr;
  expAvgSq: DevPtr;
  lr: number;
  beta1?: number;
  beta2?: number;
  eps?: number;
  weightDecay?: number;
  step: bigint | number;
  gradScale?: number;
  n: bigint | number;
  dtype: Dtype;
  stream?: Stream;
}): void {
  check(
    raw.ks_adamw(
      ptr(args.param), ptr(args.masterParam ?? 0), ptr(args.grad), ptr(args.expAvg), ptr(args.expAvgSq),
      args.lr, args.beta1 ?? 0.9, args.beta2 ?? 0.999, args.eps ?? 1e-8, args.weightDecay ?? 0,
      args.step, args.gradScale ?? 1, args.n, args.dtype, stream(args.stream),
    ),
    'ks_adamw',
  );
}

/** SGD with optional Nesterov momentum and weight decay. */
export function sgdMomentum(args: {
  param: DevPtr;
  masterParam?: DevPtr | null;
  grad: DevPtr;
  momentum: DevPtr;
  lr: number;
  momentumFactor?: number;
  weightDecay?: number;
  nesterov?: boolean;
  gradScale?: number;
  n: bigint | number;
  dtype: Dtype;
  stream?: Stream;
}): void {
  check(
    raw.ks_sgd_momentum(
      ptr(args.param), ptr(args.masterParam ?? 0), ptr(args.grad), ptr(args.momentum),
      args.lr, args.momentumFactor ?? 0.9, args.weightDecay ?? 0, args.nesterov ? 1 : 0,
      args.gradScale ?? 1, args.n, args.dtype, stream(args.stream),
    ),
    'ks_sgd_momentum',
  );
}

/**
 * Compute the global L2 norm of grad tensors for clipping.
 *
 * `grads` is a device array of `numTensors` device pointers and `sizes` is a
 * device int64 array of their element counts. `outNorm` is an fp32 [1] device
 * scalar. Build these arrays on-device or pass pre-uploaded device pointers.
 */
export function globalGradNorm(args: {
  outNorm: DevPtr;
  grads: DevPtr;
  sizes: DevPtr;
  numTensors: number;
  dtype: Dtype;
  stream?: Stream;
}): void {
  check(
    raw.ks_global_grad_norm(
      ptr(args.outNorm), ptr(args.grads), ptr(args.sizes), args.numTensors, args.dtype, stream(args.stream),
    ),
    'ks_global_grad_norm',
  );
}

// ---------------------------------------------------------------------------
// Re-exports for advanced users who want the raw FFI handles or loader.
// ---------------------------------------------------------------------------

export { resolveLibraryPath, getLib } from './lib';
export * as ffi from './lib';
