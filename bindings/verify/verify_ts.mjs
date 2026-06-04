/**
 * FFI verification driver for the kernel-set TypeScript/koffi binding.
 *
 * Loads the compiled binding (bindings/ts/dist) against the CPU stub
 * (KERNEL_SET_LIB), then exercises symbol resolution, koffi argument
 * marshalling, and ks_status_t handling — without a GPU. Pointers are dummy 0
 * and never dereferenced by the stub.
 *
 * Run:
 *   KERNEL_SET_LIB=/abs/path/libkernel_set.dylib \
 *     node bindings/verify/verify_ts.mjs
 */
import { createRequire } from 'node:module';
import * as path from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const require = createRequire(import.meta.url);

const distIndex = path.resolve(__dirname, '..', 'ts', 'dist', 'index.js');
const ks = require(distIndex);

function assert(cond, msg) {
  if (!cond) {
    console.error('[ts] ASSERT FAILED:', msg);
    process.exit(1);
  }
}

try {
  // ---- introspection ----------------------------------------------------
  const v = ks.version();
  console.log(`[ts] version()      = ${JSON.stringify(v)}`);
  assert(v === '0.0.0-stub', `unexpected version: ${v}`);

  const bn = ks.backendName();
  console.log(`[ts] backendName()  = ${JSON.stringify(bn)}`);
  assert(bn === 'stub', `unexpected backend: ${bn}`);

  console.log(`[ts] dtypeName(BF16)     = ${JSON.stringify(ks.dtypeName(ks.Dtype.BF16))}`);
  const bits = ks.dtypeSizeBits(ks.Dtype.BF16);
  console.log(`[ts] dtypeSizeBits(BF16) = ${bits}`);
  assert(bits === 16, `unexpected size bits: ${bits}`);

  console.log(`[ts] statusString(Success) = ${JSON.stringify(ks.statusString(ks.Status.Success))}`);
  console.log(`[ts] lastErrorString()     = ${JSON.stringify(ks.lastErrorString())}`);

  // ---- device queries (out-params) -------------------------------------
  const n = ks.deviceCount();
  console.log(`[ts] deviceCount()  = ${n}`);
  assert(n === 1, `stub should report 1 device, got ${n}`);

  ks.setDevice(0);
  console.log(`[ts] getDevice()    = ${ks.getDevice()}`);

  const props = ks.getDeviceProperties(0);
  console.log(`[ts] device 0: name=${JSON.stringify(props.name)} warp=${props.warpSize} maxThreads=${props.maxThreadsPerBlock}`);
  assert(props.name === 'stub-device', `unexpected device name: ${props.name}`);
  assert(props.warpSize === 32, `unexpected warp size: ${props.warpSize}`);

  // ---- streams ----------------------------------------------------------
  const s = ks.streamCreate();
  console.log(`[ts] streamCreate() = ${s}`);
  ks.streamSynchronize(s);
  ks.streamDestroy(s);
  console.log('[ts] stream create/sync/destroy OK');

  // ---- op wrappers reaching the C call with dummy (0) pointers ----------
  // The stub never dereferences pointers, so 0 is safe. These confirm koffi
  // marshalling + symbol resolution + ks_status_t handling (check() throws on
  // non-zero, so reaching the next line means KS_SUCCESS).
  ks.add(0, 0, 0, 16, ks.Dtype.F16, 0);
  console.log('[ts] add(0,0,0,...) -> success');

  ks.rmsNorm(0, 0, 0, 2, 8, 1e-6, ks.Dtype.F16, 0);
  console.log('[ts] rmsNorm(0,...) -> success');

  ks.gemm({
    c: 0, a: 0, b: 0, m: 4, n: 4, k: 4,
    transA: false, transB: false, lda: 4, ldb: 4, ldc: 4,
    alpha: 1.0, beta: 0.0, dtype: ks.Dtype.F16, stream: 0,
  });
  console.log('[ts] gemm({...}) -> success');

  // sample exercises uint64 seed/offset marshalling (BigInt)
  ks.sample({
    outTokens: 0, outProbs: 0, logits: 0,
    temperatures: 0, topKs: 0, topPs: 0,
    numSeqs: 4, vocabSize: 32000,
    seed: 1234n, philoxOffset: 0n,
    dtype: ks.Dtype.F32, stream: 0,
  });
  console.log('[ts] sample({... uint64 seed ...}) -> success');

  ks.softmax(0, 0, 4, 32000, 1.0, ks.Dtype.F32, 0);
  console.log('[ts] softmax(0,...) -> success');

  console.log('[ts] PASS');
} catch (err) {
  console.error('[ts] FAIL:', err && err.stack ? err.stack : err);
  process.exit(1);
}
