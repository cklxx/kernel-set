/**
 * kernel-set TypeScript example: RMSNorm end to end.
 *
 * Allocates device buffers, uploads f32 input + weight from Node Buffers, calls
 * ks.rmsNorm (ks_rms_norm), copies the result back, and prints row 0.
 *
 * Run:
 *   cmake -S . -B build -DCMAKE_CUDA_ARCHITECTURES=89 && cmake --build build -j
 *   cd bindings/ts && npm install && npm run build && cd ../..
 *   export KERNEL_SET_LIB="$PWD/build/libkernel_set.so"
 *   npx ts-node examples/ts/rmsnorm.ts      # or compile with tsc, then `node`
 */
import * as ks from 'kernel-set';

const ROWS = 4;
const COLS = 8;
const EPS = 1e-5;
const ELEM_BYTES = 4; // f32
const N = ROWS * COLS;

/** Pack a Float32Array into a little-endian Node Buffer for upload. */
function f32ToBuffer(xs: Float32Array): Buffer {
  const buf = Buffer.alloc(xs.length * 4);
  for (let i = 0; i < xs.length; i++) buf.writeFloatLE(xs[i], i * 4);
  return buf;
}

/** Unpack a little-endian Node Buffer back into a Float32Array. */
function bufferToF32(buf: Buffer): Float32Array {
  const xs = new Float32Array(buf.length / 4);
  for (let i = 0; i < xs.length; i++) xs[i] = buf.readFloatLE(i * 4);
  return xs;
}

function main(): void {
  console.log(`kernel-set ${ks.version()} (${ks.backendName()} backend)`);

  const stream = ks.streamCreate();

  // Host data: a ramp for x, all-ones weight.
  const xHost = Float32Array.from({ length: N }, (_, i) => i * 0.1);
  const wHost = Float32Array.from({ length: COLS }, () => 1.0);

  const x = ks.mallocDevice(N * ELEM_BYTES);
  const w = ks.mallocDevice(COLS * ELEM_BYTES);
  const y = ks.mallocDevice(N * ELEM_BYTES);

  ks.memcpy(x, f32ToBuffer(xHost), N * ELEM_BYTES, ks.MemcpyKind.HostToDevice, stream);
  ks.memcpy(w, f32ToBuffer(wHost), COLS * ELEM_BYTES, ks.MemcpyKind.HostToDevice, stream);

  // The kernel call.
  ks.rmsNorm(y, x, w, ROWS, COLS, EPS, ks.Dtype.F32, stream);
  ks.streamSynchronize(stream);

  const outBuf = Buffer.alloc(N * ELEM_BYTES);
  ks.memcpy(outBuf, y, N * ELEM_BYTES, ks.MemcpyKind.DeviceToHost, stream);
  ks.streamSynchronize(stream);

  const out = bufferToF32(outBuf);
  console.log('rms_norm(out)[0] =', Array.from(out.slice(0, COLS)));

  ks.freeDevice(x);
  ks.freeDevice(w);
  ks.freeDevice(y);
  ks.streamDestroy(stream);
}

main();
