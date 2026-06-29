"""Python reference for Janus-Pro ViT encoder — memory-efficient, only loads ViT weights.
NOTE: Uses _t (transposed) weights, so no .T is needed for GEMM."""
import json, struct, os, sys
import numpy as np

BASE = os.path.expanduser("~/models/janus-pro-7b")

VIT_KEYS = [
    "vision_model.embeddings.patch_embedding.weight",
    "vision_model.embeddings.patch_embedding.bias",
    "vision_model.embeddings.position_embedding.weight",
    "vision_model.post_layernorm.weight",
    "vision_model.post_layernorm.bias",
    "vision_model.vision_tower.attn_pool.latent",
    "vision_model.vision_tower.attn_pool.q.weight",
    "vision_model.vision_tower.attn_pool.q.bias",
    "vision_model.vision_tower.attn_pool.kv.weight",
    "vision_model.vision_tower.attn_pool.kv.bias",
    "vision_model.vision_tower.attn_pool.proj.weight",
    "vision_model.vision_tower.attn_pool.proj.bias",
    "vision_model.vision_tower.attn_pool.norm.weight",
    "vision_model.vision_tower.attn_pool.norm.bias",
    "vision_model.vision_tower.attn_pool.mlp.fc1.weight",
    "vision_model.vision_tower.attn_pool.mlp.fc1.bias",
    "vision_model.vision_tower.attn_pool.mlp.fc2.weight",
    "vision_model.vision_tower.attn_pool.mlp.fc2.bias",
]

for i in range(24):
    prefix = f"vision_model.encoder.layers.{i}"
    for suffix in [
        ".layer_norm1.weight", ".layer_norm1.bias",
        ".layer_norm2.weight", ".layer_norm2.bias",
        ".self_attn.q_proj.weight", ".self_attn.q_proj.bias",
        ".self_attn.k_proj.weight", ".self_attn.k_proj.bias",
        ".self_attn.v_proj.weight", ".self_attn.v_proj.bias",
        ".self_attn.out_proj.weight", ".self_attn.out_proj.bias",
        ".mlp.fc1.weight", ".mlp.fc1.bias",
        ".mlp.fc2.weight", ".mlp.fc2.bias",
    ]:
        VIT_KEYS.append(prefix + suffix)

VIT_KEYS = set(VIT_KEYS)

def load_vit_weights():
    tensors = {}
    for fname in ["model_fp16.safetensors", "model_fp16_t.safetensors"]:
        path = os.path.join(BASE, fname)
        if not os.path.exists(path):
            continue
        print(f"Loading ViT weights from {fname}...", file=sys.stderr)
        with open(path, "rb") as f:
            hlen = struct.unpack("<Q", f.read(8))[0]
            header = json.loads(f.read(hlen))
        header_offset = 8 + hlen
        count = 0
        with open(path, "rb") as f:
            for name, info in header.items():
                if name not in VIT_KEYS:
                    continue
                start = header_offset + info["data_offsets"][0]
                end = header_offset + info["data_offsets"][1]
                f.seek(start)
                data = f.read(end - start)
                tensors[name] = np.frombuffer(data, dtype=np.float16).reshape(info["shape"]).astype(np.float32)
                count += 1
        print(f"  Loaded {count} ViT tensors from {fname}", file=sys.stderr)
    print(f"Total ViT tensors: {len(tensors)}", file=sys.stderr)
    return tensors

def make_image(color="red", size=384):
    if color == "red":
        img = np.zeros((3, size, size), dtype=np.float32)
        img[0, :, :] = 1.0
        img[1, :, :] = -1.0
        img[2, :, :] = -1.0
    elif color == "blue":
        img = np.zeros((3, size, size), dtype=np.float32)
        img[0, :, :] = -1.0
        img[1, :, :] = -1.0
        img[2, :, :] = 1.0
    return img

def conv2d_nchw(input_nchw, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
    N, C, H, W = input_nchw.shape
    K = weight.shape[0]
    C_per_group = C // groups
    R, S = weight.shape[2], weight.shape[3]
    OH = (H + 2 * padding - dilation * (R - 1) - 1) // stride + 1
    OW = (W + 2 * padding - dilation * (S - 1) - 1) // stride + 1
    K_per_group = K // groups
    out = np.zeros((N, K, OH, OW), dtype=np.float32)
    for n in range(N):
        for g in range(groups):
            c_start = g * C_per_group
            k_start = g * K_per_group
            for kk in range(K_per_group):
                k_idx = k_start + kk
                for oh in range(OH):
                    for ow in range(OW):
                        acc = 0.0
                        for cc in range(C_per_group):
                            c_idx = c_start + cc
                            for fy in range(R):
                                for fx in range(S):
                                    iy = oh * stride + fy * dilation - padding
                                    ix = ow * stride + fx * dilation - padding
                                    if 0 <= iy < H and 0 <= ix < W:
                                        ival = input_nchw[n, c_idx, iy, ix]
                                        wval = weight[k_idx, cc, fy, fx]
                                        acc += ival * wval
                        if bias is not None:
                            acc += bias[k_idx]
                        out[n, k_idx, oh, ow] = acc
    return out

def layer_norm(x, weight, bias, eps=1e-6):
    mean = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    return (x - mean) / np.sqrt(var + eps) * weight + bias

def gelu(x):
    return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x**3)))

def softmax(x, axis=-1):
    e = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e / np.sum(e, axis=axis, keepdims=True)

def flash_attn_equivalent(q, k, v, scale, causal=False):
    B, S, H, D = q.shape
    scores = np.einsum("bshd,bthd->bhst", q, k) * scale
    if causal:
        mask = np.triu(np.ones((S, S), dtype=np.float32), k=1) * -1e10
        scores = scores + mask[np.newaxis, np.newaxis, :, :]
    attn = softmax(scores.astype(np.float32), axis=-1)
    return np.einsum("bhst,bthd->bshd", attn, v)

def transformer_block(x, weights, layer_idx, heads=16, head_dim=64):
    prefix = f"vision_model.encoder.layers.{layer_idx}"
    hidden = x.shape[-1]
    seq_len = x.shape[1]

    ln1_w = weights[f"{prefix}.layer_norm1.weight"]
    ln1_b = weights[f"{prefix}.layer_norm1.bias"]
    ln_out = layer_norm(x, ln1_w, ln1_b)

    q_w = weights[f"{prefix}.self_attn.q_proj.weight"]
    q_b = weights[f"{prefix}.self_attn.q_proj.bias"]
    k_w = weights[f"{prefix}.self_attn.k_proj.weight"]
    k_b = weights[f"{prefix}.self_attn.k_proj.bias"]
    v_w = weights[f"{prefix}.self_attn.v_proj.weight"]
    v_b = weights[f"{prefix}.self_attn.v_proj.bias"]

    # _t weights are [in_features, out_features], no transpose needed
    q = ln_out @ q_w + q_b
    k = ln_out @ k_w + k_b
    v = ln_out @ v_w + v_b

    q = q.reshape(1, seq_len, heads, head_dim)
    k = k.reshape(1, seq_len, heads, head_dim)
    v = v.reshape(1, seq_len, heads, head_dim)

    scale = 1.0 / np.sqrt(head_dim)
    attn_out = flash_attn_equivalent(q, k, v, scale, causal=False)
    attn_out = attn_out.reshape(1, seq_len, hidden)

    out_w = weights[f"{prefix}.self_attn.out_proj.weight"]
    out_b = weights[f"{prefix}.self_attn.out_proj.bias"]
    proj_out = attn_out @ out_w + out_b

    x = x + proj_out

    ln2_w = weights[f"{prefix}.layer_norm2.weight"]
    ln2_b = weights[f"{prefix}.layer_norm2.bias"]
    ln2_out = layer_norm(x, ln2_w, ln2_b)

    fc1_w = weights[f"{prefix}.mlp.fc1.weight"]
    fc1_b = weights[f"{prefix}.mlp.fc1.bias"]
    fc2_w = weights[f"{prefix}.mlp.fc2.weight"]
    fc2_b = weights[f"{prefix}.mlp.fc2.bias"]

    fc1_out = gelu(ln2_out @ fc1_w + fc1_b)
    fc2_out = fc1_out @ fc2_w + fc2_b

    x = x + fc2_out
    return x

def attn_pool_equivalent(x, latent, weights, heads=16, head_dim=64):
    prefix = "vision_model.vision_tower.attn_pool"
    hidden = x.shape[-1]
    seq_len = x.shape[1]

    latent = latent.reshape(1, hidden).copy()

    q_w = weights[f"{prefix}.q.weight"]
    q_b = weights[f"{prefix}.q.bias"]
    q = latent @ q_w + q_b

    kv_w = weights[f"{prefix}.kv.weight"]
    kv_b = weights[f"{prefix}.kv.bias"]
    kv = x @ kv_w + kv_b

    k = kv[:, :, :hidden]
    v = kv[:, :, hidden:]

    q = q.reshape(1, 1, heads, head_dim)
    k = k.reshape(1, seq_len, heads, head_dim)
    v = v.reshape(1, seq_len, heads, head_dim)

    scale = 1.0 / np.sqrt(head_dim)
    attn_out = flash_attn_equivalent(q, k, v, scale, causal=False)
    attn_out = attn_out.reshape(1, hidden)

    latent = latent + attn_out

    norm_w = weights[f"{prefix}.norm.weight"]
    norm_b = weights[f"{prefix}.norm.bias"]
    ln_out = layer_norm(latent.reshape(1, 1, hidden), norm_w, norm_b)
    ln_out = ln_out.reshape(1, hidden)

    fc1_w = weights[f"{prefix}.mlp.fc1.weight"]
    fc1_b = weights[f"{prefix}.mlp.fc1.bias"]
    fc2_w = weights[f"{prefix}.mlp.fc2.weight"]
    fc2_b = weights[f"{prefix}.mlp.fc2.bias"]

    fc1_out = gelu(ln_out @ fc1_w + fc1_b)
    fc2_out = fc1_out @ fc2_w + fc2_b

    latent = latent + fc2_out

    proj_w = weights[f"{prefix}.proj.weight"]
    proj_b = weights[f"{prefix}.proj.bias"]
    result = latent @ proj_w + proj_b

    return result.reshape(1, 1, hidden)


def main():
    tensors = load_vit_weights()
    print(f"Loaded {len(tensors)} ViT tensors", file=sys.stderr)

    for color in ["red", "blue"]:
        print(f"\n{'='*60}")
        print(f"Testing {color} image")
        print(f"{'='*60}")

        img = make_image(color)

        patch_w = tensors["vision_model.embeddings.patch_embedding.weight"]
        patch_b = tensors["vision_model.embeddings.patch_embedding.bias"]

        conv_out = conv2d_nchw(img[np.newaxis, :, :, :], patch_w, patch_b, stride=16)
        print(f"Conv2D: shape={conv_out.shape}, mean={conv_out.mean():.6f}, std={conv_out.std():.6f}")
        print(f"  min={conv_out.min():.6f}, max={conv_out.max():.6f}")

        B, K, OH, OW = conv_out.shape
        NP = OH * OW
        x = conv_out.transpose(0, 2, 3, 1).reshape(B, NP, K)
        print(f"Transpose: shape={x.shape}, mean={x.mean():.6f}, std={x.std():.6f}")

        pos_emb = tensors["vision_model.embeddings.position_embedding.weight"]
        x = x + pos_emb
        print(f"PosEmb: mean={x.mean():.6f}, std={x.std():.6f}")

        for i in range(24):
            x = transformer_block(x, tensors, i)
            if i == 0:
                print(f"Block 0: mean={x.mean():.6f}, std={x.std():.6f}, min={x.min():.6f}, max={x.max():.6f}")
            if i == 23:
                print(f"Block 23: mean={x.mean():.6f}, std={x.std():.6f}, min={x.min():.6f}, max={x.max():.6f}")

        post_ln_w = tensors["vision_model.post_layernorm.weight"]
        post_ln_b = tensors["vision_model.post_layernorm.bias"]
        x = layer_norm(x, post_ln_w, post_ln_b)
        print(f"PostLN: mean={x.mean():.6f}, std={x.std():.6f}, min={x.min():.6f}, max={x.max():.6f}")

        latent = tensors["vision_model.vision_tower.attn_pool.latent"]
        pooled = attn_pool_equivalent(x, latent, tensors)
        print(f"AttnPool: shape={pooled.shape}, mean={pooled.mean():.6f}, std={pooled.std():.6f}")
        print(f"  min={pooled.min():.6f}, max={pooled.max():.6f}")

        np.save(f"/tmp/py_vit_ln_{color}.npy", x.astype(np.float32))
        np.save(f"/tmp/py_pooled_{color}.npy", pooled.astype(np.float32))


if __name__ == "__main__":
    main()