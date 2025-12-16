#!/usr/bin/env python3
"""
Convert Megatron checkpoint to HuggingFace format.
"""
import argparse
import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer


def convert_megatron_to_hf(megatron_ckpt_path, base_model, output_path):
    """Convert Megatron checkpoint to HuggingFace format."""

    print(f"Loading Megatron checkpoint: {megatron_ckpt_path}")
    megatron_state = torch.load(megatron_ckpt_path, map_location="cpu")

    print(f"  Keys: {len(megatron_state)}")
    for key in sorted(megatron_state.keys()):
        tensor = megatron_state[key]
        print(f"    {key}: {tensor.shape}")

    print(f"\nLoading base HF model: {base_model}")
    hf_model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )

    # Resize embeddings if needed
    if "module.embedding.word_embeddings.weight" in megatron_state:
        megatron_vocab_size = megatron_state["module.embedding.word_embeddings.weight"].shape[0]
        current_vocab_size = hf_model.config.vocab_size

        if megatron_vocab_size != current_vocab_size:
            print(f"\n⚠ Resizing embeddings: {current_vocab_size} → {megatron_vocab_size}")
            hf_model.resize_token_embeddings(megatron_vocab_size)

    print(f"\nMapping Megatron → HuggingFace...")

    hf_state = {}
    n_layers = 26  # Gemma 2 2B has 26 layers

    # 1. Embedding layer (also used for LM head in weight tying)
    if "module.embedding.word_embeddings.weight" in megatron_state:
        embed_weight = megatron_state["module.embedding.word_embeddings.weight"]
        hf_state["model.embed_tokens.weight"] = embed_weight

        # Gemma uses weight tying - LM head shares embeddings
        hf_state["lm_head.weight"] = embed_weight.clone()

        print("  ✓ Mapped embeddings + LM head")

    # 2. Final layer norm
    if "module.decoder.final_layernorm.weight" in megatron_state:
        hf_state["model.norm.weight"] = megatron_state["module.decoder.final_layernorm.weight"]
        print("  ✓ Mapped final layernorm")

    # 3. Per-layer weights (stacked in Megatron format)
    # QKV weights: [26, 4096, 2304] → need to split into Q, K, V for each layer
    if "module.decoder.layers.self_attention.linear_qkv.weight" in megatron_state:
        qkv_weight = megatron_state["module.decoder.layers.self_attention.linear_qkv.weight"]  # [26, 4096, 2304]
        qkv_prenorm = megatron_state.get("module.decoder.layers.self_attention.linear_qkv.layer_norm_weight")  # [26, 2304]

        print(f"  Processing QKV weights: {qkv_weight.shape}")

        # Split QKV: 4096 = 2048 (Q) + 1024 (K) + 1024 (V) for Gemma2
        # Gemma 2 2B: 8 heads, 256 dim per head, GQA with 4 KV heads
        for layer_idx in range(n_layers):
            layer_qkv = qkv_weight[layer_idx]  # [4096, 2304]

            # Split into Q, K, V
            # Q: 2048, K: 1024, V: 1024
            q_weight = layer_qkv[:2048, :]  # [2048, 2304]
            k_weight = layer_qkv[2048:3072, :]  # [1024, 2304]
            v_weight = layer_qkv[3072:, :]  # [1024, 2304]

            hf_state[f"model.layers.{layer_idx}.self_attn.q_proj.weight"] = q_weight
            hf_state[f"model.layers.{layer_idx}.self_attn.k_proj.weight"] = k_weight
            hf_state[f"model.layers.{layer_idx}.self_attn.v_proj.weight"] = v_weight

            # Pre-norm (RMSNorm before attention)
            if qkv_prenorm is not None:
                hf_state[f"model.layers.{layer_idx}.input_layernorm.weight"] = qkv_prenorm[layer_idx]

        print(f"  ✓ Mapped Q/K/V for {n_layers} layers")

    # O projection (attention output)
    if "module.decoder.layers.self_attention.linear_proj.weight" in megatron_state:
        o_weight = megatron_state["module.decoder.layers.self_attention.linear_proj.weight"]  # [26, 2304, 2048]
        o_postnorm = megatron_state.get("module.decoder.layers.self_attention.linear_proj.post_layernorm.weight")

        for layer_idx in range(n_layers):
            hf_state[f"model.layers.{layer_idx}.self_attn.o_proj.weight"] = o_weight[layer_idx]

            # Post-norm (RMSNorm after attention)
            if o_postnorm is not None:
                hf_state[f"model.layers.{layer_idx}.post_attention_layernorm.weight"] = o_postnorm[layer_idx]

        print(f"  ✓ Mapped O projection for {n_layers} layers")

    # MLP weights
    if "module.decoder.layers.mlp.linear_fc1.weight" in megatron_state:
        fc1_weight = megatron_state["module.decoder.layers.mlp.linear_fc1.weight"]  # [26, 18432, 2304]
        fc1_prenorm = megatron_state.get("module.decoder.layers.mlp.linear_fc1.layer_norm_weight")

        # Gemma uses gated MLP: gate and up projections
        # 18432 = 9216 (gate) + 9216 (up)
        for layer_idx in range(n_layers):
            layer_fc1 = fc1_weight[layer_idx]  # [18432, 2304]

            gate_weight = layer_fc1[:9216, :]  # [9216, 2304]
            up_weight = layer_fc1[9216:, :]  # [9216, 2304]

            hf_state[f"model.layers.{layer_idx}.mlp.gate_proj.weight"] = gate_weight
            hf_state[f"model.layers.{layer_idx}.mlp.up_proj.weight"] = up_weight

            # Pre-feedforward layernorm (before MLP)
            if fc1_prenorm is not None:
                hf_state[f"model.layers.{layer_idx}.pre_feedforward_layernorm.weight"] = fc1_prenorm[layer_idx]

        print(f"  ✓ Mapped MLP gate/up for {n_layers} layers")

    if "module.decoder.layers.mlp.linear_fc2.weight" in megatron_state:
        fc2_weight = megatron_state["module.decoder.layers.mlp.linear_fc2.weight"]  # [26, 2304, 9216]
        fc2_postnorm = megatron_state.get("module.decoder.layers.mlp.linear_fc2.post_layernorm.weight")

        for layer_idx in range(n_layers):
            hf_state[f"model.layers.{layer_idx}.mlp.down_proj.weight"] = fc2_weight[layer_idx]

            # Post-feedforward layernorm (after MLP)
            if fc2_postnorm is not None:
                hf_state[f"model.layers.{layer_idx}.post_feedforward_layernorm.weight"] = fc2_postnorm[layer_idx]

        print(f"  ✓ Mapped MLP down for {n_layers} layers")

    # Load into HF model
    print(f"\nLoading into HuggingFace model...")
    missing, unexpected = hf_model.load_state_dict(hf_state, strict=False)

    print(f"  Missing keys: {len(missing)}")
    if missing:
        # Group missing keys by prefix to understand what's missing
        prefixes = {}
        for key in missing:
            prefix = key.split('.')[0] if '.' in key else key
            prefixes[prefix] = prefixes.get(prefix, 0) + 1

        print("  Missing key groups:")
        for prefix, count in sorted(prefixes.items()):
            print(f"    {prefix}: {count} keys")

        # Show sample missing keys
        print("  Sample missing keys (first 15):")
        for key in sorted(missing)[:15]:
            print(f"    - {key}")

    print(f"  Unexpected keys: {len(unexpected)}")
    if unexpected and len(unexpected) < 20:
        for key in unexpected:
            print(f"    - {key}")

    # Save
    print(f"\nSaving to: {output_path}")
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    hf_model.save_pretrained(output_path)

    # Also save tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    tokenizer.save_pretrained(output_path)

    print(f"✓ Saved HuggingFace model to {output_path}")
    print(f"  Model size: {sum(p.numel() for p in hf_model.parameters()) / 1e9:.2f}B parameters")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--megatron-checkpoint", required=True)
    parser.add_argument("--base-model", default="google/gemma-2-2b")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    convert_megatron_to_hf(
        args.megatron_checkpoint,
        args.base_model,
        args.output
    )
