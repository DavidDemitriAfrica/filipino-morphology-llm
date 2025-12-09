# Gemma3 Monkey-Patch for NeMo

## Problem

The `Gemma3SelfAttention.forward()` method in NeMo doesn't accept the `rotary_pos_cos_sin` parameter that gets passed by the transformer layer during training, causing this error:

```
TypeError: Gemma3SelfAttention.forward() got an unexpected keyword argument 'rotary_pos_cos_sin'
```

## Root Cause

Gemma 3 uses a custom attention mechanism with both local and global RoPE (Rotary Position Embeddings). The implementation has a signature mismatch with the base `Attention` class in Megatron-Core:

- The transformer layer passes `rotary_pos_cos_sin` during inference/training
- `Gemma3SelfAttention.forward()` doesn't accept this parameter
- This causes a `TypeError` during the validation step

## Solution

A monkey-patch has been applied in `training/nemo/run_cpt.py` that:

1. **Wraps the original `Gemma3SelfAttention.forward()` method**
2. **Accepts the `rotary_pos_cos_sin` parameter** (but ignores it since Gemma3 doesn't need it)
3. **Calls the original method** with only the parameters it expects

## Implementation

The monkey-patch is applied immediately after imports in `run_cpt.py`:

```python
from nemo.collections.llm.gpt.model.gemma3 import Gemma3SelfAttention

# Store the original forward method
_original_gemma3_forward = Gemma3SelfAttention.forward

# Create a wrapper that accepts rotary_pos_cos_sin but ignores it
def patched_gemma3_forward(
    self,
    hidden_states,
    attention_mask,
    key_value_states=None,
    inference_context=None,
    rotary_pos_emb=None,
    rotary_pos_cos=None,
    rotary_pos_sin=None,
    rotary_pos_cos_sin=None,  # <-- Added parameter
    attention_bias=None,
    packed_seq_params=None,
    position_ids=None,
    sequence_len_offset=None,
    *args,
    **kwargs
):
    # Call the original forward with only the parameters it expects
    return _original_gemma3_forward(
        self,
        hidden_states=hidden_states,
        attention_mask=attention_mask,
        key_value_states=key_value_states,
        inference_context=inference_context,
        rotary_pos_emb=rotary_pos_emb,
        rotary_pos_cos=rotary_pos_cos,
        rotary_pos_sin=rotary_pos_sin,
        attention_bias=attention_bias,
        packed_seq_params=packed_seq_params,
        position_ids=position_ids,
        sequence_len_offset=sequence_len_offset,
        *args,
        **kwargs
    )

# Apply the monkey-patch
Gemma3SelfAttention.forward = patched_gemma3_forward
```

## References

- **NeMo PR #14862**: [Stack Gemma 3 global and local rope into a single tensor](https://github.com/NVIDIA/NeMo/pull/14862)
  - Status: Open (as of December 2024)
  - This PR attempts to fix the same issue but hasn't been merged yet
  
- **NeMo Issue #14861**: [Gemma 3 activation recomputation fails due to RoPE](https://github.com/NVIDIA/NeMo/issues/14861)
  - Related issue about Gemma 3 RoPE implementation

## Why This Works

Gemma 3 uses its own custom RoPE implementation that handles local and global attention separately. The `rotary_pos_cos_sin` parameter is used for other models (like standard transformers) that need pre-computed combined cos/sin tensors for inference optimization.

Since Gemma 3:
- Computes its own local/global RoPE embeddings internally
- Doesn't use the combined `rotary_pos_cos_sin` tensor
- Only needs `rotary_pos_emb` for its custom attention mechanism

We can safely **accept and ignore** the `rotary_pos_cos_sin` parameter, preventing the signature mismatch error.

## Testing

After applying this patch, the training should proceed normally:

```bash
# Submit the job
qsub jobs/run_cpt.sh

# Check logs
tail -f logs/<job_id>.hopper-m-02.OU
```

You should see:
```
✓ Running in NeMo Framework X.X.X
Applying Gemma3SelfAttention monkey-patch for rotary_pos_cos_sin parameter...
✓ Gemma3SelfAttention monkey-patch applied successfully!
```

And training should proceed without the `TypeError`.

## Future

This monkey-patch should be removed once:
- NeMo PR #14862 (or similar fix) is merged into the official NeMo release
- You upgrade to a NeMo version that includes the fix

To check if the fix is available, look for the official patch in the NeMo codebase or check the release notes.

## Date

Applied: December 8, 2025
