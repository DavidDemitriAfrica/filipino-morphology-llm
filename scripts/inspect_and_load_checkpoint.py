#!/usr/bin/env python3
"""
Direct inspection and loading of distributed checkpoint.
"""
import argparse
import torch
import torch.distributed.checkpoint as dist_cp
from torch.distributed.checkpoint import FileSystemReader
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", required=True, help="Path to NeMo checkpoint directory")
parser.add_argument("--output", required=True, help="Output .pt file path")
args = parser.parse_args()

checkpoint_path = args.checkpoint
weights_dir = Path(checkpoint_path) / "weights"

print(f"Loading from: {weights_dir}")

# Read metadata
reader = FileSystemReader(str(weights_dir))
metadata = reader.read_metadata()

print(f"\nCheckpoint has {len(metadata.state_dict_metadata)} keys\n")

# Show all keys
print("All keys in checkpoint:")
for i, key in enumerate(sorted(metadata.state_dict_metadata.keys()), 1):
    print(f"  {i}. {key}")

# Try loading with a simple state dict structure matching the keys
print("\n" + "="*80)
print("Attempting to load tensors...")
print("="*80)

# Create state dict matching checkpoint structure with correct shapes
state_dict = {}
for key in metadata.state_dict_metadata.keys():
    # Skip _extra_state keys (they're not model weights)
    if "_extra_state" in key:
        continue

    # Get the tensor metadata
    tensor_meta = metadata.state_dict_metadata[key]

    # Extract shape - the API varies, try different attributes
    try:
        if hasattr(tensor_meta, 'size'):
            shape = tensor_meta.size
        elif hasattr(tensor_meta, 'shape'):
            shape = tensor_meta.shape
        elif hasattr(tensor_meta, 'chunks'):
            # For chunked tensors, get size from chunks
            shape = tensor_meta.chunks[0].sizes if tensor_meta.chunks else torch.Size([1])
        else:
            print(f"  Warning: Could not get shape for {key}, using [1]")
            shape = torch.Size([1])

        # Create empty tensor with correct shape and dtype
        state_dict[key] = torch.zeros(shape, dtype=torch.bfloat16)

    except Exception as e:
        print(f"  Warning: Error processing {key}: {e}")
        continue

print(f"Created state dict template with {len(state_dict)} keys")

# Try to load
try:
    dist_cp.load(
        state_dict=state_dict,
        storage_reader=reader,
    )

    print(f"\n✓ Successfully loaded!")
    print(f"  Total keys: {len(state_dict)}")

    # Check which tensors actually loaded
    loaded_keys = [k for k, v in state_dict.items() if v.numel() > 1]
    print(f"  Loaded keys: {len(loaded_keys)}")

    if loaded_keys:
        print(f"\n  Sample loaded tensors:")
        for key in sorted(loaded_keys)[:10]:
            tensor = state_dict[key]
            print(f"    {key}: shape={tensor.shape}, dtype={tensor.dtype}, mean={tensor.float().mean():.6f}")

        # Save the loaded weights
        output_file = Path(args.output)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        print(f"\nSaving to: {output_file}")
        torch.save(state_dict, output_file)

        file_size_gb = output_file.stat().st_size / 1e9
        print(f"✓ Saved: {file_size_gb:.2f} GB")

except Exception as e:
    print(f"\n✗ Failed to load: {e}")
    import traceback
    traceback.print_exc()
