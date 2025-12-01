#!/usr/bin/env python3
"""
Runner script for continued pretraining of Gemma 3 1B on SEA-PILE-v2.

This script can be run directly on a GPU node:
    python scripts/run_cpt_gemma3_1b.py

Or submitted via Slurm:
    qsub jobs/submit_cpt_gemma3_1b.sh

Environment variables:
    WANDB_API_KEY: WandB API key for logging
    CUDA_VISIBLE_DEVICES: GPU devices to use (default: all available)
"""

import os
import sys
import argparse
from pathlib import Path

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

import nemo.collections.llm as llm
from nemo import lightning as nl
from nemo.collections.llm import PreTrainingDataModule
from megatron.core.optimizer import OptimizerConfig
from pytorch_lightning.loggers import WandbLogger


def parse_args():
    parser = argparse.ArgumentParser(description="Continued pretraining of Gemma 3 1B")
    
    # Data arguments
    parser.add_argument(
        "--data-path",
        type=str,
        default="data/corpora/seapile-v2.jsonl",
        help="Path to training data (JSONL format)",
    )
    parser.add_argument(
        "--seq-length",
        type=int,
        default=2048,
        help="Sequence length for training",
    )
    
    # Training arguments
    parser.add_argument(
        "--max-steps",
        type=int,
        default=10000,
        help="Maximum number of training steps",
    )
    parser.add_argument(
        "--global-batch-size",
        type=int,
        default=256,
        help="Global batch size across all GPUs",
    )
    parser.add_argument(
        "--micro-batch-size",
        type=int,
        default=2,
        help="Micro batch size per GPU",
    )
    parser.add_argument(
        "--devices",
        type=int,
        default=8,
        help="Number of GPUs to use",
    )
    
    # Optimizer arguments
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Learning rate (lower for continued pretraining)",
    )
    parser.add_argument(
        "--min-lr",
        type=float,
        default=1e-5,
        help="Minimum learning rate for scheduler",
    )
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=500,
        help="Number of warmup steps",
    )
    
    # Checkpoint arguments
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="checkpoints/gemma3-1b-seapile",
        help="Directory to save checkpoints",
    )
    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=1000,
        help="Save checkpoint every N steps",
    )
    parser.add_argument(
        "--resume-from",
        type=str,
        default="google/gemma-3-1b-pt",
        help="HuggingFace model ID to import (e.g., google/gemma-3-1b)",
    )
    
    # Logging arguments
    parser.add_argument(
        "--wandb-project",
        type=str,
        default="gemma3-seapile-cpt",
        help="WandB project name",
    )
    parser.add_argument(
        "--wandb-name",
        type=str,
        default="gemma3-1b-seapile-10k",
        help="WandB run name",
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default="logs/wandb",
        help="Directory for logs",
    )
    parser.add_argument(
        "--log-every-n-steps",
        type=int,
        default=50,
        help="Log metrics every N steps",
    )
    
    # Validation arguments
    parser.add_argument(
        "--val-check-interval",
        type=int,
        default=1000,
        help="Run validation every N steps",
    )
    
    return parser.parse_args()


def setup_wandb_logger(args):
    """Set up WandB logger with configuration."""
    # Check for API key
    if not os.getenv("WANDB_API_KEY"):
        print("WARNING: WANDB_API_KEY not found in environment.")
        print("Set it with: export WANDB_API_KEY='your-key-here'")
        print("Or login with: wandb login")
        print("Disabling WandB logging (using offline mode)")
        os.environ["WANDB_MODE"] = "disabled"
    
    return WandbLogger(
        project=args.wandb_project,
        name=args.wandb_name,
        save_dir=args.log_dir,
        log_model=True,
    )


def main():
    args = parse_args()
    
    # Print configuration
    print("=" * 80)
    print("Continued Pretraining Configuration")
    print("=" * 80)
    for arg, value in vars(args).items():
        print(f"{arg:25s}: {value}")
    print("=" * 80)
    
    # Set up WandB logger
    wandb_logger = setup_wandb_logger(args)
    
    # Configure the data module
    print(f"Setting up data from: {args.data_path}")
    data = PreTrainingDataModule(
        paths=[args.data_path],
        seq_length=args.seq_length,
        global_batch_size=args.global_batch_size,
        micro_batch_size=args.micro_batch_size,
        num_workers=4,
    )
    
    # Set up optimizer
    print(f"Setting up optimizer (lr={args.lr})...")
    optimizer = nl.MegatronOptimizerModule(
        config=OptimizerConfig(
            lr=args.lr,
            optimizer="adam",
            use_distributed_optimizer=True,
            weight_decay=0.1,
            adam_beta1=0.9,
            adam_beta2=0.95,
        ),
        lr_scheduler=nl.lr_scheduler.CosineAnnealingScheduler(
            warmup_steps=args.warmup_steps,
            constant_steps=0,
            min_lr=args.min_lr,
        ),
    )
    
    # Checkpoint configuration
    print(f"Checkpoints will be saved to: {args.checkpoint_dir}")
    checkpoint_callback = nl.ModelCheckpoint(
        save_top_k=3,
        monitor="val_loss",
        save_last=True,
        every_n_train_steps=args.checkpoint_interval,
        dirpath=args.checkpoint_dir,
    )
    
    # Training configuration
    print(f"Configuring trainer with {args.devices} GPUs...")
    trainer = nl.Trainer(
        devices=args.devices,
        max_steps=args.max_steps,
        accelerator="gpu",
        strategy=nl.MegatronStrategy(
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=1,
            ddp="megatron",
        ),
        log_every_n_steps=args.log_every_n_steps,
        val_check_interval=args.val_check_interval,
        limit_val_batches=10,
        plugins=nl.MegatronMixedPrecision(precision="bf16-mixed"),
        logger=wandb_logger,
        callbacks=[checkpoint_callback],
    )
    
    # Configure model for import
    print(f"\nPreparing to import model from HuggingFace: {args.resume_from}")
    model_instance = llm.Gemma3Model(config=llm.Gemma3Config1B(seq_length=args.seq_length))
    
    # Train the model - llm.train will handle HF import with proper initialization
    print("\n" + "=" * 80)
    print("Starting training (will import HF model automatically)...")
    print("=" * 80 + "\n")
    
    llm.train(
        model=model_instance,
        data=data,
        trainer=trainer,
        optim=optimizer,
        resume=nl.AutoResume(
            resume_from_path=f"hf://{args.resume_from}",
            resume_if_exists=True,
        ),
    )
    
    print("\n" + "=" * 80)
    print("Training completed!")
    print("=" * 80)


if __name__ == "__main__":
    main()
