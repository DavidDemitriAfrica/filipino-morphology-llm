# Job Templates

This directory contains **template** PBS/SLURM job scripts for HPC clusters. All sensitive cluster-specific information (paths, queue names, etc.) has been replaced with placeholders like `YOUR_QUEUE_NAME`.

## 🚀 Quick Setup (2 minutes)

**Interactive wizard** (recommended):
```bash
bash job_templates/setup_jobs.sh
```

**Manual setup**:
```bash
# Copy template
cp job_templates/run_evaluation_batch.template.pbs jobs/run_evaluation_batch.pbs

# Edit and replace: YOUR_QUEUE_NAME, /path/to/your/logs/, etc.
vim jobs/run_evaluation_batch.pbs

# Or use sed

sed -i 's/YOUR_QUEUE_NAME/gpu/g' jobs/run_evaluation_batch.pbs
sed -i 's|/path/to/your/logs/|/home/me/logs/|g' jobs/run_evaluation_batch.pbs

# Submit
qsub jobs/run_evaluation_batch.pbs
```

## 📋 Templates

| Template | Purpose |
|----------|---------|
| `run_evaluation_test.template.pbs` | Quick test (GPT-2, 100 samples) |
| `run_evaluation_batch.template.pbs` | Full evaluation (all models) |
| `run_cpt.template.pbs` | Training with NeMo (multi-GPU) |
| `preprocess_data.template.pbs` | Data preprocessing |
| `build_tokenizer_expansions.template.pbs` | Build tokenizer |

## 🔧 Placeholders to Replace

```bash
YOUR_QUEUE_NAME          # e.g., gpu, debug, batch
YOUR_NCPUS              # e.g., 4, 16, 64
YOUR_NGPUS              # e.g., 1, 4, 8
YOUR_MEMORY             # e.g., 32GB, 64GB, 512GB
YOUR_WALLTIME           # e.g., 01:00:00, 12:00:00
YOUR_NETWORK_INTERFACE  # e.g., ib0, eth0 (for multi-node)
/path/to/your/logs/     # Your log directory
/path/to/your/project   # Your project directory
```

## � Why Templates?

The `jobs/` directory is **gitignored** to protect:
- Cluster queue names (e.g., `AISG_debug`)
- Absolute paths (e.g., `/scratch_aisg/...`)
- Network topology (e.g., `ib0` interface names)

This keeps cluster infrastructure private while sharing workflow. See `docs/SECURITY.md` for details.

---

**Questions?** See `docs/TRAINING.md` and `docs/EVALUATION.md` for workflow guides.
- Adjust resource requirements based on your cluster's availability
