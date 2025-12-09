# Security Model for Job Scripts

## Overview

This project uses a **template-based approach** to protect sensitive cluster infrastructure information while still enabling easy sharing and collaboration.

## What's Protected

### Sensitive Information (Never Committed)

- **Queue Names**: Cluster-specific queue names (e.g., `AISG_debug`, `gpu_special`)
- **Absolute Paths**: User directories, project locations, log directories
- **Network Topology**: InfiniBand interfaces, network configurations
- **Resource Limits**: Cluster-specific resource constraints
- **User Directories**: Home directories, scratch spaces, shared storage paths

### Why This Matters

Exposing cluster-specific information in public repositories can:
- Reveal infrastructure architecture
- Enable targeted attacks
- Violate organizational security policies
- Leak information about compute resources and capabilities

## How It Works

### Directory Structure

```
job_templates/          # Safe to commit (tracked in git)
├── *.template.pbs     # Sanitized templates with placeholders
├── *.template.sh      # Sanitized shell script templates
├── README.md          # Setup instructions
└── setup_jobs.sh      # Interactive wizard

jobs/                   # NEVER committed (gitignored)
├── *.pbs              # Your customized job scripts
└── *.sh               # Contains real paths and queue names
```

### Template Pattern

**Before (Unsafe - Contains Real Paths):**
```bash
#PBS -q AISG_debug
#PBS -o /scratch_aisg/SPEC-SF-AISG/railey/logs/
export GLOO_SOCKET_IFNAME=ib0
```

**After (Safe - Uses Placeholders):**
```bash
#PBS -q YOUR_QUEUE_NAME
#PBS -o /path/to/your/logs/
export GLOO_SOCKET_IFNAME=YOUR_NETWORK_INTERFACE
```

## Setup for New Users

### Quick Start

```bash
# 1. Run the setup wizard
bash job_templates/setup_jobs.sh

# 2. Follow the prompts to enter your cluster info

# 3. Review generated scripts
ls jobs/

# 4. Submit jobs
qsub jobs/your_job.pbs
```

### Manual Setup

```bash
# 1. Copy a template
cp job_templates/run_evaluation_batch.template.pbs jobs/run_evaluation_batch.pbs

# 2. Replace placeholders
sed -i 's/YOUR_QUEUE_NAME/gpu/g' jobs/run_evaluation_batch.pbs
sed -i 's|/path/to/your/logs/|/home/myuser/logs/|g' jobs/run_evaluation_batch.pbs

# 3. Submit
qsub jobs/run_evaluation_batch.pbs
```

## For Contributors

### Creating New Templates

When creating a new job script template:

1. **Start with working script**: Get your job working first
2. **Identify sensitive info**: Find all paths, queue names, network settings
3. **Replace with placeholders**: Use descriptive `YOUR_*` placeholders
4. **Add documentation**: Include comments explaining what to customize
5. **Save as template**: Save to `job_templates/` with `.template.*` extension
6. **Test**: Verify the setup wizard works with your template

### Checklist Before Committing

- [ ] No absolute paths (except example paths in docs)
- [ ] No queue names (use `YOUR_QUEUE_NAME`)
- [ ] No network interfaces (use `YOUR_NETWORK_INTERFACE`)
- [ ] No user directories (use `/path/to/your/`)
- [ ] All templates in `job_templates/` directory
- [ ] Actual job files in `jobs/` (which is gitignored)
- [ ] Documentation updated with setup instructions

### Review Process

Before pushing commits, check:

```bash
# 1. Verify no jobs/ files are tracked
git status jobs/

# Should output: "nothing to commit, working tree clean"

# 2. Check what's being committed
git diff --staged

# Look for:
#   - Queue names that aren't YOUR_QUEUE_NAME
#   - Paths that aren't /path/to/your/
#   - Network interfaces that aren't YOUR_NETWORK_INTERFACE
#   - Any other cluster-specific info

# 3. If you find sensitive info
git reset HEAD <file>
# Edit the file, replace with placeholders, then re-add
```

## GitIgnore Configuration

The `.gitignore` file protects the `jobs/` directory:

```gitignore
# Protect all job scripts (contain cluster-specific paths)
jobs/

# Also protect logs
logs/
```

This means:
- ✅ `job_templates/*` are tracked and committed
- ❌ `jobs/*` are **never** tracked or committed
- ✅ Users can create `jobs/` locally with real paths
- ❌ Those real paths never leak to git

## Verification

### Check Template Safety

```bash
# Templates should NOT contain:
grep -r "AISG_debug" job_templates/      # Should find nothing
grep -r "/scratch_aisg/" job_templates/  # Should find nothing
grep -r "railey" job_templates/          # Should find nothing (your username)

# Templates SHOULD contain:
grep -r "YOUR_QUEUE_NAME" job_templates/ # Should find many
grep -r "/path/to/your/" job_templates/  # Should find many
```

### Check Jobs Are Protected

```bash
# This should fail (good!)
git add jobs/*.pbs
# Error: jobs/ is gitignored

# Your jobs directory remains local
ls jobs/  # Works locally
git status jobs/  # Not tracked
```

## FAQ

### Q: Why not use environment variables instead?

**A:** Environment variables would still require documentation of what those variables should be, potentially exposing the same information. Templates make it explicit what needs customization.

### Q: Can I commit my actual job files?

**A:** No! The `jobs/` directory is gitignored for security. Always commit to `job_templates/` with placeholders.

### Q: What if I accidentally commit a real path?

**A:** 
1. Remove from git history: `git filter-branch` or `git rebase -i`
2. Force push to overwrite remote
3. Consider the information potentially exposed
4. Notify your security team if necessary

### Q: How do I share job configurations with teammates?

**A:**
1. Create a template in `job_templates/`
2. Commit and push the template
3. Teammates run `bash job_templates/setup_jobs.sh` to customize

### Q: What about Slurm clusters?

**A:** The same principles apply. Replace PBS directives with SLURM equivalents:
- `#PBS -q` → `#SBATCH --partition=`
- `#PBS -l` → `#SBATCH --nodes=`, `#SBATCH --gres=gpu:`
- etc.

## Best Practices

1. **Always use templates**: Never commit actual job scripts
2. **Review before commit**: Check diffs for sensitive info
3. **Use descriptive placeholders**: `YOUR_QUEUE_NAME` not `QUEUE`
4. **Document requirements**: Add comments explaining what's needed
5. **Test templates**: Verify setup wizard works
6. **Keep jobs/ local**: Never push the jobs/ directory

## Related Documentation

- `job_templates/README.md` - Setup instructions for templates
- `docs/TRAINING.md` - Training workflow documentation
- `docs/EVALUATION.md` - Evaluation workflow documentation
- `.gitignore` - Git ignore rules

---

**Remember**: When in doubt, use a placeholder. It's better to require manual setup than to leak infrastructure information.
