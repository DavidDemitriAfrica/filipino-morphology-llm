# Migration Guide: From Separate Repos to Monorepo

This guide helps you update code that references the old `stochastok` and `pacute` repositories to work with the unified `filipino-morphology-llm` monorepo.

## Repository Structure Changes

### Old Structure
```
/home/ubuntu/
├── stochastok/
│   ├── patok_processor.py
│   ├── models/
│   ├── trainers/
│   └── train.py
└── pacute/
    ├── src/pacute/
    ├── tasks/
    └── data/
```

### New Structure
```
/home/ubuntu/filipino-morphology-llm/
├── src/
│   ├── tokenization/          # From stochastok
│   ├── models/                # From stochastok
│   ├── training/              # From stochastok/trainers
│   ├── evaluation/            # From pacute/src/pacute
│   ├── data_processing/       # From stochastok/dataset_preprocessing
│   └── analysis/              # NEW
├── data/
│   ├── affixes/               # filipino_affixes.txt
│   ├── benchmarks/            # pacute/tasks
│   └── corpora/               # Combined data
├── configs/                   # From stochastok/configs
└── experiments/               # train.py, eval.py
```

## Import Changes

### Tokenization (stochastok)

**Old:**
```python
from patok_processor import PatokProcessor
from stochastok_processor import StochastokProcessor
```

**New:**
```python
from src.tokenization import PatokProcessor, StochastokProcessor
# OR after pip install -e .
from tokenization import PatokProcessor, StochastokProcessor
```

### Models (stochastok)

**Old:**
```python
from models.build_models import build_model
from models.model_shell import ModelShell
```

**New:**
```python
from src.models import build_model, ModelShell
# OR
from models import build_model, ModelShell
```

### Training (stochastok)

**Old:**
```python
from trainers.base_trainer import Trainer
from trainers.build_trainers import build_trainer
```

**New:**
```python
from src.training import Trainer, build_trainer
# OR
from training import Trainer, build_trainer
```

### Evaluation (pacute)

**Old:**
```python
from pacute.affixation import create_affixation_dataset
from pacute.composition import create_composition_dataset
from pacute.syllabification import syllabify
from pacute.sampling import sample_by_frequency
```

**New:**
```python
from src.evaluation import (
    create_affixation_dataset,
    create_composition_dataset,
    syllabify,
    sample_by_frequency,
)
# OR
from evaluation import create_affixation_dataset, ...
```

### Data Processing (stochastok)

**Old:**
```python
# Run as scripts
python dataset_preprocessing/tokenize_dataset.py
python dataset_preprocessing/patok_expand_contract_dataset.py
```

**New:**
```python
# Run as scripts
python src/data_processing/tokenize_dataset.py
python src/data_processing/patok_expand_contract_dataset.py
```

## Path Changes

### Data Paths

**Old stochastok:**
```python
affix_file = "data_other/filipino_affixes.txt"
```

**New:**
```python
affix_file = "data/affixes/filipino_affixes.txt"
```

**Old pacute:**
```python
tasks_dir = "tasks/"
data_file = "data/syllables.jsonl"
```

**New:**
```python
tasks_dir = "data/benchmarks/"
data_file = "data/corpora/pacute_data/syllables.jsonl"
```

### Config Paths

**Old:**
```bash
python train.py --config-name pretraining
```

**New:**
```bash
python experiments/train.py --config-name pretraining
```

Or from root:
```bash
cd /home/ubuntu/filipino-morphology-llm
python experiments/train.py --config-name pretraining
```

## Script Updates

### Training Script

**Old:**
```bash
cd /home/ubuntu/stochastok
python train.py --config-name pretraining
```

**New:**
```bash
cd /home/ubuntu/filipino-morphology-llm
python experiments/train.py --config-name pretraining
```

### Evaluation

**Old:**
```bash
cd /home/ubuntu/stochastok
python eval.py --checkpoint model.pt
```

**New:**
```bash
cd /home/ubuntu/filipino-morphology-llm
python experiments/eval.py --checkpoint model.pt
```

### PACUTE Generation

**Old:**
```bash
cd /home/ubuntu/pacute
python -c "from pacute import create_affixation_dataset; create_affixation_dataset(...)"
```

**New:**
```bash
cd /home/ubuntu/filipino-morphology-llm
python -c "from src.evaluation import create_affixation_dataset; create_affixation_dataset(...)"
```

## Configuration Updates

### Hydra Configs

Update path references in `configs/*.yaml`:

**Old (pretraining.yaml):**
```yaml
defaults:
  - _self_

paths:
  data_dir: data/data_as_memmaps
```

**New:**
```yaml
defaults:
  - _self_

paths:
  data_dir: ${hydra:runtime.cwd}/data/corpora
  affix_file: ${hydra:runtime.cwd}/data/affixes/filipino_affixes.txt
  benchmark_dir: ${hydra:runtime.cwd}/data/benchmarks
```

## Testing Updates

**Old:**
```bash
# In stochastok
python test_affix_processor.py

# In pacute
python test_affixation.py
python test_composition.py
```

**New:**
```bash
cd /home/ubuntu/filipino-morphology-llm
pytest tests/test_affix_processor.py
pytest tests/test_affixation.py
pytest tests/test_composition.py

# Or run all tests
pytest tests/
```

## Notebook Updates

If you have notebooks that import from old repos:

**Old:**
```python
import sys
sys.path.append('/home/ubuntu/stochastok')
sys.path.append('/home/ubuntu/pacute')

from patok_processor import PatokProcessor
from pacute.affixation import create_affixation_dataset
```

**New:**
```python
import sys
sys.path.append('/home/ubuntu/filipino-morphology-llm')

from src.tokenization import PatokProcessor
from src.evaluation import create_affixation_dataset
```

Or install the package and use clean imports:
```python
# After: pip install -e /home/ubuntu/filipino-morphology-llm
from tokenization import PatokProcessor
from evaluation import create_affixation_dataset
```

## Benefits of Monorepo

1. **Unified imports**: No more juggling between two repos
2. **End-to-end workflows**: Can tokenize → train → evaluate in one place
3. **Shared utilities**: Common analysis tools in `src/analysis/`
4. **Consistent versioning**: Single version number for the entire project
5. **Easier experimentation**: All configs and scripts in one place
6. **Better organization**: Clear separation between source code, data, and experiments

## Quick Migration Checklist

- [ ] Update all imports to use new `src/` structure
- [ ] Update data file paths (affixes, benchmarks, corpora)
- [ ] Update script execution paths (run from `experiments/`)
- [ ] Update Hydra configs with new path structure
- [ ] Move any custom notebooks to `notebooks/`
- [ ] Update test execution to use `pytest tests/`
- [ ] Install package: `pip install -e /home/ubuntu/filipino-morphology-llm`
- [ ] Verify everything works: `pytest tests/ && python experiments/train.py --help`

## Need Help?

If you encounter import errors or path issues after migration, check:
1. Are you running from the correct directory? (`cd /home/ubuntu/filipino-morphology-llm`)
2. Did you install the package? (`pip install -e .`)
3. Are paths absolute or relative? (Use `${hydra:runtime.cwd}/` in configs for absolute paths)
