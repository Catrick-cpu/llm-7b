import json
import uuid

def read(p):
    with open(p, 'r') as f:
        return f.read()

files = {
    'requirements.txt': 'requirements.txt',
    'configs/pretrain_7b_kaggle.yaml': 'configs/pretrain_7b_kaggle.yaml',
    'src/__init__.py': 'src/__init__.py',
    'src/model.py': 'src/model.py',
    'src/tokenizer.py': 'src/tokenizer.py',
    'src/data.py': 'src/data.py',
    'src/eval.py': 'src/eval.py',
    'src/train.py': 'src/train.py',
    'scripts/launch_kaggle.sh': 'scripts/launch_kaggle.sh',
}

cells = []

def mk_cell(cell_type, source, execution_count=None, outputs=None):
    cell = {
        'cell_type': cell_type,
        'metadata': {'id': str(uuid.uuid4())[:8]},
        'source': source if isinstance(source, list) else source.split('\n')
    }
    if cell_type == 'code':
        cell['execution_count'] = execution_count
        cell['outputs'] = outputs if outputs is not None else []
    return cell

cells.append(mk_cell('markdown', """# LLM-7B: 7B Pretraining from Scratch on Kaggle

This notebook contains the full pretraining system. Run all cells to train.
- All model weights are **randomly initialized** (no pretrained weights).
- Model: 7B params, GQA, RoPE, RMSNorm, SwiGLU.
- Checkpoints auto-save and auto-resume across sessions.

Instructions:
1. Set **Accelerator = GPU T4x2 or P100**, **Internet = On** (for online mode)
2. For **offline mode**: mount local data/tokenizer as Kaggle input datasets, update config
3. Run all cells
4. To resume in a new session, just re-run the training cell"""))

cells.append(mk_cell('code', """# Cell 1: Install dependencies
!pip install -q --upgrade pip
!pip install -q torch transformers datasets tokenizers accelerate pyyaml numpy tqdm pyarrow"""))

cells.append(mk_cell('code', """# Cell 2: Create project structure
import os
for d in ["configs", "src", "scripts"]:
    os.makedirs(d, exist_ok=True)
print("Project directories ready")"""))

for target, src in files.items():
    content = read(src)
    cells.append(mk_cell('code', '%%writefile ' + target + '\n' + content))

cells.append(mk_cell('code', """# Cell 12: Smoke test (2 min, tiny model to verify everything works)
!chmod +x scripts/launch_kaggle.sh
!python -m src.train --config configs/pretrain_7b_kaggle.yaml --debug-smoke"""))

cells.append(mk_cell('code', """# Cell 13: Full 7B training (run after smoke test passes)
!bash scripts/launch_kaggle.sh configs/pretrain_7b_kaggle.yaml train"""))

nb = {
    'cells': cells,
    'metadata': {
        'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
        'language_info': {'name': 'python', 'version': '3.10.0'}
    },
    'nbformat': 4,
    'nbformat_minor': 5
}

with open('kaggle_llm7b.ipynb', 'w') as f:
    json.dump(nb, f, indent=1)
print('Updated kaggle_llm7b.ipynb')