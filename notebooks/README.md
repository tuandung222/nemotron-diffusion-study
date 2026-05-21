# Notebooks

This directory will contain the Jupyter notebooks for **Series 4 — Re-implementation lab**.

See [`src/04-reimplementation-lab/README.md`](../src/04-reimplementation-lab/README.md) for the planned notebook list.

## Running the notebooks (when they exist)

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
jupyter lab
```

The notebooks target Python 3.11 and PyTorch 2.5+. Each notebook lists its specific dependencies in its first cell.
