# Transformer guideline heads

One encoder reads a case's leak-free `text_features` and 13 heads, one per guideline A to M, each predict NOT_AT_ISSUE, FOR or AGAINST. Labels, the at-issue threshold and every metric come from the TF-IDF baseline in `notebooks/04_guideline_classifier_baseline.py`, so the result tables line up column for column.

| File | What it does |
|---|---|
| `data.py` | Loads the pool, builds the 13-head targets with the baseline's `targets_for_guideline`, class weights and length-bucketed batches |
| `model.py` | Encoder, mean pooling, dropout and `Linear(hidden, 13 * 3)`, plus the masked weighted loss |
| `train.py` | Fine-tunes one model and writes logits, logs and `config.json` to `runs/<run-name>/` |
| `evaluate.py` | Temperature scaling per head, val thresholds, the test tables and `results/heads/comparison.csv` |
| `kaggle_heads.ipynb` | Clones the repo on Kaggle and runs a 20-step trial, then the full runs |

## Setup on the laptop

The RTX 5060 needs the CUDA build of torch:

```bash
.venv/Scripts/python.exe -m pip install --force-reinstall --no-deps torch==2.15.0 --index-url https://download.pytorch.org/whl/cu132
.venv/Scripts/python.exe -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

`--no-deps` leaves numpy and the rest alone. The `torch==2.15.0` pin in `requirements.txt` accepts the CUDA build, so a later `pip install -r requirements.txt` keeps it.

## The runs

| Run | Command | Batch x accumulation | Learning rate |
|---|---|---|---|
| Legal-BERT 512 | `python -m src.heads.train --model nlpaueb/legal-bert-base-uncased --max-length 512 --run-name legalbert-512` | 16 x 1 | 2e-5 |
| ModernBERT 2,048 | `python -m src.heads.train --model answerdotai/ModernBERT-base --max-length 2048 --run-name modernbert-2048` | 2 x 8 | 3e-5 |
| ModernBERT 4,096 | `python -m src.heads.train --model answerdotai/ModernBERT-base --max-length 4096 --run-name modernbert-4096` | 1 x 16 | 3e-5 |

Every run trains 3 epochs with AdamW, a linear schedule with 10% warm-up and gradient checkpointing. Each setting can be overridden: `--batch-size`, `--grad-accum`, `--lr`, `--epochs`, `--warmup`, `--weight-decay` and `--seed`.

- `--precision auto` uses bf16 on GPUs that run it natively, like the RTX 5060, and fp16 with a GradScaler on Kaggle's T4.
- `--limit 200 --max-steps 20` makes a smoke run on 200 cases from each split.
- `--resume` continues a cut-off run from its last finished epoch.
- `--eval-every N` also scores val every N steps. By default val is scored at each epoch end.

The best checkpoint is the one with the highest mean val macro-F1 over the 8 defensible guidelines, B C D E F G H J.

Then evaluate:

```bash
python -m src.heads.evaluate --run runs/legalbert-512
```

It writes `results/heads/<run>.csv`, `<run>_summary.csv` and `<run>_calibration.csv`, and adds the run to `results/heads/comparison.csv`. It also writes `runs/<run>/test_probabilities.parquet` and `runs/<run>/calibration.json` for integration and the app. Smoke runs can keep their tables out of git with `--results-dir runs/<run>/results`.

## What a run writes

Everything goes to `runs/<run-name>/`, which git ignores:

- `train_log.csv`: loss, learning rate, seconds per step, tokens per second and peak GPU memory
- `val_log.csv`: val loss and macro-F1 per guideline at every evaluation
- `best_model.pt`: weights of the best checkpoint
- `logits_val.npy` and `logits_test.npy`: `(cases, 13, 3)` logits of the best checkpoint
- `cases_<split>.csv`: the pool row and `case_number` behind each logits row
- `config.json`: arguments, git commit, versions, data stats and timings

## Kaggle

The long ModernBERT runs go on a free Kaggle GPU. The one-time steps:

1. Make a Kaggle account and verify a phone number under Settings. Without it Kaggle gives no GPU and no internet.
2. Import `src/heads/kaggle_heads.ipynb` with File, then Import Notebook.
3. Under Settings in the notebook, set the accelerator to GPU T4 x2 and turn internet on.
4. Run it with Save Version, then Save and Run All. The run then carries on after the tab closes.

With two GPUs the notebook runs ModernBERT 2,048 and 4,096 at the same time, one per GPU. Change `RUNS` in the first cell to run fewer.

The command line does the same with an API token in `~/.kaggle/access_token`. The push folder holds the notebook and a `kernel-metadata.json` with `"enable_gpu": true`, `"enable_internet": true` and `"machine_shape": "NvidiaTeslaT4"`:

```bash
kaggle kernels push -p <folder>
kaggle kernels status <user>/<slug>
kaggle kernels output <user>/<slug> -p runs/kaggle --file-pattern "(logits_|cases_|config|train_log|val_log|\.log)"
```

Then move each run folder under `runs/` and evaluate it on the laptop.
