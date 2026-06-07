# Materials Intelligence Lab

An interactive Streamlit app for experimental materials data analysis and research-oriented exploration.

## What it does

Upload one or more CSV files, map materials metadata, clean the data, run PCA, clustering, and anomaly detection, then export an HTML or PDF report.

## Features

- Single-file or multi-file CSV upload
- Multi-file upload with auto-merge heuristics
- Materials metadata mapping and sample grouping
- Robust cleaning and preprocessing
- Interactive visualization
- PCA analysis
- Clustering
- Anomaly detection
- HTML and PDF scientific report export
- Processed dataset download

## Recommended dataset workflow

- Start with [`sample_data/materials_experiment_sample.csv`](sample_data/materials_experiment_sample.csv) to test the full app.
- Use real public data such as the UCI Superconductivity dataset when you want a stronger benchmark.
- Keep private or unpublished lab data in `data/raw/` locally, not in GitHub.

## Good real datasets to test

- UCI Superconductivty Data Set: a widely used materials science regression dataset with 21,263 superconductors and 81 features.
- Materials Project / MPContribs: best if you want to test with genuine materials or experimental contribution data, but it may require an API key or a specific contribution project.
- JARVIS-DFT: great for materials-property analysis, especially if you want a larger public dataset of computed materials properties.

## Project layout

- `app.py`: Streamlit application
- `sample_data/`: small synthetic CSV for quick testing
- `data/`: local data area for private or downloaded datasets
- `outputs/`: generated files and exports
- `reports/`: exported analysis reports

## Multi-file datasets

If a dataset comes as `train.csv` plus helper files like `unique_m.csv`, upload them together.
The app will try to auto-merge shared keys such as `sample_id`, `formula`, `id`, `material`, or `batch`.
If it cannot safely merge them, it can fall back to using only the first file.

## GitHub notes

- Commit the code, README, sample data, and lightweight documentation.
- Do not commit large raw datasets unless they are public and small enough to version comfortably.
- Add screenshots later if you want the repo to look stronger for reviewers.

## Run

```bash
pip install -r requirements.txt
streamlit run app.py
```
