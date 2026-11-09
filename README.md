# FairPrice Auto

Explainable price and anomaly screening for used-car listings.

Project for the course *Business and Project Management* (University of Pisa, A.Y. 2025/2026), theme *AI-Driven Business*.
Authors: Valentina Bertei, Alex Sgammato, Francesco Tarchi.

For any used-car listing, FairPrice Auto estimates the expected asking price with a p10–p90 range and flags fields that are inconsistent with comparable cars, explaining each signal with the listing's value and the median of similar cars. It does **not** detect hidden defects: it flags statistical inconsistencies in the declared data.

## Repository contents

| Path | What it is |
|---|---|
| `FairPrice.ipynb` | Main notebook: audit, cleaning, price model, anomaly detection, validation and figures |
| `FairPriceDemo.ipynb` | Loads the saved models and evaluates a single listing, with JSON output |
| `app.py` | Streamlit interface for the prototype |
| `data/` | Cleaned datasets, audit tables and trained models (`data/models/`), produced by the main notebook |
| `figures/` | Charts produced by the main notebook (audit, price model, validation) |
| `assets/`, `.streamlit/` | Logo and theme of the Streamlit interface |

## Data

The raw dataset is not included because of its size (about 520 MB). Download the *AutoScout24 Car Listings Dataset* from Kaggle:
https://www.kaggle.com/datasets/clkmuhammed/autoscout24-car-listings-dataset

and save it as:

```
dataset/autoscout24_dataset_20251108.csv
```

The models are trained on the Italian listings only (23,957 rows; 22,632 after cleaning and de-duplication).

## Setup

Python 3.11 or later is recommended.

```
pip install -r requirements.txt
```

## How to run

1. **Main notebook.** Open `FairPrice.ipynb` and run all cells in order. It reads the raw CSV from `dataset/`, writes cleaned data and audit tables to `data/`, the trained models to `data/models/` and the figures to `figures/`.
   Sections: 1. Dataset exploration · 2. Preprocessing · 3. Price prediction model and market range · 4. Explainable anomaly detection · 5. Complete assessment pipeline · 6. Technical validation, coherence rules and figures.
2. **Demo.** After the main notebook, open `FairPriceDemo.ipynb` and run it to evaluate a random test listing.
3. **Interface.** After the main notebook, from the project folder:
   ```
   python -m streamlit run app.py
   ```

All random steps use `SEED = 1234`, so a full rerun reproduces the numbers reported in the paper (with the same library versions). The manual review of the 30 most anomalous listings (section 6) reads the annotated file `data/audit/manual_review_top30.csv`, which is included.

## License and data

Code for academic use. The dataset is distributed on Kaggle under its own license; no personal identifiers (VIN, address, seller name, free-text description) are used by the models.
