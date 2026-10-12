# Sample Weights & Fractional Differentiation for Financial Machine Learning

> Two techniques that make machine learning work on financial time series: **fractional differentiation** (make a series stationary *without* destroying its memory) and **sample weights** (stop overlapping observations from being counted as independent). Implemented and experimented with in a Jupyter notebook, with experiment tracking through MLflow.

![Python](https://img.shields.io/badge/Python-3.10+-blue) ![Notebook](https://img.shields.io/badge/Jupyter-Notebook-F37626) ![MLflow](https://img.shields.io/badge/Tracking-MLflow-0194E2)

---

## Project Overview

| | |
|---|---|
| **Problem** | Standard ML pipelines make two silent mistakes on financial data: integer differencing erases the long-term memory a model needs, and overlapping labels make samples non-independent. |
| **Solution** | Fractional differentiation to reach stationarity while keeping memory, plus sample weighting to correct for overlap. |
| **Skills shown** | Financial ML methodology, time-series stationarity, experiment tracking, ability to translate research ideas into code. |

---

## The two ideas

### 1. Fractional differentiation
Prices are non-stationary, so models usually work on returns (differencing with order d = 1). That makes the series stationary but **throws away most of its memory**. Fractional differentiation uses a non-integer order `0 < d < 1`: the smallest `d` that passes a stationarity test (e.g. ADF) keeps as much memory of the original series as possible.

### 2. Sample weights
In finance, labels are often built from outcomes that span several periods, so neighbouring observations **overlap in time** and share the same information. Treating them as independent inflates the effective sample size and leads to overfitting. Sample weights reduce the influence of redundant observations (and can emphasize observations with a larger return attribution).

---

## What is in this repository

```
sample_weihghts_fractional_diffenrentiation.ipynb   Implementation, experiments and plots
mlruns/                                             MLflow experiment tracking artifacts
requirements.txt                                    Dependencies
```

<!-- VERIFY: the notebook could not be read automatically. Confirm and fill in:
     - which dataset / asset is used,
     - how labels are built (e.g. triple-barrier) and how weights are computed,
     - the value of d selected and the ADF result,
     - the model and the metric comparison (with vs without weights / frac diff).
     Then add the results table and one figure below. -->

## Results

<!-- VERIFY: insert a short table, e.g.
| Setup | Metric | Value |
| d = 1 (returns) | ... | ... |
| Fractional d = ... | ... | ... |
| + sample weights | ... | ... | -->

---

## Quick start

```bash
pip install -r requirements.txt
jupyter notebook sample_weihghts_fractional_diffenrentiation.ipynb
mlflow ui   # browse the tracked runs
```

---

## Skills

**Financial ML**: stationarity vs memory trade-off, fractional differencing, overlapping labels, sample weighting
**Python**: pandas, NumPy, scikit-learn-style modelling, Jupyter
**MLOps**: experiment tracking and model logging with MLflow

---

## Author

**Thanh Son Potherat**: engineering student focused on quantitative finance, machine learning and data science.
[GitHub](https://github.com/Thanh-Poth) | [LinkedIn](https://www.linkedin.com/in/thanh-son-potherat-b793b333a/)
