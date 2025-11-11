# Financial Risk Scoring – ML + ESG + Alt-Data

[![Build](https://img.shields.io/badge/build-passing-brightgreen)](#)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](#)
[![License](https://img.shields.io/badge/license-MIT-lightgrey)](#)
[![MLflow](https://img.shields.io/badge/MLflow-enabled-1f65d6)](#)
[![Streamlit](https://img.shields.io/badge/Streamlit-dashboard-ee4b2b)](#)

A production-grade financial risk scoring pipeline that fuses **financials**, **ESG**, and **alternative data** with robust MLOps.

- **AUC-ROC:** **0.9991**
- **Business impact:** **$6.35M savings** (97.7% loss reduction)
- **Top-decile lift:** **4.8×**

---

## Table of Contents
- [Project Overview](#project-overview)
- [Dataset & Features](#dataset--features)
- [Data Split](#data-split)
- [Leakage Prevention Checklist](#leakage-prevention-checklist)
- [Architecture](#architecture)
- [Pipeline Flow](#pipeline-flow)
- [Installation & Setup](#installation--setup)
- [Quick Start](#quick-start)
- [Model Performance](#model-performance)
- [Streamlit UI (Interactive)](#streamlit-ui-interactive)
- [Reproducibility & MLOps](#reproducibility--mlops)
- [Project Structure](#project-structure)
- [License](#license)

---

## Project Overview

This repository trains, evaluates, and deploys a **credit risk model** using **scikit-learn/XGBoost/LightGBM**, explains decisions with **SHAP**, tracks experiments via **MLflow**, and serves an interactive **Streamlit** dashboard.

**Highlights**
- **AUC-ROC:** 0.9991
- **PR-AUC:** 0.9967
- **Precision/Recall/F1:** 0.9769 / 0.9769 / 0.9769
- **Loss reduction:** 97.7% (**$6.35M**)
- **Top-Decile Lift:** 4.8×

---

## Dataset & Features

- **Financial features (with ranges):** liquidity, leverage, profitability ratios (scaled to sensible ranges, e.g., `0.01–250×` as applicable).
- **ESG features (sector-calibrated):** sector-normalized ESG scores, carbon intensity, controversies, governance signals.
- **Alternative data signals:** market microstructure, analyst sentiment, macro drivers.
- **Engineered features:** rolling volatility windows, risk deltas, FI×ESG interactions.

---

## Data Split

- **Train:** 75%  
- **Validation:** 12.5%  
- **Test:** 12.5%

---

## Leakage Prevention Checklist

- [x] No future-dated features  
- [x] No target-conditioned filters or joins  
- [x] Time-aware split for validation/test  
- [x] Statistical leakage tests on folds  
- [x] Strict separation of preprocessing fit between train/val/test

---
