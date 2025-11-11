This repository contains a complete, reproducible pipeline for synthetic SME credit-risk modeling enhanced with ESG and alternative data.
The Gradient Boosting model achieves AUC-ROC = 0.9991 and $6.35M expected savings, validated through cross-validation, calibration, ablation, and sector-level performance analysis.
📂 Repository Structure
Folder / File	Description
data/	Synthetic dataset and schemas
notebooks/	End-to-end Jupyter notebooks
src/	Core ML pipeline and utilities
models/	Trained models and configs
api/	Prediction API scripts
tests/	Unit tests and QA scripts
docs/	Reports, diagrams, and notes
config/	Model config and env files
requirements.txt	Python dependencies
environment.yml	Conda environment
README.md	Project documentation
(13 directories in full project)
Dataset Overview
Features (31 total)
Category	Count	Notes
Financial	9	Liquidity, leverage, profitability ratios
ESG	6	Sector-calibrated scores + composite
Alternative	7	Sentiment, innovation, digital scores, competition
Engineered	9	Interaction terms, risk composites, ratios
Dataset Specs
5,000 SMEs across 8 sectors and 5 regions
Train-Validation-Test split: 75% / 12.5% / 12.5%
Leakage prevention: scaling + SMOTE applied only on train data
Reproducible seed: 42
Installation & Setup
Option 1 — Pip (Quick Start)
pip install -r requirements.txt
python -c "import sklearn, xgboost, lightgbm, shap; print('✓ Environment ready')"
Option 2 — Conda (Recommended)
conda env create -f environment.yml
conda activate esg-credit-risk-env
python -c "import sklearn, xgboost, lightgbm, shap; print('✓ All packages installed')"
Option 3 — Docker (Production-ready)
docker build -t esg-credit-risk .
docker run -p 8000:8000 esg-credit-risk
Quick Start
# Generate dataset
python src/data/generate_data.py

# Train model
python src/train.py

# Evaluate
python src/evaluate.py

# Predict on new cases
python src/predict.py --input sample.json

# Run tests
pytest --maxfail=1 --disable-warnings -q
Model Performance Summary
Metric	Value
AUC-ROC	0.9991
PR-AUC	0.9967
Precision	0.9769
Recall	0.9769
F1-Score	0.9769
Loss Reduction	97.7% ($6.35M)
Top-Decile Lift	4.8×
Additional results included:
5-fold CV stability
Calibration improvement (ECE ↓ 74.9%)
ESG ablation (+0.15% AUC impact)
Sector-level risk differentiation
