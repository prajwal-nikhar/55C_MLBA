# 1. Dataset Overview

Purpose & motivation

Intended use cases (academic, ML benchmarking, research)

Out-of-scope uses (no real lending without validation)

Dataset summary (5,000 SMEs, 31 features, 20.8% defaults)

# 2. Data Collection & Generation

Data source (synthetic with realistic parameters)

Complete generation process for all features:

Company characteristics (age, employees, revenue)

Financial features (9 indicators with distributions)

ESG features (sector-calibrated distributions)

Alternative data (7 signals: sentiment, patents, etc.)

Engineered features (9 non-linear interactions)

Default label generation (probabilistic model with formula)

Base sector rates (4.2%-19.2% by sector)

Step-by-step generation validation

# 3. Data Description & Characteristics

Detailed feature tables for all 31 features:

Financial (9): ranges, means, stds, descriptions

ESG (6): sector calibrations

Alternative data (7): sentiment, patent index, supply chain

Engineered (9): derivations

Data splits (75-12.5-12.5% stratified)

Default rates by sector

Feature correlations

# 4. Data Quality & Validation
| Metric                  | Target            | Actual     | Status |
|-------------------------|------------------|-----------|--------|
| Missing Values          | 0%               | 0%        | ✅     |
| Duplicates              | 0                | 0         | ✅     |
| VIF (multicollinearity) | <10              | 4.2 max   | ✅     |
| Leakage Detection       | 4-point checklist| All passed| ✅     |


# 5. Preprocessing & Transformations

StandardScaler (fit on train only)

SMOTE (imbalance handling)

Categorical encoding (sector 0-7 mapping)

Feature engineering pipeline

Configuration JSON

# 6. Known Limitations & Caveats (Important!)

❌ Synthetic Data: No temporal dynamics, extreme events, fraud scenarios

❌ Sector Imbalance: Some sectors 0% or 100% defaults

❌ ESG Availability: Real SMEs have limited ESG disclosure

❌ Scope: SMEs only, developed markets, traditional lending

⚠️ Recommendations for each limitation

# 7. Uses & Recommendations

✅ Recommended:

Academic research

Model prototyping

Educational use

Proof-of-concept

❌ Not Recommended:

Production lending without validation

Portfolio pricing

Regulatory capital

Individual loan decisions
