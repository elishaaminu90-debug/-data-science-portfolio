# 🧹 Titanic: A Deep Dive into Data Cleaning & EDA

A standalone data cleaning and exploratory analysis project — separate 
from predictive modeling — focused entirely on identifying, diagnosing, 
and resolving real-world data quality issues.

## Overview
Real-world data is rarely clean. This project deliberately introduces 
realistic messiness into the Titanic dataset — duplicates, inconsistent 
formatting, impossible values, and missing data — then walks through 
detecting and resolving each issue with documented reasoning, followed 
by statistical analysis and visualization of the cleaned data.

## Data Cleaning Process

| Problem | Detection Method | Solution & Reasoning |
|---|---|---|
| 20 duplicate passengers | Whole-row `.duplicated()` was unreliable due to formatting differences | Checked duplicates by `PassengerId` instead — the correct approach when formatting inconsistencies exist |
| Inconsistent Sex casing (MALE/male) | `value_counts()` revealed 4 variants instead of 2 | Lowercased and stripped whitespace |
| Embarked whitespace (' S ' vs 'S') | `.unique()` revealed hidden duplicate categories | Stripped whitespace |
| Impossible Age values (300) | `.describe()` — max value flagged as unrealistic | Converted to missing, then re-imputed rather than guessing a replacement |
| Negative Fare values | `.describe()` — min value flagged as invalid | Took absolute value, assuming a sign-entry error |
| Missing Age (180 values, 20%) | `.isnull().sum()` | Two-tier imputation: Title-based median first, overall median as fallback for rare titles with no peers |
| Missing Cabin (77% missing) | `.isnull().sum()` percentage | Extracted as binary `HasCabin` feature instead of dropping outright — preserved a strong survival signal |
| Missing Embarked (2 values) | `.isnull().sum()` | Filled with mode (most common port) |

## Key Statistical Findings

- **Fare is heavily right-skewed** (skewness = 4.79) — mean ($32.20) is 
more than double the median ($14.45), driven by a small number of very 
expensive tickets
- **Family group bookings** — several of the highest fares are shared 
by multiple family members (e.g., the Fortune family, 4 passengers at 
$263 each), showing fares often represent group bookings, not individual prices
- **Correlation with survival:**
  - HasCabin: **+0.317** (strongest positive correlation)
  - Fare: **+0.257**
  - Pclass: **-0.338** (strongest correlation overall)
  - Age: **-0.077** (negligible)

**Conclusion:** Wealth and class access — reflected across Pclass, Fare, 
and HasCabin — were the dominant factors in Titanic survival, more so 
than age or family size.

## Visualizations
- Fare distribution (histogram + boxplot) — visualizing the right-skew
- Survival rate by Class, Fare, and Cabin status — visualizing the 
correlation findings
- Missing data heatmap — visual proof of the original data quality issues

## Tools Used
- Python, Pandas, NumPy
- Matplotlib, Seaborn
- Statistical analysis (skewness, correlation)

## Files
- `titanic_data_cleaning_eda.ipynb` — full notebook with cleaning, 
statistical analysis, and visualizations

## Related Project
This is a companion piece to my [Titanic Survival Prediction](link) 
project, which focuses on predictive modeling (84.9% accuracy). This 
project instead focuses purely on the data science skill of cleaning 
and understanding data before any modeling begins.
