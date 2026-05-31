# Titanic Survival Prediction

## Overview
End-to-end data science project predicting Titanic passenger 
survival using machine learning.

## Results
| Model | Accuracy |
|---|---|
| Baseline Random Forest | 71.0% |
| Tuned Random Forest | 75.2% |
| Final Model (with Title feature) | 84.9% |

## Key Findings
- Women survived at 4x the rate of men
- 1st class passengers survived at 2.5x the rate of 3rd class
- Title was the strongest predictor — Mrs survival rate 79.2% vs Mr at 15.7%
- Feature engineering outperformed algorithm tuning

## Tools Used
- Python, Pandas, NumPy
- Scikit-learn, XGBoost
- Matplotlib, Seaborn

## Files
- `titanic_analysis.ipynb` — full notebook with code and analysis
- `train.csv` — dataset from Kaggle Titanic competition
