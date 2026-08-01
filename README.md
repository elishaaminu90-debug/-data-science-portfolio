# 🚢 Titanic Survival Prediction — Complete Data Science Project

An end-to-end data science project covering data cleaning, exploratory 
analysis, feature engineering, model training, evaluation, and 
deployment — using the classic Titanic dataset.

## Project Overview
This project walks through the complete lifecycle of a data science 
problem: starting with raw, imperfect data and ending with a deployed, 
interactive prediction tool. Every step below documents the actual 
process followed, including mistakes made and corrected along the way.

---

## Step 1: Initial Data Exploration

Started with the original Kaggle Titanic dataset (891 rows, 12 columns): 
`PassengerId`, `Survived`, `Pclass`, `Name`, `Sex`, `Age`, `SibSp`, 
`Parch`, `Ticket`, `Fare`, `Cabin`, `Embarked`.

First checks performed:
- Dataset shape and column types
- Missing value counts per column
- Basic statistical summary (`.describe()`)

**Findings:** Age had 177 missing values (20%), Cabin had 687 missing 
values (77%), and Embarked had 2 missing values.

---

## Step 2: Data Cleaning

### Handling Missing Values
- **Embarked (2 missing):** Filled with the mode (most frequent port, 
Southampton) since the count was negligible.
- **Age (177 missing):** Rather than filling with a single blanket 
average, ages were imputed using the **median age within each Title 
group** (see Step 3). A passenger missing age but titled "Master" was 
filled with the median age of other Masters, not a generic average — 
this produces far more realistic estimates.
- **Cabin (77% missing):** Too incomplete to reliably impute. Instead 
of dropping the column outright, it was transformed into a binary 
`HasCabin` feature (1 = cabin recorded, 0 = missing) — preserving a 
useful signal, since having a recorded cabin correlated with wealth 
and survival.

### Handling Data Quality Issues (Simulated Real-World Messiness)
To practice realistic data cleaning, a separate exercise deliberately 
introduced common data quality problems into a copy of the dataset, 
then resolved each one:

| Problem Introduced | How It Was Found | How It Was Fixed |
|---|---|---|
| 20 duplicate passenger records | Checking `PassengerId` duplicates (whole-row check was unreliable due to formatting differences) | Dropped duplicates using `PassengerId` as the unique key |
| Inconsistent text casing ("MALE" vs "male") | `.value_counts()` revealed 4 variants instead of 2 | Standardized with `.str.lower().str.strip()` |
| Extra whitespace in Embarked (' S ' vs 'S') | `.unique()` revealed hidden duplicate categories | Stripped whitespace |
| Impossible ages (300 years old) | `.describe()` flagged unrealistic max value | Converted to missing and re-imputed rather than guessing a number |
| Negative fare values | `.describe()` flagged unrealistic min value | Took the absolute value, treating it as a sign-entry error |

This exercise reinforced that **real-world data almost always requires 
validation before modeling** — a dataset can look "complete" while 
still containing silent errors that break assumptions later.

---

## Step 3: Feature Engineering

This was the single most impactful phase of the project.

- **Title extraction:** Used a regex pattern to pull titles (Mr, Mrs, 
Miss, Master, and rare titles like Dr/Rev/Col) directly from the `Name` 
column. Rare titles were grouped into a single "Rare" category.
- **FamilySize:** Combined `SibSp` + `Parch` + 1 to capture total 
family unit size.
- **IsAlone:** A binary flag for passengers traveling without any family.
- **HasCabin:** Binary flag extracted from the otherwise-too-incomplete 
Cabin column.

**Title alone jumped model accuracy from 75.2% to 84.9%** — a bigger 
improvement than any hyperparameter tuning attempted. Survival rates by 
title revealed a massive spread: Mrs (79.2%) vs Mr (15.7%) — a 
63.5-percentage-point gap.

---

## Step 4: Exploratory Data Analysis

Statistical and visual analysis of the cleaned dataset revealed:

- **Fare is heavily right-skewed** (skewness = 4.79) — mean ($32.20) 
more than double the median ($14.45), driven by a handful of very 
expensive tickets, several shared across family groups (e.g., 4 members 
of the Fortune family each paying $263).
- **Correlation with survival:**
  - `HasCabin`: +0.317 (strongest positive correlation)
  - `Fare`: +0.257
  - `Pclass`: -0.338 (strongest correlation overall — lower class 
  number = higher survival)
  - `Age`: -0.077 (negligible)
- **Survival by gender:** Women survived at 50%, men at only 12.9% — 
nearly a 4x difference.
- **Survival by class:** 1st class 42.1%, 2nd class 31.4%, 3rd class 16.8%.

**Conclusion:** Wealth and class access — reflected across `Pclass`, 
`Fare`, and `HasCabin` — were the dominant survival factors, far more 
than age or family size alone.

Visualizations produced:
- Fare distribution (histogram + boxplot) showing the skew
- Survival rate by class, fare, and cabin status
- Missing data heatmap showing the original pattern of incompleteness

---

## Step 5: Model Training & Evaluation

Multiple models and configurations were tested and compared:

| Model | Accuracy |
|---|---|
| Baseline Random Forest (no feature engineering) | 71.0% |
| Random Forest with balanced class weights | 69.8% |
| Random Forest with FamilySize/IsAlone added | 68.7% |
| Tuned Random Forest (manual hyperparameters) | 75.2% |
| GridSearchCV-optimized Random Forest | 73.3% |
| XGBoost (tuned) | 74.4% |
| **Final Random Forest (with Title feature)** | **84.9%** ✅ |

Interesting negative results along the way:
- Balancing class weights slightly *hurt* accuracy — the imbalance 
wasn't the real bottleneck.
- Adding FamilySize/IsAlone on top of SibSp/Parch (already present) 
introduced redundant noise and slightly *reduced* accuracy.
- GridSearchCV's "best" cross-validated parameters performed worse on 
the actual test set than manually tuned parameters — a reminder that 
cross-validation scores don't always generalize perfectly.
- XGBoost, despite being a more powerful algorithm in general, did not 
outperform Random Forest on this specific dataset — reinforcing that 
**better data (Title feature) beat a better algorithm every time.**

### Final Model
- **Algorithm:** Random Forest Classifier
- **Hyperparameters:** 200 estimators, max depth 10, min samples split 
5, min samples leaf 2
- **Features:** Pclass, Sex, Age, SibSp, Parch, Fare, Embarked, Title, HasCabin

### Final Performance (on held-out test set)
| Class | Precision | Recall | F1-Score |
|---|---|---|---|
| Did Not Survive | 0.85 | 0.90 | 0.87 |
| Survived | 0.84 | 0.78 | 0.81 |
| **Overall Accuracy** | | | **84.9%** |

---

## Step 6: Deployment

The final model was saved using `joblib` and deployed as an interactive 
web app using **Streamlit**, allowing anyone to input a hypothetical 
passenger's details and receive an instant survival prediction with 
a confidence score.

🔗 **[Try the Titanic Survival Predictor live](https://8spkuqwwp55sbesyshmfjn.streamlit.app/)**

---

## Tech Stack
- **Language:** Python
- **Data handling:** Pandas, NumPy
- **Modeling:** Scikit-learn (Random Forest), XGBoost
- **Visualization:** Matplotlib, Seaborn, Tableau
- **Deployment:** Streamlit

## How to Run Locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Files
- `titanic_analysis.ipynb` — full notebook: cleaning, EDA, feature 
engineering, modeling, evaluation
- `app.py` — Streamlit app interface
- `titanic_survival_model.pkl` — final trained model
- `requirements.txt` — project dependencies

## Author
Elisha Aminu — [GitHub](https://github.com/elishaaminu90-debug) | [LinkedIn](https://www.linkedin.com/in/elisha-aminu-21b009426)
