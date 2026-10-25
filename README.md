# Student Performance Analysis with PySpark

An analysis of student exam scores using Apache Spark (PySpark): exploration with DataFrames and Spark SQL, then a Spark ML pipeline that predicts the math score.

This project started as a PySpark project during my BSc in Computer Science at the University of Central Punjab. This repository is a revised, reproducible version, rebuilt in October 2026 with Spark DataFrames, Spark SQL and Spark ML.

## What it does

1. **Loads and cleans** the data with Spark (renames columns, drops duplicates and missing rows, adds an average score).
2. **Explores** it: score summary, correlations between the three scores, and average scores by test preparation, lunch type, parental education and gender (Spark SQL `GROUP BY`). Tables are saved as CSV and one chart is saved as PNG.
3. **Predicts the math score** with a Spark ML `Pipeline` (`StringIndexer`, `OneHotEncoder`, `VectorAssembler`) and three regressors: Linear Regression, Random Forest and Gradient-Boosted Trees. They are compared on a held-out 20% test set with RMSE and R², against a baseline that always predicts the mean.

## Dataset

"Students Performance in Exams" from Kaggle (`StudentsPerformance.csv`, 1,000 rows, with `gender`, `race/ethnicity`, `parental level of education`, `lunch`, `test preparation course`, `math score`, `reading score`, `writing score`). Download it and put it in `data/`. The data is not included in this repository.

## How to run

You need Python 3.9+ and Java 17 or newer (Spark runs on the JVM).

```bash
pip install -r requirements.txt
python analysis.py --data data/StudentsPerformance.csv
```

Results are written to `reports/`.

## Results

Dataset: 1,000 students after cleaning. Mean scores: math 66.1, reading 69.2, writing 68.1. The three scores are strongly correlated (math-reading 0.82, math-writing 0.80, reading-writing 0.96).

Average score (mean of the three subjects) by group:

| Group | Category | Students | Average |
|---|---|---|---|
| Test preparation | completed | 358 | 72.7 |
| Test preparation | none | 642 | 65.0 |
| Lunch | standard | 645 | 70.8 |
| Lunch | free/reduced | 355 | 62.2 |
| Parental education | master's degree | 59 | 73.6 |
| Parental education | bachelor's degree | 118 | 71.9 |
| Parental education | associate's degree | 222 | 69.6 |
| Parental education | some college | 226 | 68.5 |
| Parental education | some high school | 179 | 65.1 |
| Parental education | high school | 196 | 63.1 |

By gender, female students scored higher in reading (72.6 vs 65.5) and writing (72.5 vs 63.3), and male students scored higher in math (68.7 vs 63.6).

Predicting the math score (80/20 split, 202 test students; seed 42):

| Model | RMSE | R2 |
|---|---|---|
| Baseline (always the training mean) | 15.48 | - |
| Linear Regression | 5.66 | 0.866 |
| Random Forest | 6.66 | 0.814 |
| Gradient-Boosted Trees | 7.15 | 0.786 |

Linear Regression was best: it predicts math scores to within about 5.7 points on average, against 15.5 for the baseline. The simplest model winning is plausible here, because the reading and writing scores are strongly and almost linearly related to the math score, and the dataset is small. Differences between the models come from a single 202-student test split, so they should not be over-read.

These are associations in one public dataset. For example, students who completed test preparation scored higher, but this data cannot show that the preparation caused it.

The chart `reports/scores_by_group.png` and the tables `reports/avg_by_*.csv` are saved by the script.

## Limitations

- The dataset is small (1,000 rows). Spark runs here in local mode, so this project shows the Spark API and workflow, not performance at scale.
- The data is a public teaching dataset, so the patterns say nothing about real schools.
- Reading and writing scores are used as features for the math score. That makes the prediction easier, and it shows the three scores move together rather than any cause.
