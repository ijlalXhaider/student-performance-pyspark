"""Student performance analysis with PySpark.

Part 1: explore the data with Spark DataFrames and Spark SQL.
Part 2: predict the math score with Spark ML (pipeline + three regressors).

Usage:
    python analysis.py --data data/StudentsPerformance.csv
"""
import argparse
import json
import os
import sys

# Make Spark use the same Python that runs this script (avoids a common Windows error).
os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
os.environ.setdefault("PYSPARK_DRIVER_PYTHON", sys.executable)

# On Windows, Spark's launcher breaks when the temp folder path contains a space
# (for example C:\Users\First Last\AppData\Local\Temp). Use a space-free folder instead.
if os.name == "nt" and " " in os.environ.get("TEMP", ""):
    _tmp = "C:\\spark_tmp"
    os.makedirs(_tmp, exist_ok=True)
    os.environ["TEMP"] = os.environ["TMP"] = os.environ["SPARK_LOCAL_DIRS"] = _tmp

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.feature import OneHotEncoder, StringIndexer, VectorAssembler
from pyspark.ml.regression import GBTRegressor, LinearRegression, RandomForestRegressor
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

SEED = 42
RENAME = {
    "race/ethnicity": "group",
    "parental level of education": "parent_education",
    "test preparation course": "test_prep",
    "math score": "math",
    "reading score": "reading",
    "writing score": "writing",
}
CATEGORICAL = ["gender", "group", "parent_education", "lunch", "test_prep"]
SCORES = ["math", "reading", "writing"]


def load(spark, path):
    df = spark.read.csv(path, header=True, inferSchema=True)
    for old, new in RENAME.items():
        if old in df.columns:
            df = df.withColumnRenamed(old, new)
    missing = [c for c in CATEGORICAL + SCORES if c not in df.columns]
    if missing:
        raise SystemExit(f"Missing columns {missing}. Found: {df.columns}")
    df = df.dropDuplicates().dropna()
    return df.withColumn("average", F.round((F.col("math") + F.col("reading") + F.col("writing")) / 3, 2))


def explore(spark, df, out):
    n = df.count()
    print(f"\nRows after cleaning: {n}")
    print("\nScore summary:")
    df.select(*SCORES, "average").describe().show()

    summary = {"rows": n}

    # Correlations between the three scores (DataFrame API).
    corr = {}
    for i, a in enumerate(SCORES):
        for b in SCORES[i + 1:]:
            corr[f"{a}-{b}"] = round(df.stat.corr(a, b), 3)
    print("Correlations:", corr)
    summary["correlations"] = corr

    # Average score by group (Spark SQL).
    df.createOrReplaceTempView("students")
    tables = {}
    for col in ["test_prep", "lunch", "parent_education", "gender"]:
        res = spark.sql(
            f"""SELECT {col} AS category, COUNT(*) AS students,
                       ROUND(AVG(math),1) AS math, ROUND(AVG(reading),1) AS reading,
                       ROUND(AVG(writing),1) AS writing, ROUND(AVG(average),1) AS average
                FROM students GROUP BY {col} ORDER BY average DESC"""
        )
        print(f"\nAverage scores by {col}:")
        res.show(truncate=False)
        pdf = res.toPandas()
        pdf.to_csv(os.path.join(out, f"avg_by_{col}.csv"), index=False)
        tables[col] = pdf
    summary["avg_by"] = {k: v.to_dict("records") for k, v in tables.items()}

    # One chart: average score by parental education and by test prep.
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    p = tables["parent_education"].sort_values("average")
    ax[0].barh(p["category"], p["average"], color="#2B6777")
    ax[0].set_title("Average score by parental education")
    ax[0].set_xlim(50, 80)
    t = tables["test_prep"]
    ax[1].bar(t["category"], t["average"], color="#2B6777")
    ax[1].set_title("Average score by test preparation")
    ax[1].set_ylim(50, 80)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "scores_by_group.png"), dpi=150)
    plt.close(fig)
    return summary


def predict_math(df, out):
    """Predict the math score from the other columns with three Spark ML regressors."""
    train, test = df.randomSplit([0.8, 0.2], seed=SEED)
    print(f"\nTrain rows: {train.count()}  Test rows: {test.count()}")

    idx_cols = [c + "_idx" for c in CATEGORICAL]
    ohe_cols = [c + "_ohe" for c in CATEGORICAL]
    prep = [
        StringIndexer(inputCols=CATEGORICAL, outputCols=idx_cols, handleInvalid="keep"),
        OneHotEncoder(inputCols=idx_cols, outputCols=ohe_cols),
        VectorAssembler(inputCols=ohe_cols + ["reading", "writing"], outputCol="features"),
    ]
    models = {
        "LinearRegression": LinearRegression(featuresCol="features", labelCol="math"),
        "RandomForest": RandomForestRegressor(featuresCol="features", labelCol="math", numTrees=100, seed=SEED),
        "GradientBoosting": GBTRegressor(featuresCol="features", labelCol="math", maxIter=50, seed=SEED),
    }
    rmse = RegressionEvaluator(labelCol="math", predictionCol="prediction", metricName="rmse")
    r2 = RegressionEvaluator(labelCol="math", predictionCol="prediction", metricName="r2")

    # Baseline: always predict the training mean.
    mean_math = train.agg(F.avg("math")).first()[0]
    base = test.withColumn("prediction", F.lit(float(mean_math)))
    results = {"Baseline (mean)": {"rmse": round(rmse.evaluate(base), 3), "r2": round(r2.evaluate(base), 3)}}

    for name, model in models.items():
        fitted = Pipeline(stages=prep + [model]).fit(train)
        pred = fitted.transform(test)
        results[name] = {"rmse": round(rmse.evaluate(pred), 3), "r2": round(r2.evaluate(pred), 3)}
        print(f"  {name:18s} RMSE {results[name]['rmse']}  R2 {results[name]['r2']}")
    print("  Baseline (mean)    RMSE", results["Baseline (mean)"]["rmse"])
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="path to StudentsPerformance.csv")
    ap.add_argument("--out", default="reports", help="folder for results")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    spark = (SparkSession.builder.appName("student-performance").master("local[*]")
             .config("spark.sql.shuffle.partitions", "8").getOrCreate())
    spark.sparkContext.setLogLevel("ERROR")

    df = load(spark, args.data).cache()
    summary = explore(spark, df, args.out)
    summary["math_prediction_test_set"] = predict_math(df, args.out)

    with open(os.path.join(args.out, "results.json"), "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nSaved results, tables and chart to {args.out}/")
    spark.stop()


if __name__ == "__main__":
    main()
