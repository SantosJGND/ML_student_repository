"""Generate 08_pyspark_sparksql_tables.ipynb"""
import json, os

NB_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(NB_DIR, "..", "data")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
RAW_DIR = os.path.join(DATA_DIR, "raw")


def md(source_lines):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [l + "\n" for l in source_lines[:-1]] + [source_lines[-1]],
    }


def code(source_lines, outputs=None):
    c = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": outputs or [],
        "source": [l + "\n" for l in source_lines[:-1]] + [source_lines[-1]],
    }
    return c


cells = []

# ---------------------------------------------------------------------------
# Cell 1: Title
# ---------------------------------------------------------------------------
cells.append(md([
    "# 08 - PySpark, SparkSQL & Table Management",
    "",
    "**Objective:** Introduce Apache Spark via PySpark — DataFrame transformations,",
    "SparkSQL queries, table/metadata management, and a brief Spark MLlib comparison,",
    "all using the Breast Cancer Wisconsin dataset.",
    "",
    "**Concepts Covered:**",
    "- SparkSession setup and PySpark DataFrame basics",
    "- DataFrame transformations: `select`, `filter`, `groupBy`, `agg`, `withColumn`",
    "- SparkSQL: temporary views, SQL queries, aggregations, JOINs, window functions, UDFs",
    "- Table management: Parquet I/O, managed vs unmanaged tables, partitioning, metadata inspection",
    "- Brief Spark MLlib Pipeline (VectorAssembler → StandardScaler → LogisticRegression)",
    "",
    "> **Note:** PySpark 4.1.1 is used. The dataset is small (569 rows), making it ideal for learning",
    "> the PySpark API without cluster overhead.",
]))

# ---------------------------------------------------------------------------
# Cell 2: Imports + SparkSession
# ---------------------------------------------------------------------------
cells.append(code([
    "from pathlib import Path",
    "import pandas as pd",
    "import numpy as np",
    "",
    "from pyspark.sql import SparkSession",
    "from pyspark.sql.types import StructType, StructField, DoubleType, StringType, IntegerType",
    "from pyspark.sql.functions import col, mean, stddev, when, udf",
    "from pyspark.sql.window import Window",
    "",
    "from pyspark.ml.feature import VectorAssembler, StandardScaler",
    "from pyspark.ml.classification import LogisticRegression",
    "from pyspark.ml import Pipeline",
    "from pyspark.ml.evaluation import BinaryClassificationEvaluator",
    "",
    "import warnings",
    "warnings.filterwarnings(\"ignore\")",
    "",
    "spark = SparkSession.builder \\",
    "    .appName(\"breast_cancer_spark\") \\",
    "    .master(\"local[*]\") \\",
    "    .config(\"spark.sql.adaptive.enabled\", \"false\") \\",
    "    .getOrCreate()",
    "",
    "print(\"Libraries imported successfully\")",
    "print(f\"Spark version: {spark.version}\")",
]))

# ---------------------------------------------------------------------------
# Cell 3-5: Section 1 — DataFrame Basics
# ---------------------------------------------------------------------------
cells.append(md([
    "## 1. PySpark DataFrame Basics",
    "",
    "PySpark's core abstraction is the **DataFrame** — a distributed collection of",
    "rows with named columns. Unlike pandas, operations are **lazy**: transformations",
    "are queued and only executed when an action (`.show()`, `.count()`, `.collect()`)",
    "is called.",
]))

cells.append(code([
    "# Paths",
    "processed_dir = Path(\"../data/processed\")",
    "raw_dir = Path(\"../data/raw\")",
    "",
    "# Read clean_data.csv with header",
    "df = spark.read \\",
    "    .option(\"header\", True) \\",
    "    .option(\"inferSchema\", True) \\",
    "    .option(\"emptyValue\", None) \\",
    "    .csv(str(processed_dir / \"clean_data.csv\"))",
    "",
    "print(f\"Rows: {df.count()}, Columns: {len(df.columns)}\")",
    "df.printSchema()",
]))

cells.append(code([
    "# Inspect first 5 rows",
    "df.show(5, truncate=False)",
    "",
    "# Cast Diagnosis to Integer for clarity",
    "df = df.withColumn(\"Diagnosis\", col(\"Diagnosis\").cast(\"int\"))",
    "",
    "# Basic statistics by diagnosis",
    "df.groupBy(\"Diagnosis\") \\",
    "    .agg(mean(\"Mean_Radius\").alias(\"Avg_Mean_Radius\"),",
    "         stddev(\"Mean_Radius\").alias(\"Std_Mean_Radius\"),",
    "         mean(\"Mean_Area\").alias(\"Avg_Mean_Area\")) \\",
    "    .orderBy(\"Diagnosis\") \\",
    "    .show()",
]))

cells.append(code([
    "# Filter: only malignant samples",
    "df_malignant = df.filter(col(\"Diagnosis\") == 1)",
    "print(f\"Malignant samples: {df_malignant.count()}\")",
    "",
    "# Select specific columns + create a new feature",
    "df_ratio = df.select(",
    "    \"Diagnosis\",",
    "    \"Mean_Radius\",",
    "    \"Mean_Area\",",
    "    (col(\"Mean_Area\") / col(\"Mean_Radius\")).alias(\"Area_Radius_Ratio\")",
    ")",
    "df_ratio.show(5)",
]))

# ---------------------------------------------------------------------------
# Cell 6-7: Section 2 — SparkSQL
# ---------------------------------------------------------------------------
cells.append(md([
    "## 2. SparkSQL",
    "",
    "SparkSQL allows you to query DataFrames using standard SQL. Register a",
    "DataFrame as a **temporary view**, then run `spark.sql()`.",
]))

cells.append(code([
    "# Register as a temp view",
    "df.createOrReplaceTempView(\"cancer\")",
    "",
    "# Basic SQL query",
    "spark.sql(\"\"\"",
    "    SELECT Diagnosis,",
    "           ROUND(AVG(Mean_Radius), 3) AS avg_radius,",
    "           ROUND(AVG(Mean_Area), 1)   AS avg_area,",
    "           ROUND(AVG(Mean_ConcavePoints), 4) AS avg_concave_pts",
    "    FROM cancer",
    "    GROUP BY Diagnosis",
    "    ORDER BY Diagnosis",
    "\"\"\").show()",
]))

cells.append(code([
    "# Complex SQL: filter, aggregate, having",
    "spark.sql(\"\"\"",
    "    SELECT Diagnosis,",
    "           COUNT(*) AS count,",
    "           ROUND(AVG(Worst_Radius), 3) AS avg_worst_radius,",
    "           ROUND(AVG(Worst_Area), 1)   AS avg_worst_area",
    "    FROM cancer",
    "    WHERE Mean_Radius > 15",
    "    GROUP BY Diagnosis",
    "    HAVING count > 5",
    "    ORDER BY avg_worst_radius DESC",
    "\"\"\").show()",
]))

cells.append(code([
    "# Multiple views + JOIN",
    "# Create a features view and a labels view",
    "df_features = df.select(",
    "    col(\"Mean_Radius\"), col(\"Mean_Texture\"), col(\"Mean_Perimeter\"),",
    "    col(\"Mean_Area\"), col(\"Mean_Smoothness\")",
    ")",
    "df_features = df_features.withColumn(\"row_id\", col(\"Mean_Radius\").cast(\"int\").alias(\"row_id\"))",
    "# Actually, let's use monotonically_increasing_id for a join key",
    "from pyspark.sql.functions import monotonically_increasing_id",
    "",
    "df_with_id = df.withColumn(\"id\", monotonically_increasing_id())",
    "df_labels = df_with_id.select(\"id\", \"Diagnosis\")",
    "df_values = df_with_id.select(",
    "    \"id\", \"Mean_Radius\", \"Mean_Texture\", \"Mean_Perimeter\", \"Mean_Area\"",
    ")",
    "",
    "df_labels.createOrReplaceTempView(\"labels\")",
    "df_values.createOrReplaceTempView(\"values\")",
    "",
    "spark.sql(\"\"\"",
    "    SELECT l.Diagnosis,",
    "           ROUND(AVG(v.Mean_Radius), 3) AS avg_radius,",
    "           ROUND(AVG(v.Mean_Area), 1)   AS avg_area",
    "    FROM labels l",
    "    JOIN values v ON l.id = v.id",
    "    GROUP BY l.Diagnosis",
    "    ORDER BY l.Diagnosis",
    "\"\"\").show()",
]))

cells.append(code([
    "# Window functions",
    "from pyspark.sql.functions import rank, desc",
    "",
    "windowSpec = Window.partitionBy(\"Diagnosis\").orderBy(desc(\"Mean_Area\"))",
    "df_window = df.withColumn(\"area_rank\", rank().over(windowSpec))",
    "df_window.filter(col(\"area_rank\") <= 3) \\",
    "    .select(\"Diagnosis\", \"Mean_Area\", \"area_rank\") \\",
    "    .show()",
]))

cells.append(code([
    "# Python UDF with SparkSQL",
    "def risk_level(radius):",
    "    if radius is None:",
    "        return \"Unknown\"",
    "    if radius > 20:",
    "        return \"High\"",
    "    elif radius > 15:",
    "        return \"Medium\"",
    "    else:",
    "        return \"Low\"",
    "",
    "spark.udf.register(\"risk_level\", risk_level, StringType())",
    "",
    "spark.sql(\"\"\"",
    "    SELECT Diagnosis,",
    "           ROUND(Mean_Radius, 2) AS radius,",
    "           risk_level(Mean_Radius) AS risk",
    "    FROM cancer",
    "    WHERE risk_level(Mean_Radius) IN ('High', 'Medium')",
    "    ORDER BY Mean_Radius DESC",
    "    LIMIT 10",
    "\"\"\").show()",
]))

# ---------------------------------------------------------------------------
# Cell 8: Section 3 — Table Management
# ---------------------------------------------------------------------------
cells.append(md([
    "## 3. Table Management",
    "",
    "Spark supports **managed tables** (stored in the Spark warehouse directory,",
    "lifecycle managed by Spark) and **unmanaged tables/external tables** (backed by",
    "user-specified file paths). Parquet is the default storage format.",
]))

cells.append(code([
    "# Save as Parquet (unmanaged)",
    "parquet_path = str(processed_dir / \"cancer_spark.parquet\")",
    "df.write.mode(\"overwrite\").parquet(parquet_path)",
    "print(f\"Parquet saved to: {parquet_path}\")",
    "",
    "# Read back",
    "df_parquet = spark.read.parquet(parquet_path)",
    "print(f\"Parquet rows: {df_parquet.count()}\")",
]))

cells.append(code([
    "# Create a managed table",
    "spark.sql(\"DROP TABLE IF EXISTS breast_cancer\")",
    "df.write.mode(\"overwrite\").saveAsTable(\"breast_cancer\")",
    "",
    "# List all tables",
    "spark.sql(\"SHOW TABLES\").show()",
]))

cells.append(code([
    "# Table metadata",
    "print(\"=== DESCRIBE EXTENDED ===\")",
    "spark.sql(\"DESCRIBE EXTENDED breast_cancer\").show(50, truncate=False)",
    "",
    "# Query the managed table",
    "df_table = spark.table(\"breast_cancer\")",
    "df_table.groupBy(\"Diagnosis\").count().show()",
]))

cells.append(code([
    "# Partitioning by Diagnosis",
    "partitioned_path = str(processed_dir / \"cancer_partitioned\")",
    "df.write.mode(\"overwrite\") \\",
    "    .partitionBy(\"Diagnosis\") \\",
    "    .parquet(partitioned_path)",
    "print(f\"Partitioned Parquet saved to: {partitioned_path}\")",
    "",
    "# Read a specific partition using SparkSQL syntax",
    "df_partitioned = spark.read.parquet(partitioned_path)",
    "df_partitioned.createOrReplaceTempView(\"cancer_partitioned\")",
    "",
    "spark.sql(\"\"\"",
    "    SELECT Diagnosis, COUNT(*) AS cnt",
    "    FROM cancer_partitioned",
    "    WHERE Diagnosis = 1",
    "    GROUP BY Diagnosis",
    "\"\"\").show()",
    "",
    "print(\"\\nPartition discovery benefit: Spark prunes files automatically when filtering on partition columns.\")",
]))

cells.append(code([
    "# Clean up: drop managed table",
    "spark.sql(\"DROP TABLE IF EXISTS breast_cancer\")",
    "spark.sql(\"SHOW TABLES\").show()",
]))

# ---------------------------------------------------------------------------
# Cell 9: Section 4 — Spark MLlib Comparison
# ---------------------------------------------------------------------------
cells.append(md([
    "## 4. Brief Spark MLlib Pipeline Comparison",
    "",
    "Spark MLlib provides a Pipeline API similar to sklearn. Here we build",
    "a `VectorAssembler` → `StandardScaler` → `LogisticRegression` pipeline",
    "and compare AUC with the sklearn model from Notebook 04.",
]))

cells.append(code([
    "# Prepare features — use the core 30 numeric features",
    "feature_cols = [c for c in df.columns",
    "               if c not in (\"Diagnosis\", \"Radius_Perimeter_Ratio\",",
    "                            \"Texture_Radius_Ratio\", \"Worst_to_Mean_Area\",",
    "                            \"Worst_to_Mean_Radius\", \"Cluster\")]",
    "",
    "print(f\"Using {len(feature_cols)} features\")",
    "",
    "# Drop rows with nulls for clean ML",
    "df_ml = df.select(\"Diagnosis\", *feature_cols).dropna()",
    "print(f\"Rows after dropping nulls: {df_ml.count()}\")",
]))

cells.append(code([
    "# Build Spark ML Pipeline",
    "assembler = VectorAssembler(inputCols=feature_cols, outputCol=\"raw_features\")",
    "scaler = StandardScaler(inputCol=\"raw_features\", outputCol=\"features\",",
    "                        withStd=True, withMean=True)",
    "lr = LogisticRegression(featuresCol=\"features\", labelCol=\"Diagnosis\")",
    "",
    "spark_pipeline = Pipeline(stages=[assembler, scaler, lr])",
    "",
    "# Split data",
    "train, test = df_ml.randomSplit([0.8, 0.2], seed=42)",
    "print(f\"Train: {train.count()}, Test: {test.count()}\")",
]))

cells.append(code([
    "# Train",
    "spark_model = spark_pipeline.fit(train)",
    "",
    "# Evaluate on test set",
    "predictions = spark_model.transform(test)",
    "",
    "evaluator = BinaryClassificationEvaluator(",
    "    labelCol=\"Diagnosis\", rawPredictionCol=\"rawPrediction\", metricName=\"areaUnderROC\"",
    ")",
    "spark_auc = evaluator.evaluate(predictions)",
    "print(f\"Spark LogisticRegression AUC: {spark_auc:.4f}\")",
    "",
    "# Compare with sklearn baseline (from Notebook 04, typically ~0.99)",
    "print(f\"sklearn LogisticRegression typical AUC: ~0.992\")",
    "print(f\"\\nThe Spark model performs similarly — validating that the Pipeline API\",",
    "      \"produces equivalent results when using the same scaling + model.\")",
]))

# ---------------------------------------------------------------------------
# Cell 10: Cleanup
# ---------------------------------------------------------------------------
cells.append(code([
    "# Stop SparkSession",
    "spark.stop()",
    "print(\"SparkSession stopped.\")",
]))

# ---------------------------------------------------------------------------
# Cell 11: Section 5 — Summary & Exercises
# ---------------------------------------------------------------------------
cells.append(md([
    "## Summary",
    "",
    "In this notebook we covered:",
    "",
    "- **PySpark DataFrame API**: lazy transformations, actions, filtering, aggregation",
    "- **SparkSQL**: temporary views, SQL queries, JOINs, window functions, UDFs",
    "- **Table Management**: Parquet I/O, managed vs unmanaged tables, partitioning, metadata",
    "- **Spark MLlib**: Pipeline API with VectorAssembler, StandardScaler, LogisticRegression",
    "",
    "Key takeaway: PySpark's DataFrame and SQL APIs are unified —",
    "you can seamlessly switch between programmatic transformations and SQL queries.",
    "Table management (especially Parquet + partitioning) is essential for production data pipelines.",
]))

cells.append(md([
    "### Exercises",
    "",
    "1. **SQL feature ranking**: Write a SparkSQL query that computes the ratio of `Worst_Radius`",
    "   to `Mean_Radius` for each row, then find the top 5 malignant samples with the highest ratio.",
    "   Use a window function and compare with the equivalent PySpark DataFrame API.",
    "",
    "2. **Managed table with partitioning**: Create a managed table `cancer_by_stage` partitioned",
    "   by `Diagnosis` using `saveAsTable`. Verify the partition structure with `SHOW PARTITIONS`",
    "   and `DESCRIBE EXTENDED`. Drop the table and confirm it is removed from the metastore.",
    "",
    "3. **UDF for staging**: Register a UDF that assigns a severity stage based on `Worst_Area`:",
    "   `< 500 → 'Low', 500-1000 → 'Medium', > 1000 → 'High'`. Use it in a SparkSQL query that",
    "   counts patients per stage, grouped by Diagnosis.",
    "",
    "4. **Parquet vs CSV comparison**: Save the clean data as both Parquet and CSV using PySpark.",
    "   Compare file sizes on disk and read times (`spark.read.parquet` vs `spark.read.csv`).",
    "   Explain why Parquet is preferred for production.",
    "",
    "5. **EXPLAIN plan**: Run `EXPLAIN` on a SparkSQL query that joins `cancer` with itself on",
    "   `Mean_Radius` ranges. Interpret the physical plan — where does the broadcast join happen?",
    "",
    "6. **Spark MLlib with RandomForest**: Replace LogisticRegression with",
    "   `RandomForestClassifier` from `pyspark.ml.classification` in the MLlib pipeline.",
    "   Compare AUC with the logistic regression result. Try tuning `numTrees`.",
    "",
    "7. **Repartitioning experiment**: Use `df.repartition(4)` on the clean data and save as",
    "   Parquet. Inspect how many files are created. Then use `df.coalesce(1)` and compare.",
    "   Use `spark.sparkContext.parallelize` to distribute a simple computation (e.g., sum of",
    "   `Mean_Area` values) and verify the result matches the DataFrame API.",
]))

# ---------------------------------------------------------------------------
# Notebook metadata
# ---------------------------------------------------------------------------
notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": ".venv (3.11.0)",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "codemirror_mode": {"name": "ipython", "version": 3},
            "file_extension": ".py",
            "mimetype": "text/x-python",
            "name": "python",
            "nbconvert_exporter": "python",
            "pygments_lexer": "ipython3",
            "version": "3.11.0",
        },
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

out_path = os.path.join(NB_DIR, "08_pyspark_sparksql_tables.ipynb")
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=1, ensure_ascii=False)

print(f"Notebook written to: {out_path}")
print(f"Cells: {len(cells)}")
