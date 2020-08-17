-- Databricks notebook source
-- MAGIC %md
-- MAGIC # Classification of Fire Calls with Spark SQL
-- MAGIC 
-- MAGIC - This project is a modification of the final project of Distributed Computing with Spark SQL on Coursera.
-- MAGIC - Goal: to predict two of the most common `Call_Type_Group` given information from the rest of the table.

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ## 1. Loading data

-- COMMAND ----------

-- MAGIC %run ./Includes/Classroom-Setup

-- COMMAND ----------

USE DATABRICKS;

CREATE TABLE IF NOT EXISTS fireCallsClean
USING parquet
OPTIONS (
  path "/mnt/davis/fire-calls/fire-calls-clean.parquet"
)

-- COMMAND ----------

-- Sanity check
SELECT * FROM fireCallsClean LIMIT 10

-- COMMAND ----------

ANALYZE TABLE fireCallsClean COMPUTE STATISTICS NOSCAN

-- COMMAND ----------

DESCRIBE EXTENDED fireCallsClean

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ## 2. Data Preparation

-- COMMAND ----------

SELECT 
  Call_Type_Group, 
  COUNT(*) AS calls
FROM fireCallsClean
GROUP BY Call_Type_Group
ORDER BY calls DESC

-- COMMAND ----------

-- MAGIC %md
-- MAGIC Drop all the rows where `Call_Type_Group = null`. 
-- MAGIC 
-- MAGIC Since we don't have a lot of `Call_Type_Group` with the value `Alarm` and `Fire`, we will also drop these calls from the table. 

-- COMMAND ----------

CREATE OR REPLACE VIEW fireCallsGroupCleaned AS (
  SELECT *
  FROM fireCallsClean
  WHERE 
    Call_Type_Group IS NOT NULL
    AND Call_Type_Group NOT IN ('Alarm', 'Fire')
)

-- COMMAND ----------

-- Sanity check
-- Call_Type_Group should be either 'Potentially Life-Threatening' or 'Non Life-threatening'.
SELECT DISTINCT Call_Type_Group
FROM fireCallsGroupCleaned

-- COMMAND ----------

SELECT COUNT(*) AS rows
FROM fireCallsGroupCleaned

-- COMMAND ----------

-- MAGIC %md
-- MAGIC Subset the columns of interest to a view called `fireCallsDF`.
-- MAGIC * "Call_Type"
-- MAGIC * "Fire_Prevention_District"
-- MAGIC * "Neighborhooods_-\_Analysis_Boundaries" 
-- MAGIC * "Number_of_Alarms"
-- MAGIC * "Original_Priority" 
-- MAGIC * "Unit_Type" 
-- MAGIC * "Battalion"
-- MAGIC * "Call_Type_Group"

-- COMMAND ----------

CREATE OR REPLACE VIEW fireCallsDF AS (
  SELECT 
    Call_Type AS call_type,
    Fire_Prevention_District AS fire_prevention_district,
    `Neighborhooods_-_Analysis_Boundaries` AS neighborhoods,
    Number_of_Alarms AS n_alarms,
    Original_Priority AS original_priority, 
    Unit_Type AS unit_type,
    Battalion as battalion,
    Call_Type_Group AS call_type_group
  FROM fireCallsGroupCleaned
)

-- COMMAND ----------

-- MAGIC %md
-- MAGIC Load the `fireCallsDF` table into python.

-- COMMAND ----------

-- MAGIC %python
-- MAGIC df = sql("SELECT * FROM fireCallsDF")
-- MAGIC display(df)

-- COMMAND ----------

-- MAGIC %python
-- MAGIC # Convert the Spark DataFrame to pandas
-- MAGIC pdDF = df.toPandas()
-- MAGIC pdDF.info()

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ### Preprocessing

-- COMMAND ----------

-- MAGIC %python
-- MAGIC import numpy as np
-- MAGIC import pandas as pd
-- MAGIC import seaborn as sns

-- COMMAND ----------

-- MAGIC %md
-- MAGIC #### Encoding `y` as numerical
-- MAGIC Non Life-threatening -> 0
-- MAGIC 
-- MAGIC Potentially Life-Threatening -> 1

-- COMMAND ----------

-- MAGIC %python
-- MAGIC from sklearn.preprocessing import LabelEncoder
-- MAGIC 
-- MAGIC le = LabelEncoder()
-- MAGIC numerical_pdDF = pdDF.apply(le.fit_transform)
-- MAGIC 
-- MAGIC X = numerical_pdDF.drop("call_type_group", axis=1)
-- MAGIC y = numerical_pdDF["call_type_group"].values

-- COMMAND ----------

-- MAGIC %md
-- MAGIC #### Checking class balance

-- COMMAND ----------

-- MAGIC %python
-- MAGIC 
-- MAGIC np.bincount(y)

-- COMMAND ----------

-- MAGIC %md
-- MAGIC #### Feature Engineering

-- COMMAND ----------

-- MAGIC %python
-- MAGIC X.nunique().reset_index().rename(
-- MAGIC     columns={'index':'feature', 0:'n_unique'}).sort_values(by='n_unique', 
-- MAGIC                                                            ascending=False)

-- COMMAND ----------

-- MAGIC %md
-- MAGIC We will drop `n_alarms` as it's invariant.

-- COMMAND ----------

-- MAGIC %python
-- MAGIC X = X.drop("n_alarms", axis=1)

-- COMMAND ----------

-- MAGIC %python
-- MAGIC SEED = 42

-- COMMAND ----------

-- MAGIC %python
-- MAGIC from sklearn.model_selection import train_test_split
-- MAGIC X_train, X_test, y_train, y_test = train_test_split(X, y, 
-- MAGIC                                                     test_size=0.2, 
-- MAGIC                                                     random_state=SEED)

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ## 3. Model
-- MAGIC - We have a binary classification problem
-- MAGIC - A two-step pipeline
-- MAGIC   - One-hot-encoding
-- MAGIC   - Classifier

-- COMMAND ----------

-- MAGIC %python
-- MAGIC from sklearn.preprocessing import OneHotEncoder
-- MAGIC from sklearn.pipeline import make_pipeline
-- MAGIC from sklearn.linear_model import LogisticRegression
-- MAGIC from sklearn.tree import DecisionTreeClassifier
-- MAGIC from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
-- MAGIC from sklearn.metrics import accuracy_score
-- MAGIC 
-- MAGIC def fit_model(X_train, y_train, X_test, y_test):
-- MAGIC   ohe = OneHotEncoder(handle_unknown="ignore")
-- MAGIC   classifiers = [
-- MAGIC     LogisticRegression(), 
-- MAGIC     DecisionTreeClassifier(random_state=SEED), 
-- MAGIC     RandomForestClassifier(random_state=SEED),
-- MAGIC     AdaBoostClassifier(random_state=SEED)
-- MAGIC   ]
-- MAGIC   cols = ["classifier", "accuracy"]
-- MAGIC   results = pd.DataFrame(columns=cols)
-- MAGIC   
-- MAGIC   best_acc, best_model = 0, None
-- MAGIC   
-- MAGIC   for clf in classifiers:
-- MAGIC     pipeline = make_pipeline(ohe, clf)
-- MAGIC     clf_name = clf.__class__.__name__
-- MAGIC     pipeline.fit(X_train, y_train)
-- MAGIC     y_pred = pipeline.predict(X_test)
-- MAGIC     acc = round(100*accuracy_score(y_pred, y_test), 2)
-- MAGIC     res = pd.DataFrame([[clf_name, acc]], columns=cols)
-- MAGIC     results = results.append(res)
-- MAGIC     if acc >= best_acc:
-- MAGIC       best_model = pipeline
-- MAGIC   
-- MAGIC   return best_model, results

-- COMMAND ----------

-- MAGIC %python
-- MAGIC best_model, results = fit_model(X_train, y_train, X_test, y_test)

-- COMMAND ----------

-- MAGIC %python
-- MAGIC results.sort_values(by="accuracy", ascending=False)

-- COMMAND ----------

-- MAGIC %md
-- MAGIC Save pipeline  to disk.

-- COMMAND ----------

-- MAGIC %python
-- MAGIC import mlflow
-- MAGIC from mlflow.sklearn import save_model
-- MAGIC 
-- MAGIC model_path = "/dbfs/" + username + "/call_type_group_rf"
-- MAGIC dbutils.fs.rm(username + "/call_type_group_rf", recurse=True)
-- MAGIC save_model(best_model, model_path)

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ### Prediction with UDF

-- COMMAND ----------

-- MAGIC %md
-- MAGIC Register the `.predict` function of the sklearn pipeline as a UDF which we can use later to apply in parallel. 

-- COMMAND ----------

-- MAGIC %python
-- MAGIC import mlflow
-- MAGIC from mlflow.pyfunc import spark_udf
-- MAGIC 
-- MAGIC predict = spark_udf(spark, model_path, result_type="int")
-- MAGIC spark.udf.register("predictUDF", predict)

-- COMMAND ----------

-- MAGIC %md
-- MAGIC Create a view called `testTable` of our test data `X_test` so that we can see this table in SQL.

-- COMMAND ----------

-- MAGIC %python
-- MAGIC spark_df = spark.createDataFrame(X_test)
-- MAGIC spark_df.createOrReplaceTempView("testTable")

-- COMMAND ----------

USE DATABRICKS;

DROP TABLE IF EXISTS predictions;

CREATE TEMPORARY VIEW predictions AS (
  SELECT 
    CAST(predictUDF(call_type, fire_prevention_district, neighborhoods, 
    original_priority, unit_type, battalion) AS int)
    AS prediction, *
  FROM testTable
  LIMIT 10000)

-- COMMAND ----------

SELECT * FROM predictions LIMIT 10
