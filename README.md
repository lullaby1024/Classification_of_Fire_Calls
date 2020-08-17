# Classification of Fire Calls with Spark SQL
- This project is a modification of the final project of Distributed Computing with Spark SQL on Coursera.
- Goal: to predict two of the most common Call_Type_Group given information from the rest of the table.
  - Binary classification: Potentially Life-Threatening/Non Life-threatening

# Outline
The following are the steps through which this project has taken. Refer to the .sql file for codes.
- Loading data
- Data Preparation
  - Checking class balance
  - Feature engineering
  - Train-test-split
- Model
  - A two-step pipeline: (1) One-hot-encoding (2) Classifier
  - Best model: random forest (test accuracy=0.82)
- Prediction with UDF
  
