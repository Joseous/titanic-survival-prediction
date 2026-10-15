# Loading relevant libraries
import pandas as pd
import joblib
from pathlib import Path

# Path
MODEL_PATH = "../models/titanic_model.joblib"

TEST_PATH = "../data/test.csv"

OUTPUT_PATH = "../submissions/submissions.csv"

# Load the the model

model = joblib.load(MODEL_PATH)

# Load the test datset
test_df = pd.read_csv(TEST_PATH)

# Keep passenger IDS
passenger_ids = test_df["PassengerId"]

# Feature Engineering
X_test = test_df[["Pclass",	"Sex", "Age", "SibSp",	"Parch", "Fare", "Embarked"]]

# Make predicction

predictions = model.predict(X_test)

# Create submission
submission = pd.DataFrame({
                "PassengerId" : passenger_ids,
                "Survived" : predictions
})

# Saving the submissions
submission.to_csv(OUTPUT_PATH, index=False)

print("Submission saved in", OUTPUT_PATH)