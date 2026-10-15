# Titanic Survival Prediction

A machine learning classification project that predicts whether a passenger survived the Titanic disaster based on passenger characteristics such as passenger class, sex, age, family relationships, fare, and port of embarkation.

The project demonstrates a complete supervised machine learning workflow, including data inspection, preprocessing, feature engineering/selection, model comparison, cross-validation, hyperparameter tuning, final model training, evaluation, model persistence, and inference on new passenger data.

## Project Overview

The objective of this project is to build a binary classification model capable of predicting Titanic passenger survival.

The target variable is:

- `0` — Did not survive
- `1` — Survived

The project follows a pipeline-based approach so that preprocessing operations and model training are handled consistently. Several classification algorithms are evaluated before Random Forest is selected and optimized through multiple hyperparameter-search techniques.

## Objectives

The main objectives of the project are to:

1. Load and inspect the Titanic dataset.
2. Identify data types, missing values, duplicates, and basic statistics.
3. Remove features that are not used for prediction.
4. Handle missing numerical and categorical values.
5. Scale numerical features.
6. Encode categorical features.
7. Compare several classification algorithms.
8. Use stratified cross-validation to obtain more reliable model-performance estimates.
9. Evaluate models using F1-score and ROC-AUC.
10. Tune the Random Forest hyperparameters using:
   - Grid Search
   - Randomized Search
   - Bayesian Optimization with Optuna
11. Train the final Random Forest model.
12. Evaluate the final model on a held-out test set.
13. Save the trained pipeline with Joblib.
14. Load the saved model and make predictions for new passengers.

## Dataset

The project uses the Titanic training dataset:

```text
../data/train.csv
```

The original dataset contains:

- **891 observations**
- **12 columns**

The columns found during the initial data checks were:

| Column | Description / Role |
|---|---|
| `PassengerId` | Passenger identifier |
| `Survived` | Target variable |
| `Pclass` | Passenger class |
| `Name` | Passenger name |
| `Sex` | Passenger sex |
| `Age` | Passenger age |
| `SibSp` | Number of siblings/spouses aboard |
| `Parch` | Number of parents/children aboard |
| `Ticket` | Ticket identifier |
| `Fare` | Passenger fare |
| `Cabin` | Cabin information |
| `Embarked` | Port of embarkation |

### Target Distribution

The target variable is `Survived`.

From the dataset:

- `0` represents passengers who did not survive.
- `1` represents passengers who survived.
- The target mean is approximately `0.3838`, meaning about 38.4% of observations have a target value of 1.

## Features

The project excludes the following columns from model input:

```text
PassengerId
Name
Ticket
Survived
```

The remaining predictive features are divided into numerical and categorical groups.

### Numerical Features

```text
Age
SibSp
Parch
Fare
```

### Categorical Features

```text
Pclass
Sex
Embarked
```

Although `Pclass` is represented numerically in the source data, it is treated as a categorical feature in the preprocessing pipeline.

## Project Workflow

The machine learning workflow is:

```text
Raw Titanic Dataset
        |
        v
Data Inspection
        |
        v
Data Quality Checks
        |
        v
Remove Cabin
        |
        v
Separate Features and Target
        |
        v
Train/Test Split
        |
        v
Numerical + Categorical Preprocessing
        |
        v
Model Comparison
        |
        v
Cross-Validation
        |
        v
Hyperparameter Tuning
        |
        v
Final Random Forest Model
        |
        v
Test-Set Evaluation
        |
        v
Save Model
        |
        v
Load Model and Predict
```

## Data Quality and Preprocessing

### Initial Data Checks

The notebook performs checks for:

- Dataset shape
- Column names
- Data types
- Missing values
- Duplicate rows
- Numerical descriptive statistics
- Categorical descriptive statistics

There were **no duplicate rows** in the initial dataset.

### Missing Values

The initial missing-value counts were:

| Column | Missing Values |
|---|---:|
| `Age` | 177 |
| `Cabin` | 687 |
| `Embarked` | 2 |
| Other columns | 0 |

`Cabin` contains a large number of missing observations hence the column was removed:

```python
titanic_df = titanic_df.drop(columns=["Cabin"])
```

### Train/Test Split

The data is divided into training and test sets using:

```python
train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
```

This produces an 80/20 split while preserving the target-class proportions through stratification.

### Numerical Preprocessing

Numerical features are processed using a pipeline containing:

1. Median imputation
2. Standard scaling

```python
numerical_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])
```

Median imputation handles missing `Age` values without requiring rows containing missing values to be removed.

### Categorical Preprocessing

Categorical features are processed using:

1. Most-frequent-value imputation
2. One-hot encoding

```python
categorical_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])
```

`handle_unknown="ignore"` makes the preprocessing pipeline safer when new prediction data contains a category that was not present during training.

### Combined Preprocessor

The numerical and categorical pipelines are combined with `ColumnTransformer`:

```python
preprocessor = ColumnTransformer(transformers=[
    ("num", numerical_pipeline, numerical_columns),
    ("cat", categorical_pipeline, cat_columns)
])
```

This allows different preprocessing operations to be applied automatically to the appropriate feature types.

## Machine Learning Models

Five classification algorithms are compared:

| Model | Configuration |
|---|---|
| Logistic Regression | `max_iter=1000` |
| K-Nearest Neighbors | `n_neighbors=7`, Euclidean distance |
| Random Forest | `n_estimators=300`, `max_depth=15` |
| Support Vector Classifier | RBF kernel, probability estimates enabled |
| Gradient Boosting | `n_estimators=300`, `max_depth=15` |

Each model is placed inside a complete preprocessing-and-model pipeline.

## Cross-Validation

The notebook uses stratified 5-fold cross-validation:

```python
StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)
```

Stratification is useful for classification because each fold attempts to maintain a similar distribution of the target classes.

### F1 Cross-Validation Results

The initial model comparison produced the following mean F1-scores:

| Model | Mean CV F1 | Std. CV F1 |
|---|---:|---:|
| Logistic Regression | 0.7301 | 0.0217 |
| K-Nearest Neighbors | 0.7194 | 0.0229 |
| Random Forest | 0.7270 | 0.0204 |
| Support Vector Classifier | **0.7447** | 0.0311 |
| Gradient Boosting | 0.6977 | 0.0205 |

Based on this initial F1 comparison, the Support Vector Classifier achieved the highest mean F1-score among the five baseline models.

The project subsequently focuses on optimizing Random Forest through several hyperparameter-search approaches.

## Model Comparison

The notebook also evaluates models using ROC-AUC.

ROC-AUC measures the model's ability to distinguish between the two target classes across different classification thresholds.

The test-set ROC curves are generated for the models using predicted probabilities.

The notebook therefore considers more than one performance perspective rather than relying exclusively on accuracy.

## Hyperparameter Tuning

Hyperparameter tuning is used to search for better Random Forest configurations.

The project demonstrates three approaches.

### 1. Grid Search

Grid Search evaluates every combination specified in the parameter grid.

The search space contains:

```python
n_estimators = [100, 200, 300]
max_depth = [None, 5, 10]
min_samples_split = [2, 5]
min_samples_leaf = [1, 2]
```

This produces:

```text
3 × 3 × 2 × 2 = 36 combinations
```

The best Grid Search result was:

```text
F1-score: 0.7584169773
```

Best parameters:

```text
max_depth = None
min_samples_leaf = 1
min_samples_split = 5
n_estimators = 100
```

### 2. Randomized Search

Randomized Search samples a specified number of combinations instead of evaluating every possible combination.

The notebook uses:

```python
n_iter=30
```

The search space includes:

```python
n_estimators = [100, 200, 300, 400, 500, 600]
max_depth = [None, 4, 5, 6, 7, 10]
min_samples_split = [2, 3, 4, 5]
min_samples_leaf = [1, 2, 3, 4]
```

The best Randomized Search result was:

```text
F1-score: 0.7538035271
```

Best parameters:

```text
n_estimators = 100
min_samples_split = 5
min_samples_leaf = 1
max_depth = 10
```

### 3. Bayesian Optimization with Optuna

The project also uses Optuna for Bayesian-style hyperparameter optimization.

The search space includes:

```text
n_estimators: 100–1000
max_depth: 3–30
min_samples_split: 2–20
min_samples_leaf: 1–20
```

The study runs:

```python
study.optimize(objective, n_trials=50)
```

The best recorded configuration was:

```text
n_estimators = 948
max_depth = 21
min_samples_split = 9
min_samples_leaf = 2
```

with a best cross-validation F1-score of approximately:

```text
0.75536
```

## Final Model

The final model is a Random Forest classifier integrated into the preprocessing pipeline:

```python
rf_pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", RandomForestClassifier(
        n_estimators=948,
        max_depth=21,
        min_samples_split=9,
        min_samples_leaf=2
    ))
])
```

The final pipeline is trained on:

```python
X_train
y_train
```

The use of a single pipeline means preprocessing and prediction remain connected, reducing the risk of applying inconsistent transformations during inference.

## Model Evaluation

The final model is evaluated on the held-out test set using a classification report.

### Test-Set Results

| Class | Precision | Recall | F1-score | Support |
|---|---:|---:|---:|---:|
| 0 | 0.81 | 0.92 | 0.86 | 110 |
| 1 | 0.83 | 0.65 | 0.73 | 69 |
| **Accuracy** | | | **0.82** | **179** |
| Macro Avg | 0.82 | 0.79 | 0.80 | 179 |
| Weighted Avg | 0.82 | 0.82 | 0.81 | 179 |

### Interpretation

The final model achieved approximately **82% accuracy** on the held-out test set.

For class `0`, the model achieved:

- Precision: `0.81`
- Recall: `0.92`
- F1-score: `0.86`

For class `1`, the model achieved:

- Precision: `0.83`
- Recall: `0.65`
- F1-score: `0.73`

This indicates that the model was better at identifying class `0` than class `1`. In particular, the lower recall for class `1` means that some passengers belonging to the positive survival class were not identified by the model.

Because the project uses a classification problem with two classes, it is important to consider class-specific precision and recall rather than relying only on overall accuracy.

## Model Persistence

The complete trained pipeline is saved using Joblib:

```python
joblib.dump(
    rf_pipeline,
    "../models/titanic_model.joblib"
)
```

The saved artifact is:

```text
../models/titanic_model.joblib
```

Because the preprocessing steps are part of the pipeline, the saved model contains both the transformations and the trained Random Forest estimator.

## Making Predictions

The saved model can be loaded with:

```python
model = joblib.load("../models/titanic_model.joblib")
```

A new passenger is represented as a pandas DataFrame:

```python
new_passenger = pd.DataFrame({
    "Pclass": [1],
    "SibSp": [2],
    "Sex": ["female"],
    "Age": [12],
    "Parch": [1],
    "Fare": [54.72],
    "Embarked": ["S"]
})
```

The model can then generate a class prediction:

```python
model.predict(new_passenger)
```

For the example included in the notebook, the prediction was:

```text
[1]
```

The model can also return the probability of the positive class:

```python
model.predict_proba(new_passenger)[:, 1]
```

For the example passenger, the recorded probability of class `1` was approximately:

```text
0.9334
```

This means the model assigned a high predicted probability to the positive class for that particular input.

## Project Structure

The notebook uses relative paths indicating a project structure similar to:

```text
titanic-survival-prediction/
│
├── data/
│   └── train.csv
│
├── models/
│   └── titanic_model.joblib
│
├── notebooks/
│   └── titanic.ipynb
│
└── README.md
```

The exact directory layout outside the notebook is not fully defined by the notebook itself, but the relative paths indicate that the dataset is stored under `data/` and the saved model under `models/`.

## Technologies and Libraries

The project is implemented in Python and uses the following libraries:

### Data Manipulation

- **Pandas** — loading and manipulating tabular data
- **NumPy** — numerical operations

### Data Visualization

- **Matplotlib** — plotting
- **Seaborn** — statistical visualization

### Machine Learning

- **Scikit-learn**
  - `train_test_split`
  - `StratifiedKFold`
  - `cross_val_score`
  - `GridSearchCV`
  - `RandomizedSearchCV`
  - `SimpleImputer`
  - `StandardScaler`
  - `OneHotEncoder`
  - `ColumnTransformer`
  - `Pipeline`
  - `LogisticRegression`
  - `RandomForestClassifier`
  - `GradientBoostingClassifier`
  - `SVC`
  - `KNeighborsClassifier`
  - Classification and ROC-AUC metrics

### Hyperparameter Optimization

- **Optuna** — Bayesian optimization / automated hyperparameter search

### Model Persistence

- **Joblib** — saving and loading the trained machine learning pipeline

## Installation

Create and activate a Python virtual environment, then install the required packages.

Example:

```bash
python -m venv titanicvenv
```

Windows:

```bash
titanicvenv\Scripts\activate
```

Install the dependencies:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost optuna joblib jupyter
```

## Running the Project

1. Clone or download the project.
2. Ensure the Titanic training data is available at:

```text
data/train.csv
```

3. Ensure the notebook is located in the expected project directory.
4. Activate the virtual environment.
5. Launch Jupyter Notebook:

```bash
jupyter notebook
```

6. Open:

```text
titanic.ipynb
```

7. Run the notebook cells sequentially.

The notebook performs data loading, preprocessing, model evaluation, hyperparameter optimization, final model training, model saving, and prediction.

## Key Findings

Several useful observations emerge from the project:

### 1. Preprocessing is essential

The dataset contains missing values, particularly in `Age`, `Cabin`, and `Embarked`. The project addresses these issues using explicit preprocessing pipelines.

### 2. Different algorithms produce different results

The initial cross-validation comparison showed that the Support Vector Classifier produced the highest mean F1-score among the baseline models, while Gradient Boosting produced the lowest.

### 3. Hyperparameter tuning improved Random Forest performance

The untuned Random Forest had a mean cross-validation F1-score of approximately `0.7270`.

The tuning experiments produced higher F1-scores:

- Grid Search: `0.7584`
- Randomized Search: `0.7538`
- Optuna: `0.7554`

The best recorded tuning result came from Grid Search on the evaluated configurations, while the final model uses the Optuna-selected parameters.

### 4. Accuracy is not the whole story

The final model achieved 82% accuracy, but class-level recall differs substantially.

The recall for class `0` was `0.92`, compared with `0.65` for class `1`.

This demonstrates why precision, recall, and F1-score should be considered alongside accuracy.

## Conclusion

This project demonstrates an end-to-end machine learning workflow for Titanic survival classification.

The workflow covers data inspection, missing-value handling, categorical encoding, numerical scaling, train/test splitting, model comparison, stratified cross-validation, hyperparameter optimization, final model training, evaluation, model persistence, and inference.

The final Random Forest pipeline achieved approximately **82% accuracy** on the held-out test set, with an F1-score of **0.73 for the positive class**.

More importantly, the project demonstrates how a machine learning model should be developed as a reproducible pipeline rather than simply fitting an estimator directly to raw data. The use of `ColumnTransformer`, `Pipeline`, cross-validation, hyperparameter tuning, and Joblib model persistence provides a solid foundation for extending the project toward a production-ready prediction application.
