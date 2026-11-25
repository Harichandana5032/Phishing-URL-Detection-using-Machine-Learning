
import pandas as pd
import joblib

from sklearn.model_selection import GroupShuffleSplit
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score
)

from urllib.parse import urlsplit

from url_features import extract_features_dataframe


# --------------------------------------------------
# 1. Load dataset
# --------------------------------------------------

df = pd.read_csv("phishing_dataset.csv")

# Find URL and label columns without case sensitivity
column_lookup = {
    col.strip().lower(): col for col in df.columns
}

if "url" not in column_lookup:
    raise ValueError(
        "Raw URL column not found. "
        "Your dataset must contain a URL column."
    )

if "label" not in column_lookup:
    raise ValueError("Target column 'label' not found.")

url_col = column_lookup["url"]
label_col = column_lookup["label"]

df = df[[url_col, label_col]].copy()

df = df.dropna(subset=[url_col, label_col])
df[url_col] = df[url_col].astype(str).str.strip()

df = df[df[url_col] != ""]

df[label_col] = pd.to_numeric(
    df[label_col], errors="coerce"
)

df = df.dropna(subset=[label_col])
df[label_col] = df[label_col].astype(int)

# Confirm the expected target mapping
if not set(df[label_col].unique()).issubset({0, 1}):
    raise ValueError("Expected labels 0 and 1.")

if df[label_col].nunique() != 2:
    raise ValueError("Both target classes must be present.")

print("Dataset shape:", df.shape)
print(df[label_col].value_counts())


# --------------------------------------------------
# 2. Extract URL features
# --------------------------------------------------

X = extract_features_dataframe(df[url_col])

y = df[label_col].reset_index(drop=True)

print("\nExtracted features:")
print(X.head())

print("\nFeature count:", X.shape[1])


# --------------------------------------------------
# 3. Create domain groups
# --------------------------------------------------

def get_domain(url):

    url = str(url).strip()

    if "://" not in url:
        url = "https://" + url

    hostname = urlsplit(url).hostname

    return hostname.lower() if hostname else "unknown"


groups = df[url_col].apply(get_domain).reset_index(drop=True)


# --------------------------------------------------
# 4. Split by domain
# --------------------------------------------------

# This keeps the same domain out of both train and test
splitter = GroupShuffleSplit(
    n_splits=30,
    test_size=0.20,
    random_state=42
)

selected_split = None

for train_idx, test_idx in splitter.split(X, y, groups):

    if (
        y.iloc[train_idx].nunique() == 2
        and y.iloc[test_idx].nunique() == 2
    ):
        selected_split = (train_idx, test_idx)
        break

if selected_split is None:
    raise ValueError(
        "Could not create a domain-separated split "
        "containing both classes. Check your dataset."
    )

train_idx, test_idx = selected_split

X_train = X.iloc[train_idx]
X_test = X.iloc[test_idx]

y_train = y.iloc[train_idx]
y_test = y.iloc[test_idx]

print("\nTraining rows:", len(X_train))
print("Testing rows:", len(X_test))


# --------------------------------------------------
# 5. Identify numerical and categorical columns
# --------------------------------------------------

numerical_cols = X.select_dtypes(
    include=["number"]
).columns.tolist()

categorical_cols = X.select_dtypes(
    include=["object"]
).columns.tolist()

print("\nNumerical columns:", numerical_cols)
print("\nCategorical columns:", categorical_cols)


# --------------------------------------------------
# 6. Create preprocessing pipelines
# --------------------------------------------------

numerical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("numerical", numerical_pipeline, numerical_cols),
    ("categorical", categorical_pipeline, categorical_cols)
])


# --------------------------------------------------
# 7. Create Logistic Regression pipeline
# --------------------------------------------------

model_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", LogisticRegression(max_iter=2000))
])


# --------------------------------------------------
# 8. Train the model
# --------------------------------------------------

model_pipeline.fit(X_train, y_train)

print("\nModel training completed.")


# --------------------------------------------------
# 9. Evaluate the model
# --------------------------------------------------

y_pred = model_pipeline.predict(X_test)

y_prob = model_pipeline.predict_proba(X_test)[:, 1]

print("\nAccuracy:")
print(accuracy_score(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(
    y_test,
    y_pred,
    target_names=["Phishing", "Legitimate"]
))

print("\nROC-AUC:")
print(roc_auc_score(y_test, y_prob))


# --------------------------------------------------
# 10. Save the complete model
# --------------------------------------------------

model_artifact = {
    "pipeline": model_pipeline,
    "feature_names": X.columns.tolist(),
    "target_mapping": {
        0: "Phishing",
        1: "Legitimate"
    },
    "threshold": 0.5
}

joblib.dump(
    model_artifact,
    "url_phishing_model.pkl"
)

print("\nModel saved successfully!")
