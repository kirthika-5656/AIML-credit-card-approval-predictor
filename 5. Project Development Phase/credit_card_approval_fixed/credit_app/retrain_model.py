"""
retrain_model.py
================
Run this once to retrain and save the model using YOUR installed
scikit-learn version so it is fully compatible with app.py.

Usage:
    python retrain_model.py

Place this file in the same folder as app.py alongside:
    application_record.csv
    credit_record.csv
"""

import os
import pickle
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ── Load data ──────────────────────────────────────────────────────────────────
print("Loading data...")
app    = pd.read_csv(os.path.join(BASE_DIR, "application_record.csv"))
credit = pd.read_csv(os.path.join(BASE_DIR, "credit_record.csv"))

# ── Build target label ─────────────────────────────────────────────────────────
# STATUS 2-5 = overdue 60+ days → bad applicant → Rejected (0)
# All others (X, C, 0, 1)       → good applicant → Approved (1)
bad_ids = credit[credit["STATUS"].isin(["2", "3", "4", "5"])]["ID"].unique()
label_df = credit[["ID"]].drop_duplicates().copy()
label_df["TARGET"] = (~label_df["ID"].isin(bad_ids)).astype(int)

df = app.merge(label_df, on="ID", how="inner")
print(f"Merged dataset: {df.shape[0]:,} rows | Approval rate: {df['TARGET'].mean():.1%}")

# ── Encode categoricals ────────────────────────────────────────────────────────
cat_cols = [
    "CODE_GENDER", "FLAG_OWN_CAR", "FLAG_OWN_REALTY",
    "NAME_INCOME_TYPE", "NAME_EDUCATION_TYPE",
    "NAME_FAMILY_STATUS", "NAME_HOUSING_TYPE", "OCCUPATION_TYPE",
]

encoders = {}
for col in cat_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    encoders[col] = le          # kept in case you want to inspect mappings

df["CNT_FAM_MEMBERS"] = df["CNT_FAM_MEMBERS"].fillna(df["CNT_FAM_MEMBERS"].median())

# ── Features ───────────────────────────────────────────────────────────────────
FEATURE_COLS = [
    "CODE_GENDER", "FLAG_OWN_CAR", "FLAG_OWN_REALTY", "CNT_CHILDREN",
    "AMT_INCOME_TOTAL", "NAME_INCOME_TYPE", "NAME_EDUCATION_TYPE",
    "NAME_FAMILY_STATUS", "NAME_HOUSING_TYPE", "DAYS_BIRTH",
    "DAYS_EMPLOYED", "FLAG_MOBIL", "FLAG_WORK_PHONE", "FLAG_PHONE",
    "FLAG_EMAIL", "OCCUPATION_TYPE", "CNT_FAM_MEMBERS",
]

X = df[FEATURE_COLS]
y = df["TARGET"]

# ── Train ──────────────────────────────────────────────────────────────────────
print("Training DecisionTreeClassifier...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

clf = DecisionTreeClassifier(max_depth=8, random_state=42)
clf.fit(X_train, y_train)

acc = clf.score(X_test, y_test)
print(f"Test accuracy : {acc:.4f}")

# ── Save ───────────────────────────────────────────────────────────────────────
out_path = os.path.join(BASE_DIR, "model__1_.pkl")
with open(out_path, "wb") as f:
    pickle.dump(clf, f)

print(f"Model saved → {out_path}")
print("\nDone! You can now run:  python app.py")
