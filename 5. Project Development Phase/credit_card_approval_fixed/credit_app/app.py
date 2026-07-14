"""
app.py – Flask backend for Credit Card Approval Prediction
==========================================================
Run:
    python app.py

Then open http://127.0.0.1:5000/ in your browser.

Dependencies:
    pip install flask scikit-learn pandas numpy
"""

from flask import Flask, render_template, request
import joblib
import numpy as np
import os
from datetime import date

# ── App setup ─────────────────────────────────────────────────────────────────

app = Flask(__name__)

# ── Model loading ──────────────────────────────────────────────────────────────

MODEL_PATH = os.path.join(os.path.dirname(__file__), "model__1_.pkl")

try:
    model = joblib.load(MODEL_PATH)
    print(f"[INFO] Model loaded successfully: {type(model).__name__}")
except FileNotFoundError:
    raise RuntimeError(
        f"Model file not found at '{MODEL_PATH}'. "
        "Place model__1_.pkl in the same folder as app.py."
    )
except Exception as exc:
    raise RuntimeError(f"Failed to load model: {exc}")


# ── Encoding maps ──────────────────────────────────────────────────────────────
# These maps match the LabelEncoder output from the training data.

GENDER_MAP = {'F': 0, 'M': 1}
YES_NO_MAP = {'N': 0, 'Y': 1}

INCOME_TYPE_MAP = {
    'Commercial associate': 0,
    'Pensioner': 1,
    'State servant': 2,
    'Student': 3,
    'Working': 4
}

EDUCATION_MAP = {
    'Academic degree': 0,
    'Higher education': 1,
    'Incomplete higher': 2,
    'Lower secondary': 3,
    'Secondary / secondary special': 4
}

FAMILY_STATUS_MAP = {
    'Civil marriage': 0,
    'Married': 1,
    'Separated': 2,
    'Single / not married': 3,
    'Widow': 4
}

HOUSING_MAP = {
    'Co-op apartment': 0,
    'House / apartment': 1,
    'Municipal apartment': 2,
    'Office apartment': 3,
    'Rented apartment': 4,
    'With parents': 5
}

OCCUPATION_MAP = {
    'Accountants': 0,
    'Cleaning staff': 1,
    'Cooking staff': 2,
    'Core staff': 3,
    'Drivers': 4,
    'HR staff': 5,
    'High skill tech staff': 6,
    'IT staff': 7,
    'Laborers': 8,
    'Low-skill Laborers': 9,
    'Managers': 10,
    'Medicine staff': 11,
    'Private service staff': 12,
    'Realty agents': 13,
    'Sales staff': 14,
    'Secretaries': 15,
    'Security staff': 16,
    'Waiters/barmen staff': 17,
    'nan': 18
}


# ── Helper ─────────────────────────────────────────────────────────────────────

def preprocess_form(form) -> np.ndarray:
    """
    Convert raw form data into the feature vector the model expects.
    Matches the training pipeline: 17 features.
    DAYS_BIRTH and DAYS_EMPLOYED are negative values representing days since today.
    """
    today = date.today()

    # Date of Birth → DAYS_BIRTH (negative)
    dob_str = form.get("DOB", "")
    try:
        dob = date.fromisoformat(dob_str)
        days_birth = -(today - dob).days
    except ValueError:
        days_birth = -12000 # Default approx 33 years

    # Employment Start Date → DAYS_EMPLOYED (negative)
    emp_str = form.get("EMP_DATE", "")
    try:
        emp_date = date.fromisoformat(emp_str)
        days_employed = -(today - emp_date).days
    except ValueError:
        days_employed = -1000 # Default approx 3 years

    features = [
        GENDER_MAP.get(form.get("CODE_GENDER", "M"), 1),
        YES_NO_MAP.get(form.get("FLAG_OWN_CAR", "N"), 0),
        YES_NO_MAP.get(form.get("FLAG_OWN_REALTY", "N"), 0),
        int(form.get("CNT_CHILDREN", 0)),
        float(form.get("AMT_INCOME_TOTAL", 0)),
        INCOME_TYPE_MAP.get(form.get("NAME_INCOME_TYPE", "Working"), 4),
        EDUCATION_MAP.get(form.get("NAME_EDUCATION_TYPE", "Higher education"), 1),
        FAMILY_STATUS_MAP.get(form.get("NAME_FAMILY_STATUS", "Married"), 1),
        HOUSING_MAP.get(form.get("NAME_HOUSING_TYPE", "House / apartment"), 1),
        days_birth,
        days_employed,
        1, # FLAG_MOBIL
        int(form.get("FLAG_WORK_PHONE", 0)),
        int(form.get("FLAG_PHONE", 0)),
        int(form.get("FLAG_EMAIL", 0)),
        OCCUPATION_MAP.get(form.get("OCCUPATION_TYPE", "nan"), 18),
        float(form.get("CNT_FAM_MEMBERS", 1)),
    ]
    
    return np.array(features, dtype=float).reshape(1, -1)



# ── Routes ─────────────────────────────────────────────────────────────────────

@app.route("/")
def home():
    """Landing / overview page."""
    return render_template("home.html")


@app.route("/predict", methods=["GET"])
def predict_form():
    """Render the applicant input form."""
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    """
    Accept form submission, run model inference, display result.
    """
    try:
        features = preprocess_form(request.form)
        prediction = model.predict(features)[0]

        # Model outputs 0 = Rejected, 1 = Approved
        result = "Approved" if int(prediction) == 1 else "Rejected"

    except Exception as exc:
        # Graceful fallback – show error on result page
        result = f"Error: {exc}"

    return render_template("result.html", result=result, inputs=request.form)


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Starting CreditPredict Flask server …")
    print("Open http://127.0.0.1:5000/ in your browser.")
    app.run(debug=True, host="0.0.0.0", port=5000)
