# main_rf.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from joblib import dump


# ---------- Load data ----------
def load_data(path="data.csv"):
    df = pd.read_csv(path)

    # Drop id if present
    df = df.drop(columns=["id"], errors="ignore")

    # Target and features
    y = df["Response"]
    X = df.drop(columns=["Response"])

    return X, y


# ---------- Build Random Forest pipeline ----------
def build_pipeline(X: pd.DataFrame):
    cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    num_cols = [c for c in X.columns if c not in cat_cols]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", "passthrough", num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ]
    )

    model = RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced"
    )

    pipeline = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", model),
        ]
    )

    return pipeline


# ---------- Train & save ----------
def main():
    X, y = load_data("data.csv")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    model = build_pipeline(X_train)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print(f"Accuracy: {acc:.4f}")
    print(classification_report(y_test, y_pred))

    # ✅ SAVE MODEL (THIS WAS MISSING)
    dump(model, "rf_insurance_model.joblib")
    print(" Model saved as rf_insurance_model.joblib")


if __name__ == "__main__":
    main()