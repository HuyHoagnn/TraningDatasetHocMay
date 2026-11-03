from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

PROCESSED_DIR = Path("data/processed")
RESULTS_DIR = Path("results")
MODEL_DIR = Path("models")
TARGET = "Class"
RANDOM_STATE = 42


def build_models():
    return {
        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            random_state=RANDOM_STATE,
            class_weight="balanced",
            n_jobs=-1,
        ),
        "Extra Trees": ExtraTreesClassifier(
            n_estimators=300,
            random_state=RANDOM_STATE,
            class_weight="balanced",
            n_jobs=-1,
        ),
        "HistGradientBoosting": HistGradientBoostingClassifier(
            max_iter=250,
            learning_rate=0.08,
            random_state=RANDOM_STATE,
        ),
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(
                max_iter=2000,
                class_weight="balanced",
                random_state=RANDOM_STATE,
            )),
        ]),
        "SVM RBF": Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", SVC(
                C=10,
                kernel="rbf",
                class_weight="balanced",
                random_state=RANDOM_STATE,
            )),
        ]),
    }


def main():
    train_path = PROCESSED_DIR / "train_data.csv"
    validation_path = PROCESSED_DIR / "validation_data.csv"
    test_path = PROCESSED_DIR / "test_data.csv"

    if not train_path.exists() or not validation_path.exists() or not test_path.exists():
        raise FileNotFoundError(
            "Khong tim thay train_data.csv, validation_data.csv hoac test_data.csv. "
            "Hay chay 04_prepare_data.py truoc."
        )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    train_df = pd.read_csv(train_path)
    validation_df = pd.read_csv(validation_path)
    test_df = pd.read_csv(test_path)
    X_train = train_df.drop(columns=[TARGET])
    y_train = train_df[TARGET]
    X_validation = validation_df.drop(columns=[TARGET])
    y_validation = validation_df[TARGET]
    X_test = test_df.drop(columns=[TARGET])
    y_test = test_df[TARGET]

    results = []
    fitted_models = {}

    print("===== SO SANH CAC MO HINH =====")
    print("Thang do chon mo hinh: Validation Macro F1")

    for name, model in build_models().items():
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
        cv_scores = cross_val_score(
            model,
            X_train,
            y_train,
            cv=cv,
            scoring="f1_macro",
            n_jobs=-1,
        )
        model.fit(X_train, y_train)
        validation_pred = model.predict(X_validation)
        validation_macro_f1 = f1_score(
            y_validation,
            validation_pred,
            average="macro",
        )
        combined_df = pd.concat([train_df, validation_df], ignore_index=True)
        combined_X = combined_df.drop(columns=[TARGET])
        combined_y = combined_df[TARGET]
        model.fit(combined_X, combined_y)
        test_pred = model.predict(X_test)
        fitted_models[name] = model
        results.append({
            "Model": name,
            "CV Macro F1": cv_scores.mean(),
            "CV Macro F1 Std": cv_scores.std(),
            "Validation Macro F1": validation_macro_f1,
            "Test Accuracy": accuracy_score(y_test, test_pred),
            "Test Macro F1": f1_score(y_test, test_pred, average="macro"),
            "Test Weighted F1": f1_score(y_test, test_pred, average="weighted"),
            "Test Balanced Accuracy": balanced_accuracy_score(y_test, test_pred),
        })

    comparison = pd.DataFrame(results).sort_values(
        by="Validation Macro F1",
        ascending=False,
    )
    comparison.to_csv(
        RESULTS_DIR / "model_comparison.csv",
        index=False,
        encoding="utf-8-sig",
    )

    random_forest = fitted_models["Random Forest"]
    joblib.dump(random_forest, MODEL_DIR / "random_forest.joblib")

    print("\n===== KET QUA TEST =====")
    print(comparison.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print("\nMo hinh chinh duoc chon: Random Forest")
    print("Da luu bang so sanh tai:", RESULTS_DIR / "model_comparison.csv")
    print("Da luu mo hinh Random Forest tai:", MODEL_DIR / "random_forest.joblib")


if __name__ == "__main__":
    main()