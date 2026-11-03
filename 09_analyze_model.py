from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)

TEST_PATH = Path("data/processed/test_data.csv")
MODEL_PATH = Path("models/random_forest.joblib")
RESULTS_DIR = Path("results")
TARGET = "Class"


def main():
    if not TEST_PATH.exists():
        raise FileNotFoundError(f"Khong tim thay tap Test: {TEST_PATH}")
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Khong tim thay mo hinh: {MODEL_PATH}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    test_data = pd.read_csv(TEST_PATH)
    model = joblib.load(MODEL_PATH)
    X_test = test_data.drop(columns=[TARGET])
    y_test = test_data[TARGET]

    if hasattr(model, "feature_names_in_"):
        X_test = X_test[list(model.feature_names_in_)]

    y_pred = model.predict(X_test)
    labels = list(model.classes_)

    cm = confusion_matrix(y_test, y_pred, labels=labels)
    cm_df = pd.DataFrame(cm, index=labels, columns=labels)
    cm_df.index.name = "Actual"
    cm_df.columns.name = "Predicted"
    cm_df.to_csv(RESULTS_DIR / "confusion_matrix_values.csv", encoding="utf-8-sig")

    precision, recall, f1, support = precision_recall_fscore_support(
        y_test,
        y_pred,
        labels=labels,
        zero_division=0,
    )
    class_metrics = pd.DataFrame({
        "Class": labels,
        "Correct": cm.diagonal(),
        "Incorrect": cm.sum(axis=1) - cm.diagonal(),
        "Precision": precision,
        "Recall": recall,
        "F1-score": f1,
        "Support": support,
    })
    class_metrics.to_csv(
        RESULTS_DIR / "class_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )

    importance_df = pd.DataFrame({
        "Feature": X_test.columns,
        "Importance": model.feature_importances_,
    }).sort_values("Importance", ascending=False)
    importance_df.to_csv(
        RESULTS_DIR / "feature_importance_analysis.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print("===== MA TRAN NHAM LAN =====")
    print("Hang: Nhan thuc te")
    print("Cot: Nhan du doan")
    print(cm_df.to_string())

    print("\n===== SO LIEU THEO LOP =====")
    print(class_metrics.to_string(index=False, float_format=lambda value: f"{value:.4f}"))

    print("\n===== TOP 5 DAC TRUNG QUAN TRONG NHAT =====")
    print(importance_df.head(5).to_string(index=False, float_format=lambda value: f"{value:.6f}"))

    print("\n===== 5 DAC TRUNG IT QUAN TRONG NHAT =====")
    print(importance_df.tail(5).sort_values("Importance").to_string(
        index=False,
        float_format=lambda value: f"{value:.6f}",
    ))

    print("\n===== TAP TIN DA LUU =====")
    print(RESULTS_DIR / "confusion_matrix_values.csv")
    print(RESULTS_DIR / "class_metrics.csv")
    print(RESULTS_DIR / "feature_importance_analysis.csv")


if __name__ == "__main__":
    main()