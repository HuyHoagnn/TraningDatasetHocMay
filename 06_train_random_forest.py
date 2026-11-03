
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score
)

PROCESSED_DIR = Path("data/processed")
RESULTS_DIR = Path("results")
MODEL_DIR = Path("models")

TARGET = "Class"


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

    print("===== DU LIEU HUAN LUYEN =====")
    print("So mau train:", len(X_train))
    print("So mau validation:", len(X_validation))
    print("So mau test:", len(X_test))
    print("So dac trung:", X_train.shape[1])

    # Khoi tao mo hinh Random Forest
    model = RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1
    )

    # Huan luyen mo hinh
    print("\nDang huan luyen mo hinh...")
    model.fit(X_train, y_train)

    validation_pred = model.predict(X_validation)

    combined_df = pd.concat([train_df, validation_df], ignore_index=True)
    X_combined = combined_df.drop(columns=[TARGET])
    y_combined = combined_df[TARGET]
    model.fit(X_combined, y_combined)
    test_pred = model.predict(X_test)

    # Danh gia mo hinh
    validation_accuracy = accuracy_score(y_validation, validation_pred)
    validation_macro_f1 = f1_score(y_validation, validation_pred, average="macro")
    test_accuracy = accuracy_score(y_test, test_pred)
    test_macro_f1 = f1_score(y_test, test_pred, average="macro")

    print("\n===== KET QUA DANH GIA =====")
    print(f"Validation Accuracy: {validation_accuracy:.4f}")
    print(f"Validation Macro F1-score: {validation_macro_f1:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f} ({test_accuracy * 100:.2f}%)")
    print(f"Test Macro F1-score: {test_macro_f1:.4f}")

    report = classification_report(
        y_test,
        test_pred,
        zero_division=0
    )

    print("\n===== CLASSIFICATION REPORT =====")
    print(report)

    # Luu bao cao danh gia
    report_path = RESULTS_DIR / "classification_report.txt"

    with open(report_path, "w", encoding="utf-8") as file:
        file.write(f"Validation Accuracy: {validation_accuracy:.4f}\n")
        file.write(f"Validation Macro F1-score: {validation_macro_f1:.4f}\n")
        file.write(f"Test Accuracy: {test_accuracy:.4f}\n")
        file.write(f"Test Macro F1-score: {test_macro_f1:.4f}\n\n")
        file.write(report)

    # Ve ma tran nham lan
    labels = sorted(y_test.unique())
    cm = confusion_matrix(
        y_test,
        test_pred,
        labels=labels
    )

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels
    )

    plt.title("Confusion Matrix - Random Forest")
    plt.xlabel("Nhan du doan")
    plt.ylabel("Nhan thuc te")
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(
        RESULTS_DIR / "03_confusion_matrix.png",
        dpi=300
    )
    plt.close()

    # Luu muc do quan trong cua cac dac trung
    importance_df = pd.DataFrame({
        "Feature": X_combined.columns,
        "Importance": model.feature_importances_
    }).sort_values(
        by="Importance",
        ascending=False
    )

    importance_df.to_csv(
        RESULTS_DIR / "feature_importance.csv",
        index=False,
        encoding="utf-8-sig"
    )

    # Ve bieu do top 10 dac trung quan trong
    top_features = importance_df.head(10).sort_values(
        by="Importance",
        ascending=True
    )

    plt.figure(figsize=(10, 6))
    plt.barh(
        top_features["Feature"],
        top_features["Importance"]
    )
    plt.title("Top 10 Feature Importances - Random Forest")
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.savefig(
        RESULTS_DIR / "04_feature_importance.png",
        dpi=300
    )
    plt.close()

    # Luu mo hinh da huan luyen
    model_path = MODEL_DIR / "random_forest.joblib"
    joblib.dump(model, model_path)

    print("Da luu bao cao tai:", report_path)
    print("Da luu ma tran nham lan tai:",
          RESULTS_DIR / "03_confusion_matrix.png")
    print("Da luu bieu do dac trung tai:",
          RESULTS_DIR / "04_feature_importance.png")
    print("Da luu mo hinh tai:", model_path)


if __name__ == "__main__":
    main()
