from pathlib import Path
import json
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

DATA = Path("data/processed")
OUT = Path("results/random_forest")


def main():
    train, validation, test = [pd.read_csv(DATA / name) for name in ("train_data.csv", "validation_data.csv", "test_data.csv")]
    target = "Class"
    x_train, y_train = train.drop(columns=target), train[target]
    x_validation, y_validation = validation.drop(columns=target), validation[target]
    x_test, y_test = test.drop(columns=target), test[target]
    model = RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1,
    )
    model.fit(x_train, y_train)
    train_pred, validation_pred = model.predict(x_train), model.predict(x_validation)
    combined = pd.concat([train, validation], ignore_index=True)
    model.fit(combined.drop(columns=target), combined[target])
    test_pred = model.predict(x_test)
    OUT.mkdir(parents=True, exist_ok=True)
    scores = {
        "train": {"accuracy": accuracy_score(y_train, train_pred), "macro_f1": f1_score(y_train, train_pred, average="macro")},
        "validation": {"accuracy": accuracy_score(y_validation, validation_pred), "macro_f1": f1_score(y_validation, validation_pred, average="macro")},
        "test": {"accuracy": accuracy_score(y_test, test_pred), "macro_f1": f1_score(y_test, test_pred, average="macro")},
    }
    payload = {"model_name": "Random Forest", "dataset": "Dry Bean Dataset", "train_samples": len(train), "validation_samples": len(validation), "test_samples": len(test), "features": x_train.shape[1], **scores}
    (OUT / "metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    (OUT / "classification_report.json").write_text(json.dumps(classification_report(y_test, test_pred, output_dict=True), indent=2), encoding="utf-8")
    pd.DataFrame(confusion_matrix(y_test, test_pred, labels=model.classes_), index=model.classes_, columns=model.classes_).to_csv(OUT / "confusion_matrix.csv", encoding="utf-8-sig")


if __name__ == "__main__":
    main()
