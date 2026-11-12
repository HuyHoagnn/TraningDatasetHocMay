from pathlib import Path
import json

import joblib
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


DATA = Path("data/processed")
RESULTS = Path("results/dimensionality_reduction")
MODELS = Path("models/dimensionality_reduction")
TARGET = "Class"


def evaluate(model, x_train, y_train, x_validation, y_validation, x_test, y_test):
    model.fit(x_train, y_train)
    validation_pred = model.predict(x_validation)

    combined_x = pd.concat([x_train, x_validation], ignore_index=True)
    combined_y = pd.concat([y_train, y_validation], ignore_index=True)
    model.fit(combined_x, combined_y)
    test_pred = model.predict(x_test)

    return {
        "validation": {
            "accuracy": accuracy_score(y_validation, validation_pred),
            "macro_f1": f1_score(y_validation, validation_pred, average="macro"),
        },
        "test": {
            "accuracy": accuracy_score(y_test, test_pred),
            "macro_f1": f1_score(y_test, test_pred, average="macro"),
        },
    }


def main():
    train, validation, test = [
        pd.read_csv(DATA / name)
        for name in ("train_data.csv", "validation_data.csv", "test_data.csv")
    ]
    x_train, y_train = train.drop(columns=TARGET), train[TARGET]
    x_validation, y_validation = validation.drop(columns=TARGET), validation[TARGET]
    x_test, y_test = test.drop(columns=TARGET), test[TARGET]

    models = {
        "PCA": Pipeline([
            ("scaler", StandardScaler()),
            ("reducer", PCA(n_components=0.95, random_state=42)),
            ("classifier", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42)),
        ]),
        "LDA": Pipeline([
            ("scaler", StandardScaler()),
            ("reducer", LinearDiscriminantAnalysis(n_components=min(6, x_train.shape[1]))),
            ("classifier", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42)),
        ]),
    }

    RESULTS.mkdir(parents=True, exist_ok=True)
    MODELS.mkdir(parents=True, exist_ok=True)
    output = {
        "dataset": "Dry Bean Dataset",
        "classifier": "Logistic Regression",
        "features_before_reduction": x_train.shape[1],
        "train_samples": len(train),
        "validation_samples": len(validation),
        "test_samples": len(test),
        "methods": {},
    }

    for name, model in models.items():
        scores = evaluate(
            model,
            x_train,
            y_train,
            x_validation,
            y_validation,
            x_test,
            y_test,
        )
        reducer = model.named_steps["reducer"]
        output["methods"][name] = {
            "features_after_reduction": int(
                getattr(reducer, "n_components_", reducer.n_components)
            ),
            **scores,
        }
        joblib.dump(model, MODELS / f"{name.lower()}.joblib")

    (RESULTS / "metrics.json").write_text(
        json.dumps(output, indent=2), encoding="utf-8"
    )
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()