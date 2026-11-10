from pathlib import Path
import json

RESULTS = Path("results")
MODEL_NAMES = ("random_forest", "svm_rbf", "mlp", "logistic_regression")


def main():
    output = [json.loads((RESULTS / name / "metrics.json").read_text(encoding="utf-8")) for name in MODEL_NAMES]
    (RESULTS / "all_models_metrics.json").write_text(json.dumps(output, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
