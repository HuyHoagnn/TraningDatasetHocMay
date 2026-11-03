
from pathlib import Path

import joblib
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
TEST_PATH = BASE_DIR / "data" / "processed" / "test_data.csv"
MODEL_PATH = BASE_DIR / "models" / "random_forest.joblib"


def main():
    if not TEST_PATH.exists():
        raise FileNotFoundError(
            "Khong tim thay test_data.csv. "
            "Hay chay 04_prepare_data.py truoc."
        )

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "Khong tim thay random_forest.joblib. "
            "Hay huan luyen mo hinh truoc."
        )

    test_df = pd.read_csv(TEST_PATH)
    model = joblib.load(MODEL_PATH)

    # Lay mot mau tu tap Test
    sample = test_df.sample(n=1, random_state=42)

    X_sample = sample.drop(columns=["Class"])
    y_true = sample["Class"].iloc[0]

    # Dam bao dung thu tu dac trung khi du doan
    if hasattr(model, "feature_names_in_"):
        X_sample = X_sample[list(model.feature_names_in_)]

    y_pred = model.predict(X_sample)[0]

    print("===== KET QUA DU DOAN MAU TEST =====")
    print("So dac trung:", X_sample.shape[1])
    print("Nhan thuc te :", y_true)
    print("Nhan du doan :", y_pred)
    print("Mo hinh su dung:", MODEL_PATH.name)

    if y_pred == y_true:
        print("Ket qua      : DU DOAN DUNG")
    else:
        print("Ket qua      : DU DOAN SAI")

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(X_sample)[0]
        classes = model.classes_
        best_index = probabilities.argmax()

        print(
            "Do tin cay tham khao: {:.2f}%".format(
                probabilities[best_index] * 100
            )
        )
        print("Nhan co xac suat cao nhat:", classes[best_index])


if __name__ == "__main__":
    main()
