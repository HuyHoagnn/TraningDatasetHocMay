
from pathlib import Path
import numpy as np
import pandas as pd

DATA_PATH = Path("data/raw/Dry_Bean_Dataset.xlsx")
TARGET = "Class"


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Khong tim thay file: {DATA_PATH}"
        )

    df = pd.read_excel(DATA_PATH)

    print("===== 1. KICH THUOC DU LIEU =====")
    print("So dong:", df.shape[0])
    print("So cot:", df.shape[1])

    print("\n===== 2. KIEM TRA COT =====")
    print("Danh sach cot:")
    print(df.columns.tolist())

    if TARGET not in df.columns:
        raise ValueError("Khong tim thay cot Class.")

    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    print("\n===== 3. KIEM TRA GIA TRI THIEU =====")
    print(df.isnull().sum())

    print("\nTong gia tri thieu:", df.isnull().sum().sum())

    print("\n===== 4. KIEM TRA GIA TRI VO CUC =====")
    numeric_X = X.select_dtypes(include="number")
    infinite_count = np.isinf(numeric_X.to_numpy()).sum()
    print("So gia tri vo cuc:", int(infinite_count))

    print("\n===== 5. KIEM TRA KIEU DU LIEU =====")
    print(X.dtypes)

    non_numeric = X.select_dtypes(exclude="number").columns.tolist()
    print("Cot dac trung khong phai so:", non_numeric)

    print("\n===== 6. PHAN BO GIONG DAU =====")
    print(y.value_counts())

    print("\nSo lop:", y.nunique())

    if df.isnull().sum().sum() == 0 and infinite_count == 0:
        print("\nKet luan: Khong phat hien gia tri thieu hoac vo cuc.")

    if len(non_numeric) == 0:
        print("Ket luan: Tat ca dac trung deu la du lieu so.")


if __name__ == "__main__":
    main()
