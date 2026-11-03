
from pathlib import Path

import pandas as pd

DATA_DIR = Path("data/raw")


def load_dataset():
    file_path = DATA_DIR / "Dry_Bean_Dataset.xlsx"
    if not file_path.exists():
        raise FileNotFoundError(
            f"Khong tim thay file du lieu: {file_path}"
        )

    print(f"File duoc su dung: {file_path}")
    return pd.read_excel(file_path)


def main():
    df = load_dataset()

    print("\n===== KICH THUOC DU LIEU =====")
    print(f"So dong: {df.shape[0]}")
    print(f"So cot: {df.shape[1]}")

    print("\n===== TEN CAC COT =====")
    print(df.columns.tolist())

    print("\n===== NAM DONG DAU TIEN =====")
    print(df.head())

    print("\n===== THONG TIN DU LIEU =====")
    df.info()

    print("\n===== GIA TRI THIEU =====")
    print(df.isnull().sum())

    print("\n===== SO DONG TRUNG LAP =====")
    print(df.duplicated().sum())

    print("\n===== PHAN BO GIONG DAU =====")
    if "Class" in df.columns:
        print(df["Class"].value_counts())
    else:
        print("Khong tim thay cot Class.")
        print("Hay kiem tra lai ten cot nhan.")


if __name__ == "__main__":
    main()
