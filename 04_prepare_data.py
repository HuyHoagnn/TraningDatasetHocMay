
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

DATA_PATH = Path("data/raw/Dry_Bean_Dataset.xlsx")
PROCESSED_DIR = Path("data/processed")

TARGET = "Class"


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Khong tim thay file du lieu: {DATA_PATH}"
        )

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_excel(DATA_PATH)

    if TARGET not in df.columns:
        raise ValueError(f"Khong tim thay cot nhan {TARGET}")

    if df.isnull().values.any():
        raise ValueError("Du lieu co gia tri thieu.")

    # Loai bo cac dong trung hoan toan
    original_rows = len(df)
    df = df.drop_duplicates().copy()
    removed_rows = original_rows - len(df)

    # Tach dac trung va nhan
    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    print("===== TONG QUAN DU LIEU =====")
    print("So dong ban dau:", original_rows)
    print("So dong trung da loai:", removed_rows)
    print("So dong sau xu ly:", len(df))
    print("So dac trung:", X.shape[1])
    print("So lop:", y.nunique())
    print("Danh sach lop:", sorted(y.unique()))

    # Chia du lieu 70% train, 15% validation, 15% test
    X_train, X_remaining, y_train, y_remaining = train_test_split(
        X,
        y,
        test_size=0.3,
        random_state=42,
        stratify=y
    )
    X_validation, X_test, y_validation, y_test = train_test_split(
        X_remaining,
        y_remaining,
        test_size=0.5,
        random_state=42,
        stratify=y_remaining
    )

    print("\n===== KICH THUOC CAC TAP =====")
    print("X_train:", X_train.shape)
    print("X_validation:", X_validation.shape)
    print("X_test:", X_test.shape)
    print("y_train:", y_train.shape)
    print("y_validation:", y_validation.shape)
    print("y_test:", y_test.shape)

    # Luu du lieu sau xu ly
    df.to_csv(
        PROCESSED_DIR / "dry_bean_clean.csv",
        index=False,
        encoding="utf-8-sig"
    )

    # Luu tap huan luyen
    train_df = X_train.copy()
    train_df[TARGET] = y_train

    train_df.to_csv(
        PROCESSED_DIR / "train_data.csv",
        index=False,
        encoding="utf-8-sig"
    )

    validation_df = X_validation.copy()
    validation_df[TARGET] = y_validation

    validation_df.to_csv(
        PROCESSED_DIR / "validation_data.csv",
        index=False,
        encoding="utf-8-sig"
    )

    # Luu tap kiem tra
    test_df = X_test.copy()
    test_df[TARGET] = y_test

    test_df.to_csv(
        PROCESSED_DIR / "test_data.csv",
        index=False,
        encoding="utf-8-sig"
    )

    print("\nDa luu 4 file tai:", PROCESSED_DIR)


if __name__ == "__main__":
    main()
