
from pathlib import Path
import pandas as pd

DATA_PATH = Path("data/raw/Dry_Bean_Dataset.xlsx")
RESULTS_DIR = Path("results")


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Khong tim thay file du lieu: {DATA_PATH}"
        )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_excel(DATA_PATH)

    # Kiem tra cac cot dac trung, khong tinh cot Class
    features = df.drop(columns=["Class"])

    # Tat ca cac dong co bo dac trung trung nhau
    duplicate_mask = features.duplicated(keep=False)
    duplicate_df = df.loc[duplicate_mask].copy()

    print("===== TONG QUAN DU LIEU =====")
    print("Tong so dong:", len(df))
    print("So dong trung hoan toan:",
          int(df.duplicated().sum()))
    print("So dong co dac trung trung nhau:",
          int(duplicate_mask.sum()))

    # Kiem tra cac bo dac trung giong nhau co nhieu nhan khong
    grouped = (
        df.groupby(list(features.columns), dropna=False)["Class"]
        .agg(["count", "nunique"])
        .reset_index()
    )

    conflicts = grouped[grouped["nunique"] > 1]

    print("\n===== KIEM TRA NHAN =====")
    print("So nhom dac trung co nhieu nhan:",
          len(conflicts))

    if not conflicts.empty:
        print(conflicts.head(10).to_string(index=False))
    else:
        print("Khong phat hien nhom dac trung co nhieu nhan.")

    print("\n===== PHAN BO NHAN TRONG CAC DONG TRUNG =====")
    print(duplicate_df["Class"].value_counts())

    output_path = RESULTS_DIR / "duplicate_rows.csv"
    duplicate_df.to_csv(
        output_path,
        index=False,
        encoding="utf-8-sig"
    )

    print("\nDa luu cac dong can xem xet tai:", output_path)


if __name__ == "__main__":
    main()
