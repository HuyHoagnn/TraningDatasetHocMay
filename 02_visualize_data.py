
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

DATA_DIR = Path("data/raw")
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid")


def load_dataset():
    file_path = DATA_DIR / "Dry_Bean_Dataset.xlsx"
    if not file_path.exists():
        raise FileNotFoundError(
            f"Khong tim thay file du lieu: {file_path}"
        )

    print(f"File duoc su dung: {file_path}")
    return pd.read_excel(file_path)


def plot_class_distribution(df):
    counts = df["Class"].value_counts()

    plt.figure(figsize=(10, 6))
    ax = sns.barplot(
        x=counts.index,
        y=counts.values
    )

    ax.set_title("Phan bo so luong mau theo giong dau")
    ax.set_xlabel("Giong dau")
    ax.set_ylabel("So luong mau")
    plt.xticks(rotation=30)

    for container in ax.containers:
        ax.bar_label(container)

    plt.tight_layout()
    plt.savefig(
        RESULTS_DIR / "01_class_distribution.png",
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()


def plot_correlation(df):
    features = df.drop(columns=["Class"])
    correlation = features.corr()

    plt.figure(figsize=(14, 11))
    sns.heatmap(
        correlation,
        cmap="coolwarm",
        center=0,
        vmin=-1,
        vmax=1
    )

    plt.title("Ma tran tuong quan giua cac dac trung")
    plt.tight_layout()
    plt.savefig(
        RESULTS_DIR / "02_correlation_matrix.png",
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()


def plot_feature_boxplots(df):
    features = [
        "Area",
        "Perimeter",
        "MajorAxisLength",
        "MinorAxisLength",
        "Compactness"
    ]

    for feature in features:
        plt.figure(figsize=(10, 6))

        sns.boxplot(
            data=df,
            x="Class",
            y=feature
        )

        plt.title(f"Phan bo {feature} theo giong dau")
        plt.xlabel("Giong dau")
        plt.ylabel(feature)
        plt.xticks(rotation=30)
        plt.tight_layout()

        filename = f"boxplot_{feature}.png"
        plt.savefig(
            RESULTS_DIR / filename,
            dpi=300,
            bbox_inches="tight"
        )
        plt.close()


def main():
    df = load_dataset()

    print("Kich thuoc du lieu:", df.shape)
    print("\nThong ke mo ta:")
    print(df.describe().round(3))

    plot_class_distribution(df)
    plot_correlation(df)
    plot_feature_boxplots(df)

    print("\nDa luu cac bieu do vao thu muc results.")


if __name__ == "__main__":
    main()
