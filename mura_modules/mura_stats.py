import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import os

def main(csv_path, save_output=False):
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"❌ CSV file not found: {csv_path}")

    # Load CSV
    df = pd.read_csv(csv_path, names=["study", "label"], skiprows=1)
    df["label"] = df["label"].astype(int)
    df["body_part"] = df["study"].apply(lambda x: x.split("/")[2].upper())

    # Print overall stats
    print("📊 Overall class distribution (0 = normal, 1 = abnormal):")
    print(df["label"].value_counts().sort_index())

    print("\n🦴 Samples per body part:")
    print(df["body_part"].value_counts())

    print("\n🧮 Class distribution per body part:")
    part_label = pd.crosstab(df["body_part"], df["label"], rownames=["Body Part"], colnames=["Label"])
    print(part_label)

    # Plot
    plt.figure(figsize=(10, 6))
    sns.heatmap(part_label, annot=True, fmt="d", cmap="Blues")
    plt.title("Normal vs Abnormal Samples per Body Part")
    plt.xlabel("Label (0 = Normal, 1 = Abnormal)")
    plt.ylabel("Body Part")
    plt.tight_layout()
    plt.show()

    # Optional: Save to CSV
    if save_output:
        output_file = "mura_class_distribution.csv"
        part_label.to_csv(output_file)
        print(f"\n💾 Saved distribution table to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze MURA dataset class imbalance.")
    parser.add_argument("csv_path", help="Path to MURA_train_labeled.csv")
    parser.add_argument("--save", action="store_true", help="Save the output table to CSV")

    args = parser.parse_args()
    main(args.csv_path, save_output=args.save)
