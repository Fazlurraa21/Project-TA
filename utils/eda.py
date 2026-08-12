import matplotlib
matplotlib.use('Agg')  # Backend non-GUI

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os

# Membuat folder output jika belum ada
os.makedirs("output_charts", exist_ok=True)


# ==========================================================
# TOP 10 MENU TERLARIS
# ==========================================================
def top_menu_chart(df):

    top_menu = (
        df.groupby("Nama Produk")["Jumlah Produk"]
        .sum()
        .sort_values(ascending=False)
        .head(10)
    )

    plt.figure(figsize=(10,6))

    plt.bar(
        top_menu.index,
        top_menu.values
    )

    plt.xlabel("Nama Produk")
    plt.ylabel("Jumlah Produk Terjual")
    plt.title("Top 10 Menu Terlaris")

    plt.xticks(rotation=45, ha="right")

    plt.tight_layout()

    plt.savefig(
        "output_charts/top_menu_chart.png"
    )

    plt.close()

    print("Saved : output_charts/top_menu_chart.png")


# ==========================================================
# DISTRIBUSI KATEGORI
# ==========================================================
def category_chart(df):

    plt.figure(figsize=(8,6))

    sns.countplot(
        data=df,
        x="Kategori"
    )

    plt.title("Distribusi Kategori")

    plt.xticks(rotation=45, ha="right")

    plt.tight_layout()

    plt.savefig(
        "output_charts/category_chart.png"
    )

    plt.close()

    print("Saved : output_charts/category_chart.png")


# ==========================================================
# TREN TRANSAKSI BULANAN
# ==========================================================
def monthly_transaction_chart(df):

    temp_df = df.copy()

    temp_df["Tanggal & Waktu"] = pd.to_datetime(
        temp_df["Tanggal & Waktu"],
        errors="coerce"
    )

    temp_df = temp_df.dropna(
        subset=["Tanggal & Waktu"]
    )

    temp_df["Bulan"] = (
        temp_df["Tanggal & Waktu"]
        .dt.to_period("M")
    )

    monthly = (
        temp_df.groupby("Bulan")["ID Struk"]
        .nunique()
    )

    plt.figure(figsize=(12,5))

    plt.plot(
        monthly.index.astype(str),
        monthly.values,
        marker="o"
    )

    plt.title("Tren Transaksi Per Bulan")

    plt.xlabel("Bulan")
    plt.ylabel("Jumlah Transaksi")

    plt.xticks(rotation=45)

    plt.grid(True)

    plt.tight_layout()

    plt.savefig(
        "output_charts/monthly_transaction_chart.png"
    )

    plt.close()

    print("Saved : output_charts/monthly_transaction_chart.png")


# ==========================================================
# HEATMAP KORELASI
# ==========================================================
def heatmap_chart(df):

    numeric_df = df.select_dtypes(
        include="number"
    )

    plt.figure(figsize=(10,8))

    sns.heatmap(
        numeric_df.corr(),
        annot=True,
        fmt=".2f",
        cmap="Blues"
    )

    plt.title("Heatmap Korelasi")

    plt.tight_layout()

    plt.savefig(
        "output_charts/heatmap_chart.png"
    )

    plt.close()

    print("Saved : output_charts/heatmap_chart.png")