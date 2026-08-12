import pandas as pd

from utils.data_cleaning import clean_data
from utils.matrix_builder import build_user_item_matrix
from utils.eda import (
    top_menu_chart,
    category_chart,
    monthly_transaction_chart,
    heatmap_chart
)

from models.popularity_model import PopularityRecommender
from models.ibcf_model import IBCFRecommender
from models.hybrid_model import HybridRecommender


# ==========================================================
# LOAD DATASET
# ==========================================================
df_raw = pd.read_csv(
    "data/Dataset_Transaksi_April_2025.csv",
    sep=";",
    encoding="utf-8-sig"
)

print("=" * 60)
print("DATASET")
print("=" * 60)
print(f"Shape data mentah : {df_raw.shape}")
print(f"Kolom             : {list(df_raw.columns)}")
print(df_raw.head())


# ==========================================================
# DATA CLEANING
# ==========================================================
df = clean_data(df)


# ==========================================================
# USER ITEM MATRIX
# ==========================================================
user_item_matrix = build_user_item_matrix(df)


# ==========================================================
# MENU INFORMATION
# ==========================================================
menu_info = df[
    [
        "Nama Produk",
        "Kategori"
    ]
].drop_duplicates()


# ==========================================================
# POPULARITY MODEL
# ==========================================================
pop_model = PopularityRecommender()
pop_model.fit(df)


# ==========================================================
# IBCF MODEL
# ==========================================================
ibcf_model = IBCFRecommender()
ibcf_model.fit(
    user_item_matrix,
    menu_info
)


# ==========================================================
# HYBRID MODEL
# ==========================================================
hybrid_model = HybridRecommender(
    pop_model,
    ibcf_model
)


# ==========================================================
# CUSTOMER SAMPLE
# ==========================================================

customer_name = df["Nama Pelanggan"].dropna().iloc[0]

print("=" * 60)
print("CUSTOMER")
print("=" * 60)
print(customer_name)


# ==========================================================
# POPULARITY RECOMMENDATION
# ==========================================================
print("=" * 60)
print("POPULARITY RECOMMENDATION")
print("=" * 60)

print(
    pop_model.recommend(
        top_n=5
    )
)


# ==========================================================
# IBCF RECOMMENDATION
# ==========================================================
print("=" * 60)
print("IBCF RECOMMENDATION")
print("=" * 60)

print(
    ibcf_model.recommend(
        customer_name=customer_name,
        top_n=5
    )
)


# ==========================================================
# HYBRID RECOMMENDATION
# ==========================================================
print("=" * 60)
print("HYBRID RECOMMENDATION")
print("=" * 60)

print(
    hybrid_model.recommend(
        customer_name=customer_name,
        top_n=5
    )
)


# ==========================================================
# EDA VISUALIZATION
# ==========================================================
top_menu_chart(df)
category_chart(df)
monthly_transaction_chart(df)
heatmap_chart(df)