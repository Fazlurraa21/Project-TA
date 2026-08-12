from flask import Flask, jsonify
import pandas as pd

# IMPORT MODELS & UTILS
from models.popularity_model import PopularityRecommender
from models.ibcf_model import IBCFRecommender
from models.hybrid_model import HybridRecommender

from utils.matrix_builder import build_user_item_matrix
from utils.data_cleaning import clean_data

# INIT FLASK
app = Flask(__name__)

# ==========================================================
# LOAD DATASET
# ==========================================================
df_raw = pd.read_csv(
    "data/Dataset_Transaksi_April_2025.csv",
    sep=";",
    encoding="utf-8-sig"
)

print(f"Shape data mentah : {df_raw.shape}")
print(df_raw.columns.tolist())

# ==========================================================
# CLEAN DATA
# ==========================================================
df = clean_data(df_raw)

print(df.columns)

print(df["customer_id"].head())

print("Jumlah customer:", df["customer_id"].nunique())

print("Apakah C1906 ada?", "C1906" in df["customer_id"].astype(str).values)

# TRAIN MODEL,
# customer_id TIDAK dibuat manual di sini lagi. Logika
# normalisasi nama & pembuatan customer_id sekarang sepenuhnya
# ada di dalam IBCFRecommender.fit() (lihat models/ibcf_model.py).
#
# Setelah fit() dijalankan, kita ambil kembali df yang SUDAH
# punya kolom customer_id yang benar lewat ibcf_model.df_with_id,
# lalu pakai df itu untuk semua kebutuhan lain di app.py
# (pop_model, route /top-menu, dst) supaya seluruh bagian
# aplikasi konsisten memakai customer_id yang sama.

ibcf_model = IBCFRecommender()
ibcf_model.fit(df)

print("Jumlah customer di matrix:", len(ibcf_model.user_item_matrix.index))

print("Apakah C1906 ada di matrix?",
      "C1906" in ibcf_model.user_item_matrix.index)

print(ibcf_model.user_item_matrix.index[:20])
print(ibcf_model.show_customers())

df = ibcf_model.df_with_id          # <-- df global sekarang sudah
                                     #     punya customer_id yang benar

user_item_matrix = ibcf_model.user_item_matrix
menu_info = ibcf_model.menu_info.reset_index()

pop_model = PopularityRecommender()
pop_model.fit(df)

hybrid_model = HybridRecommender(
    pop_model,
    ibcf_model
)

# HOME

@app.route("/")
def home():

    return jsonify({
        "message": "Recommendation API Running"
    })



# TOP MENU

@app.route("/top-menu")
def top_menu():

    top_menu = (
        df.groupby("Nama Produk")["Jumlah Produk"]
        .sum()
        .sort_values(ascending=False)
        .head(10)
    )

    return jsonify({

        "menu": top_menu.index.tolist(),

        "quantity": top_menu.values.tolist()

    })


# CATEGORY

@app.route("/category")
def category():

    category_data = (
        df.groupby("Kategori")["Jumlah Produk"]
        .sum()
    )

    return jsonify({

        "category": category_data.index.tolist(),

        "quantity": category_data.values.tolist()

    })



# MONTHLY TRANSACTION

@app.route("/monthly-transaction")
def monthly_transaction():

    temp_df = df.copy()

    temp_df["Tanggal & Waktu"] = pd.to_datetime(
        temp_df["Tanggal & Waktu"],
        errors="coerce"
    )

    temp_df = temp_df.dropna(
        subset=["Tanggal & Waktu"]
    )

    temp_df["month"] = (
        temp_df["Tanggal & Waktu"]
        .dt.to_period("M")
        .astype(str)
    )

    monthly = (
        temp_df.groupby("month")["ID Struk"]
        .nunique()
    )

    return jsonify({

        "month": monthly.index.tolist(),

        "transaction": monthly.values.tolist()

    })


# POPULARITY

@app.route("/popularity")
def popularity():

    result = pop_model.recommend(
        top_n=3
    )

    return jsonify(
        result.to_dict(
            orient="records"
        )
    )



# IBCF

@app.route("/ibcf/<customer_id>")
def ibcf(customer_id):

    result = ibcf_model.recommend(
         customer_id=customer_id,
         top_n=3

    )

    if result.empty:

        return jsonify({
            "message": "Customer tidak ditemukan"
        })

    return jsonify(
        result.to_dict(
            orient="records"
        )
    )
    
   
   
# HISTORY CUSTOMER

@app.route("/history/<customer_id>")
def history(customer_id):

    history = (
        df[df["customer_id"] == customer_id][
            ["Nama Produk", "Kategori"]
        ]
        .drop_duplicates()
        .reset_index(drop=True)
    )

    if history.empty:
        return jsonify({
            "message": "Customer tidak ditemukan"
        })

    return jsonify(
        history.to_dict(orient="records")
    )



# HYBRID

@app.route("/hybrid/<customer_id>")
def hybrid(customer_id):

    result = hybrid_model.recommend(

        customer_id=customer_id,

        top_n=3

    )

    if result.empty:

        return jsonify({
            "message": "Customer tidak ditemukan"
        })

    return jsonify(
        result.to_dict(
            orient="records"
        )
    )


# ==========================================================
# CUSTOMER LOOKUP (BARU - opsional, untuk memudahkan testing)
# ==========================================================
@app.route("/customer-lookup/<nama>")
def customer_lookup(nama):
    """
    Endpoint bantu untuk mencari customer_id dari nama pelanggan
    asli, supaya tidak perlu tebak-tebak ID mana yang mau dites
    di /ibcf/<customer_id> atau /hybrid/<customer_id>.
    """
    cid = ibcf_model.get_customer_id_by_name(nama)

    if cid is None:
        return jsonify({"message": "Nama pelanggan tidak ditemukan"})

    return jsonify({"nama": nama, "customer_id": cid})


# ==========================================================
# RUN SERVER
# ==========================================================
if __name__ == "__main__":
    app.run(debug=True)
