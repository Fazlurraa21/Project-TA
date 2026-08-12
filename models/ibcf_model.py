
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity




# LANGKAH 1: MEMBANGUN USER-ITEM MATRIX
# (Sesuai Bagian 4.2 jurnal Mahendra dkk., 2024)Q


def build_user_item_matrix(df):
    temp = df.copy()
    temp["value"] = 1
    matrix = df.pivot_table(
        index="customer_id",
        columns="Nama Produk",
        values="Jumlah Produk",
        aggfunc="sum",
        fill_value=0
    )
    return matrix


# LANGKAH 2-4: ITEM-BASED COLLABORATIVE FILTERING

class IBCFRecommender:

    def __init__(self, k=10):
        self.k = k
        self.user_item_matrix = None
        self.item_similarity = None
        self.menu_info = None

    def fit(self, df):
        
        # Bangun customer_id yang benar (fungsi sudah diperbaiki)
        
        # df = build_customer_id(df, name_col="Nama Pelanggan")
               
        # Bangun User-Item Matrix
        self.user_item_matrix = build_user_item_matrix(df)

        
        # Cosine Similarity antar item 
        
        similarity = cosine_similarity(self.user_item_matrix.T)
        np.fill_diagonal(similarity, 0)

        self.item_similarity = pd.DataFrame(
            similarity,
            index=self.user_item_matrix.columns,
            columns=self.user_item_matrix.columns
        )
     
        # Info menu untuk ditampilkan bersama hasil rekomendasi
      
        self.menu_info = (
            df[["Nama Produk", "Kategori"]]
            .drop_duplicates()
            .set_index("Nama Produk")
        )

        # simpan juga df yang sudah ada customer_id-nya,
        # supaya bisa dipakai untuk lookup nama asli -> customer_id
        self.df_with_id = df

        return self

    def predict_score(self, customer_id, product_name, min_items=2):
        if customer_id not in self.user_item_matrix.index:
            return 0.0
        if product_name not in self.item_similarity.index:
            return 0.0

        user_rating = self.user_item_matrix.loc[customer_id]
        bought_items = user_rating[user_rating > 0]
        
        print("Customer :", customer_id)
        print("Histori pada matrix:")
        print(bought_items)

        # Jika pelanggan baru pernah membeli < min_items produk unik,
        # rumus weighted average akan collapse menjadi rating item
        # itu sendiri (numerator/denominator selalu 1), sehingga
        # similarity tidak lagi berarti apa-apa. Prediksi semacam
        # ini tidak reliable, jadi kita anggap belum ada prediksi.
        
        if len(bought_items) < min_items:
            return 0.0

        if product_name in bought_items.index:
            return 0.0

        # Kemiripan item target dengan item yang sudah dibeli (2.3.4)
        similarity = self.item_similarity.loc[product_name, bought_items.index]
        similarity = (similarity[similarity > 0].sort_values(ascending=False).head(self.k))
        # Prediksi rating (2.3.5)
        numerator = (similarity * bought_items[similarity.index]).sum()
        denominator = similarity.abs().sum()

        if denominator == 0:
            return 0.0

        prediction = numerator / denominator
        return round(float(prediction), 4)

    def recommend(self, customer_id, top_n=3, min_items=2):
        if customer_id not in self.user_item_matrix.index:
            return pd.DataFrame(columns=["Kategori", "Nama Produk", "predicted_score"])

        user_rating = self.user_item_matrix.loc[customer_id]
        already_bought = set(user_rating[user_rating > 0].index)

        # Langkah 2.3.3
        # Mengambil item yang mirip dengan histori user
        
        candidate_items = set()

        for bought_item in already_bought:

            similar_items = (
                self.item_similarity[bought_item]
                .sort_values(ascending=False)
                .head(self.k)
                .index
            )

            candidate_items.update(similar_items)

        # Hilangkan item yang sudah pernah dibeli
        candidate_items = candidate_items - already_bought
        
        
        # Histori terlalu tipis (< min_items produk unik) -> IBCF
        # tidak reliable untuk pelanggan ini, serahkan ke Popularity
        # lewat mekanisme fallback di HybridRecommender.
        if len(already_bought) < min_items:
            return pd.DataFrame(columns=["Kategori", "Nama Produk", "predicted_score"])

        scores = {}
        for item in candidate_items:
            if item in already_bought:
                continue
            score = self.predict_score(customer_id, item, min_items=min_items)
            if score > 0:
                scores[item] = score

        if not scores:
            return pd.DataFrame(columns=["Kategori", "Nama Produk", "predicted_score"])

        result = (
            pd.DataFrame(scores.items(), columns=["Nama Produk", "predicted_score"])
            .sort_values(by="predicted_score", ascending=False)
            .head(top_n)
        )

        result = result.merge(
            self.menu_info.reset_index(), on="Nama Produk", how="left"
        )

        return result[["Kategori", "Nama Produk", "predicted_score"]]

    def get_customer_id_by_name(self, nama_pelanggan):
        """
        Helper tambahan: cari customer_id dari nama pelanggan asli
        (menggunakan normalisasi yang sama seperti saat fit()).
        """
        target = normalize_name(nama_pelanggan)
        if target is None:
            return None

        matched = self.df_with_id[
            self.df_with_id["Nama Pelanggan"].apply(normalize_name) == target
        ]

        if matched.empty:
            return None

        return matched["customer_id"].iloc[0]


    def show_customers(self):
        """
        Menampilkan daftar customer_id beserta nama pelanggan.
        """
        return (
            self.df_with_id[
                ["customer_id", "Nama Pelanggan","Nama Produk"]
            ]
            .drop_duplicates()
            .sort_values("customer_id")
            .reset_index(drop=True)
        )