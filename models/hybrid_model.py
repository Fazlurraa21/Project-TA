import pandas as pd


class HybridRecommender:

    def __init__(self, pop_model, ibcf_model):
        self.pop_model = pop_model
        self.ibcf_model = ibcf_model

    def recommend(self, customer_id, top_n=3):

        # Rekomendasi menggunakan IBCF
        ibcf_result = self.ibcf_model.recommend(
            customer_id=customer_id,
            top_n=top_n
        )

        # Jika customer belum memiliki histori pembelian,
        # gunakan Popularity Recommendation
        if ibcf_result is None or ibcf_result.empty:

            pop = self.pop_model.recommend(top_n).copy()
            pop["predicted_score"] = None
            pop["source"] = "Popularity"

            return pop

        ibcf_result["source"] = "IBCF"

        # Ambil skor popularity (Weighted Rating)
        pop_score = self.pop_model.recommend(1000)[
            ["Nama Produk", "weighted_rating"]
        ]

        # Gabungkan hasil IBCF dengan Popularity
        ibcf_result = ibcf_result.merge(
            pop_score,
            on="Nama Produk",
            how="left"
        )

        return ibcf_result