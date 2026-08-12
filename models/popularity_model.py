import pandas as pd


class PopularityRecommender:

    def __init__(self):
        self.popularity_df = None

    def fit(self, df):

        
        # Menghitung statistik setiap produk
        #
        # total_quantity : jumlah produk yang terjual
        # vote           : jumlah transaksi unik (ID Struk)
       

        menu_stats = (
            df.groupby(["Nama Produk", "Kategori"])
            .agg(
                total_quantity=("Jumlah Produk", "sum"),
                vote=("ID Struk", "nunique")
            )
            .reset_index()
        )

        
        # Membentuk rating sintetis
        #
        # Dataset tidak memiliki rating dari pengguna,
        # sehingga rating dibentuk menggunakan:
        #
        # Rating = Total Quantity / Vote

        menu_stats["rating"] = (
            menu_stats["total_quantity"] /
            menu_stats["vote"]
        )
        
                
        # Normalisasi rating sintetis ke skala 1-5
        #
        # Karena tidak tersedia rating asli dari pelanggan,
        # rating didekati dari rata-rata kuantitas pembelian
        # per transaksi, kemudian dinormalisasi ke skala 1-5
        # menggunakan Min-Max Scaling agar konsisten dengan
        # interpretasi rating pada umumnya.
       

        r_min = menu_stats["rating"].min()
        r_max = menu_stats["rating"].max()

        menu_stats["rating"] = (
            1 + 4 * (menu_stats["rating"] - r_min) / (r_max - r_min)
        )

        
        # RUMUS (1)
        # Average Rating
        # Menghitung rata-rata rating seluruh produk
       

        C = menu_stats["rating"].mean()

        
        # Menentukan nilai minimum vote (m)
        #
        # m dihitung menggunakan persentil ke-75
        # dari seluruh jumlah vote produk
       

        m = menu_stats["vote"].quantile(0.75)

      
        # Menghitung bobot produk
        #
        # weight_product = v / (v + m)
       

        weight_product = (
            menu_stats["vote"] /
            (menu_stats["vote"] + m)
        )

        # ======================================================
        # Menghitung bobot global
        #
        # weight_global = m / (v + m)
        # ======================================================

        weight_global = (
            m /
            (menu_stats["vote"] + m)
        )


        # RUMUS (2)
        # Weighted Rating (WR)
       
        menu_stats["weighted_rating"] = (++++++++++++++++
            weight_product * menu_stats["rating"] +
            weight_global * C
        )

        # ======================================================
        # Mengurutkan produk berdasarkan nilai WR
        # dari yang terbesar ke yang terkecil
        # ======================================================

        self.popularity_df = (
            menu_stats
            .sort_values(
                by="weighted_rating",
                ascending=False
            )
            .reset_index(drop=True)
        )

        return self

    def recommend(self, top_n=3):

        return self.popularity_df.head(top_n)[
            [
                "Nama Produk",
                "Kategori",
                "weighted_rating"
            ]
        ]