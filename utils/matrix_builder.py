import pandas as pd

def build_user_item_matrix(df):

    matrix = (
    df.pivot_table(
        index="customer_id",
        columns="Nama Produk",
        values="Jumlah Produk",
        aggfunc="sum",
        fill_value=0
    )
)

    return matrix