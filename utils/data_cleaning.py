import pandas as pd
import numpy as np


def parse_date(s):
    for fmt in [
        '%Y-%m-%d',
        '%d/%m/%Y',
        '%m/%d/%Y',
        '%d-%m-%Y',
        '%d/%m/%Y %H:%M',
        '%Y-%m-%d %H:%M:%S'
    ]:
        try:
            return pd.to_datetime(s, format=fmt)
        except:
            pass

    return pd.to_datetime(s, errors='coerce')


def clean_data(df):

    df = df.copy()

    # Hapus data duplikat
    df = df.drop_duplicates()

    # Ganti string kosong menjadi NaN
    df.replace(r'^\s*$', np.nan, regex=True, inplace=True)

    # Hapus transaksi tanpa Nama Pelanggan
    df = df.dropna(subset=['Nama Pelanggan'])

    # Imputasi Jumlah Produk dengan median setiap Nama Produk
    median_qty = (
        df.groupby('Nama Produk')['Jumlah Produk']
        .transform('median')
    )

    df['Jumlah Produk'] = (
        df['Jumlah Produk']
        .fillna(median_qty)
    )

    df['Jumlah Produk'] = pd.to_numeric(
        df['Jumlah Produk'],
        errors='coerce'
    ).fillna(1).astype(int)

    # Konversi tanggal transaksi
    df['Tanggal & Waktu'] = (
        df['Tanggal & Waktu']
        .apply(parse_date)
    )

    df = df.dropna(
        subset=['Tanggal & Waktu']
    )
    

    # Pastikan nilai Total sesuai
    mask = (
        df['Total'] !=
        (df['Harga Produk'] * df['Jumlah Produk'])
    )

    df.loc[mask, 'Total'] = (
        df.loc[mask, 'Harga Produk'] *
        df.loc[mask, 'Jumlah Produk']
    )

    df = df.reset_index(drop=True)

    print(f"Shape setelah cleaning : {df.shape}")
    print(df.isnull().sum())

    return df