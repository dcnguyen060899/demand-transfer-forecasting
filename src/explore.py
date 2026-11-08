"""Load UCI Online Retail II, save a clean parquet, and summarize data per country.

Run from the project root:  .venv/bin/python -I src/explore.py
"""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "online_retail_II.xlsx"
CLEAN = ROOT / "data" / "processed" / "transactions.parquet"


def load():
    if CLEAN.exists():
        return pd.read_parquet(CLEAN)
    sheets = pd.read_excel(RAW, sheet_name=None, dtype={"Invoice": str, "StockCode": str})
    df = pd.concat(sheets.values(), ignore_index=True)
    df = df.rename(columns={"Customer ID": "CustomerID"})
    # Keep real sales: drop cancellations (invoice starts with C), returns and zero-price lines.
    df = df[~df["Invoice"].str.startswith("C") & (df["Quantity"] > 0) & (df["Price"] > 0)]
    df["StockCode"] = df["StockCode"].str.upper().str.strip()
    # Drop non-product codes such as POST, DOT, M, BANK CHARGES, AMAZONFEE.
    df = df[df["StockCode"].str.match(r"^\d{5}")]
    df["date"] = df["InvoiceDate"].dt.normalize()
    df = df[["Invoice", "StockCode", "Description", "Quantity", "Price", "CustomerID", "Country", "date"]]
    CLEAN.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(CLEAN, index=False)
    return df


def main():
    df = load()
    print(f"rows {len(df):,}  dates {df['date'].min().date()} to {df['date'].max().date()}")
    by = df.groupby("Country").agg(orders=("Invoice", "nunique"), lines=("Invoice", "size"),
                                   units=("Quantity", "sum"), products=("StockCode", "nunique"),
                                   months=("date", lambda s: s.dt.to_period("M").nunique()))
    uk_codes = set(df.loc[df["Country"] == "United Kingdom", "StockCode"])
    by["shared_with_uk"] = [len(set(df.loc[df["Country"] == c, "StockCode"]) & uk_codes) for c in by.index]
    by["orders_per_month"] = (by["orders"] / by["months"]).round(1)
    by = by.sort_values("orders", ascending=False)
    pd.set_option("display.width", 160)
    print(by.head(25).to_string())
    by.to_csv(ROOT / "results" / "country_summary.csv")


if __name__ == "__main__":
    main()
