import polars as pl


df = pl.read_parquet("datasets/data.parquet")
print(len(df))
