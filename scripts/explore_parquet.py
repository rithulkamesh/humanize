from polars import read_parquet

df = read_parquet("datasets/replacements.parquet")
print(df)
