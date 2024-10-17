import pyarrow.parquet as pq

# Read the Parquet file
table = pq.read_table("Data/tmp/arxiv/papers_100/data/kaggle_data.parquet")

# Convert it to a pandas DataFrame (optional)
df = table.to_pandas()

# Display the DataFrame
print(df.head())