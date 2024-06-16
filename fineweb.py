import pandas as pd
from datatrove.pipeline.readers import ParquetReader
from functools import lru_cache

limit = 1000

data_reader = ParquetReader(
    "hf://datasets/HuggingFaceFW/fineweb-edu/sample/10BT",
    limit=limit,
)


@lru_cache(maxsize=None)
def create_df(data_reader):
    docs = []
    for document in data_reader():
        # Append each document to the list
        docs.append(document)

    return pd.DataFrame(docs)


# Convert the list of documents into a pandas DataFrame
DF = create_df(data_reader)

# Display the DataFrame
print(DF.columns)

with open("fineweb.txt", "w", encoding="utf-8") as f:
    for i in range(limit):
        # print(f"Text from iteration {i}: \n\n")
        f.write(DF.iloc[i]["text"])
