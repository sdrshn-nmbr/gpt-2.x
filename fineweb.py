import pandas as pd
from datatrove.pipeline.readers import ParquetReader
from functools import lru_cache

limit = 10000

data_reader = ParquetReader(
    "hf://datasets/HuggingFaceFW/fineweb-edu/sample/10BT", limit=limit,
)


@lru_cache(maxsize=None)
def create_df(data_reader):
    docs = []
    for document in data_reader():
        # Append each document to the list
        docs.append(document)

    return pd.DataFrame(docs)


# Initialize an empty list to store documents
documents = create_df(data_reader)

# Convert the list of documents into a pandas DataFrame
DF = pd.DataFrame(documents)

# Display the DataFrame
print(DF.columns)

for i in range(limit):
    print(f"Text from iteration {i}: \n\n")
    print(DF.iloc[i]["text"])
