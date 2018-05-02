import pandas as pd
import numpy as np

df = pd.read_csv("../data/iris.csv")
df["species"] = df["species"].map({"setosa": 0, "versicolor": 1, "virginica": 2})

# Split the dataset in two using a boolean mask with an 80% probability
np.random.seed(18)
msk = np.random.rand(len(df)) < 0.8
train = df[msk]
test = df[~msk]

# Write train and test csv to disk
train.to_csv("../data/train.csv", index=False, header=False)
test.to_csv("../data/test.csv", index=False, header=False)
