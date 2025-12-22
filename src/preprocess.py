import pandas as pd

def load_and_preprocess(path):
    df = pd.read_csv(path)
    X = df.drop("Cover_Type", axis=1)
    y = df["Cover_Type"]
    return X, y
