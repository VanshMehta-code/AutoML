"""Rank each input column by how well it predicts the target on its own."""

import argparse

import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier

parser = argparse.ArgumentParser()
parser.add_argument("--data", required=True)
parser.add_argument("--target", required=True)
parser.add_argument("--top", type=int, default=10)
args = parser.parse_args()

df = pd.read_csv(args.data)
y = df[args.target].astype("category").cat.codes
X = df.drop(columns=[args.target])

results = []
for name in X.columns:
    col = X[name]
    if pd.api.types.is_numeric_dtype(col):
        col = col.fillna(col.median())
    else:
        col = col.astype("category").cat.codes
    score = cross_val_score(
        DecisionTreeClassifier(max_depth=3), col.to_frame(), y, cv=3
    ).mean()
    results.append((score, name))

for score, name in sorted(results, reverse=True)[: args.top]:
    print(f"{score:.3f}  {name}")
