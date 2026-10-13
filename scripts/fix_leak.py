"""Write a copy of the training CSV without leaking columns."""

import pandas as pd

DROP = ["Diabetes_Risk_Score", "AI_Health_Recommendation", "Doctor_Consultation_Needed"]

df = pd.read_csv("data/train.csv")
df.drop(columns=DROP).to_csv("data/train_clean.csv", index=False)
print(f"Wrote data/train_clean.csv without {DROP}")
