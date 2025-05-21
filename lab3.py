import pandas as pd, numpy as np

data = pd.read_csv("enjoysport.csv")
X, y = np.array(data.iloc[:, :-1]), np.array(data.iloc[:, -1])
S = X[0].copy()
G = [["?" for _ in S] for _ in S]

for i, x in enumerate(X):
    if y[i] == "yes":
        S = ["?" if S[j] != x[j] else S[j] for j in range(len(S))]
    else:
        for j in range(len(S)):
            if S[j] != x[j]:
                G[j][j] = S[j]
            else:
                G[j][j] = "?"

G = [g for g in G if g != ["?"] * len(S)]
print("Final Specific Hypothesis:\n", S)
print("Final General Hypotheses:\n", G)
