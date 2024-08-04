import pandas as pd
import numpy as np

# Initialize the hypothesis
H = [0] * 6

# Read the CSV file
df = pd.read_csv("finds.csv", header=None)

# Extract attributes and target
attributes = df.iloc[:, :-1].values
target = df.iloc[:, -1].values

# Update hypothesis based on target values
for i in range(len(target)):
    if target[i] == "Yes":
        for j in range(len(attributes[i])):
            if H[j] == 0:
                H[j] = attributes[i][j]
            elif H[j] != attributes[i][j]:
                H[j] = '?'

# Display the final hypothesis
print(H)
