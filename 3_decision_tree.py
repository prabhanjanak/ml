import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
from sklearn import tree

# Load the data
df = pd.read_csv("decision_tree1.csv", sep=",")

# Encode categorical features as numeric codes
for col in df.columns:
    if df[col].dtype == 'object':
        df[col] = df[col].astype('category')
        df[col] = df[col].cat.codes

# Prepare the feature matrix and target vector
x = df.iloc[:, :-1]
y = df.iloc[:, -1]

# Train the Decision Tree classifier
model = DecisionTreeClassifier(criterion="entropy")
clf = model.fit(x, y)

# Plot the tree
plt.figure(figsize=(6,4))
tree.plot_tree(clf, feature_names=x.columns.tolist(), filled=True)
plt.show()

# Verify the number of features in the training data
print(f"Number of features in training data: {x.shape[1]}")

# Adjust new_data to have the same number of features
# Replace the list with the appropriate number of features for your dataset
new_data = [[1, 0, 0, 1]]  # Example adjustment

# Predict using new data
ypred = clf.predict(new_data)
print(ypred)