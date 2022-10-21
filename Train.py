# Import
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn import __version__;
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import joblib

# Inmport Data
df = pd.read_csv("./Data/Fish.csv")

# Process data
df["Species"] = df["Species"].astype("category") # Type casting Species column to category datatype
d = dict(enumerate(df["Species"].cat.categories)) # Get dictionary of representation of Species
df["Species"] = df["Species"].cat.codes # Encode Species column
print(d)
# Split features and Label
X = df.drop(columns = "Species")
Y = df["Species"]

# split data into training and test
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size= 0.2, random_state= 0) # 20% for test

# Create model
Lr = LogisticRegression(solver='lbfgs', max_iter=100000)

# Train model
Lr.fit(x_train, y_train)


# Test predict accuracy
print(classification_report(y_test, Lr.predict(x_test)))

# Export model
# sklearn_version = __version__
# joblib.dump(Lr,"model_{version}.pkl".format(version=sklearn_version))

