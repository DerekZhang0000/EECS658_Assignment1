from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix
import numpy as np
import pandas as pd

# Reads data from csv
df = pd.read_csv("iris.csv", header=None)
# Shuffles data
df = df.iloc[np.random.RandomState(seed=0).permutation(len(df))]

# X is a list of lists. The inner list is comprised of the characteristics of the corresponding flower type.
X = df.drop(df.columns[[4]], axis=1).to_numpy()
# Y is a list of flower types that correspond to a list of characteristics in X
Y = df.transpose().iloc[4].to_numpy()

# Splits sample in half
Train_X = X[:75]
Train_Y = Y[:75]
Test_X = X[75:]
Test_Y = Y[75:]

# Creates Naive Bayesian classifier and trains it
classifier = GaussianNB()
classifier.fit(Train_X, Train_Y)

# Creates predictions
Predict_Y = classifier.predict(Test_X)

# Creates confusion matrix
Flower_Types = ["Iris-setosa", "Iris-versicolor", "Iris-virginica"]
ConfusionMatrix = confusion_matrix(Test_Y, Predict_Y, labels=Flower_Types)

# Calculates P-Scores
P_Scores = [ConfusionMatrix[i][i] / sum(ConfusionMatrix[i]) for i in range(len(ConfusionMatrix))]

# Calculates R-Scores
TransposedMatrix = ConfusionMatrix.transpose()
R_Scores = [TransposedMatrix[i][i] / sum(TransposedMatrix[i]) for i in range(len(TransposedMatrix))]

# Calculates F1-Scores
F1_Scores = [2 * (P_Scores[i] * R_Scores[i]) / (P_Scores[i] + R_Scores[i]) for i in range(len(ConfusionMatrix))]

# Prints overall accuracy and confusion matrix
print("Overall Accuracy:", classifier.score(Test_X, Test_Y))
print("\nConfusion Matrix:\n", ConfusionMatrix, sep="")

# Prints P-Score, R-Score, and F1-Score for each flower type
for i in range(len(Flower_Types)):
    print("\n", Flower_Types[i], "\nP-Score: ", P_Scores[i], "\nR-Score: ", R_Scores[i], "\nF1-Score: ", F1_Scores[i], sep="")