import pandas as pd
import numpy as np
from matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
import joblib
import os

from azureml.core import Run

run = Run.get_context()

print("Loading Data...")
penguins = pd.read_csv("penguin-data.csv")
penguins = penguins.dropna()
penguin_features = ['CulmenLength','CulmenDepth','FlipperLength','BodyMass']
penguin_label = 'Species'

penguin_X, penguin_y = df[penguin_features].values,df[penguin_label].values
x_train, x_test, y_train, y_test = train_test_split(penguin_X,penguin_y, test_size=0.3, random_state=0, stratify=penguin_y)

print("Training model...")
reg=0.1
model = LogisticRegression(C=1/reg, solver='lbfgs', multi_class='auto', max_iter=10000).fit(x_train,y_train)

penguin_predictions = model.predict(x_test)

#scores
run.log("Accuracy", np.float(accuracy_score(y_test, penguin_predictions)))
run.log("Precision", np.float(precision_score(y_test, penguin_predictions, average="macro")))
run.log("Recall", np.float(recall_score(y_test, penguin_predictions, average="macro")))


c_matrix = confusion_matrix(y_test,penguin_predictions)
fig = plt.figure(figsize=8,8)
plt.imshow(c_matrix, interpolation="nearest", cmap=plt.cm.Blues)
plt.colorbar()
penguin_classes = ['Adelie','Chinstrap','Gentoo']
tick_marks = np.arange(len(penguin_classes))
plt.xticks(tick_marks, penguin_classes, rotation=45)
plt.yticks(tick_marks, penguin_classes)
plt.xlabel("Predicted species")
plt.ylabel("Actual Species")
plt.show()
run.log_image(name = 'confusion_matrix', plot = fig)

os.makedirs('outputs',exist_ok=True)
joblib.dump(value=model, filename = 'outputs/penguin-model.pkl')