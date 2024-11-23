import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
from matplotlib import pyplot as plt
import os
import pickle
from sklearn.model_selection import train_test_split


#https://www.kaggle.com/datasets/sulianova/cardiovascular-disease-dataset/data

df = pd.read_csv("cardio_train.csv", sep=";")

print(df.head())

print(df.info())

from matplotlib import rcParams

rcParams['figure.figsize'] = 11, 8
df['years'] = (df['age'] / 365).round().astype('int')
sns.countplot(x='years', hue='cardio', data=df, palette="Set2");
plt.show()
df_categorical = df.loc[:, ['cholesterol', 'gluc', 'smoke', 'alco', 'active']]
sns.countplot(x="variable", hue="value", data=pd.melt(df_categorical));
plt.show()

df_long = pd.melt(df, id_vars=['cardio'], value_vars=['cholesterol', 'gluc', 'smoke', 'alco', 'active'])
sns.catplot(x="variable", hue="value", col="cardio",
            data=df_long, kind="count");
plt.show()
df.isnull().values.any()

X = df[['age', 'gender', 'height', 'weight', 'ap_hi', 'ap_lo', 'cholesterol', 'gluc', 'smoke', 'alco', 'active']]

y = df['cardio']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=0)

from sklearn.neural_network import MLPClassifier
#from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# Train the model
#classifier = MLPClassifier(hidden_layer_sizes=(100, 50), activation='relu', solver='adam', max_iter=500)
classifier = MLPClassifier()
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)
print(classification_report(y_test, y_pred))

clreport = classification_report(y_test, y_pred)

print("Accuracy on training set: {:.2f}".format(classifier.score(X_train, y_train)))
print("Accuracy on test set: {:.3f}".format(classifier.score(X_test, y_test)))

Tacc = "Accuracy on training set: {:.2f}".format(classifier.score(X_train, y_train))
Testacc = "Accuracy on test set: {:.3f}".format(classifier.score(X_test, y_test))

# Creating a pickle file for the classifier
filename = 'heart-model.pkl'
pickle.dump(classifier, open(filename, 'wb'))
'''data = np.array([[age, gender, height, weight, aphi, aplo, choles, glucose, smoke, alcohol]])
my_prediction = classifier2.predict(data)
print(my_prediction[0])'''