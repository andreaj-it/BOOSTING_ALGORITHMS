import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
import pickle

from xgboost import XGBClassifier

from sklearn import metrics
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import classification_report

#Step 1

df = pd.read_csv('https://raw.githubusercontent.com/4GeeksAcademy/random-forest-project-tutorial/main/titanic_train.csv')

#Step 2

df=df.drop(columns=['Name'])

df['Sex_encoded']=df['Sex'].apply(lambda x: 1 if x == 'female' else 0 )

df['Sex'] = df['Sex'].map({'male':1,'female':0})

df=df.drop(columns=['Ticket'])

df=df.drop(columns=['Cabin'])

df['Embarked'] = df['Embarked'].map({'S':2,'C':1,'Q':0})

df['Age'][np.isnan(df['Age'])]=df['Age'].mean()

df['Embarked'][np.isnan(df['Embarked'])]=2

X = df.drop(['Survived','Sex_encoded'], axis=1)
y = df['Survived']

X_train, X_test, y_train, y_test = train_test_split(X,y, random_state=70)

# Creo el modelo

model3 = XGBClassifier()

model3.fit(X,y)


xgb.plot_importance(model3, ax= plt.gca())

predict = model3.predict(X_test)

model3.score(X_train, y_train)

pred_train = model3.predict(X_train)



confusion_matrix(y_train, pred_train)



filename = '../models/final_model.sav'
pickle.dump(model3, open(filename, 'wb'))