import numpy as np
import pandas as pd
from sklearn import *
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import warnings
import pickle
warnings.filterwarnings("ignore")

df = pd.read_csv("Breast_cancer_data.csv")

#training and testing data

X =  df.drop('diagnosis',axis= 1)
y = df['diagnosis']
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X,y,test_size = 0.3,random_state = 0)

from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train, y_train)

with open('model.pkl','wb') as file:
    pickle.dump(classifier,file)

