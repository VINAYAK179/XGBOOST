import pandas as pd
import numpy as np

dataset=pd.read_csv('V:\\MLE TAK\\mle_task\\train.csv')
total=dataset.iloc[:,1:7].values
import random
random.shuffle(total)
X=total[:,0:5]
y=total[:,5:6]

from sklearn.preprocessing import Imputer
imputer=Imputer(missing_values='NaN',strategy='mean',axis=0)
imputer=imputer.fit(X[:,0:5])
X[:,0:5]=imputer.transform(X[:,0:5])

from sklearn.cross_validation import train_test_split
X_train,X_test1,y_train,y_test1=train_test_split(X,y,test_size=0.2,random_state=0)

from xgboost import XGBClassifier
classifier=XGBClassifier()
classifier.fit(X_train,y_train)

y_pred=classifier.predict(X_test1)

from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test1,y_pred)

classifier_total=XGBClassifier()
classifier_total.fit(X,y)

dataset_test = pd.read_csv('V:\\MLE TAK\\mle_task\\test.csv')
X_test = dataset_test.iloc[:,1:6].values
imputer1=imputer.fit(X_test[:,2:5])
X_test[:,2:5]=imputer1.transform(X_test[:,2:5])

y_pred_true=classifier_total.predict(X_test)
y_pred_true=np.array(y_pred_true)


df = pd.DataFrame(y_pred_true)
df.to_csv("V:\\MLE TAK\\mle_task\\predicted.csv")

