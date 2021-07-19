import sklearn
import pickle
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix
data=pd.read_csv("android_dataset-v1.csv")
Y = data['class']#label
X = data.drop(['class'], axis=1)#feature

###splite the dataset##########

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)

# defining parameter range 
param_grid = {'C': [0.1, 1, 10, 100, 1000], 
			'gamma': [1, 0.1, 0.01, 0.001, 0.0001], 
			'kernel': ['rbf']} 

grid = GridSearchCV(SVC(), param_grid, refit = True, verbose = 3) 
grid.fit(X_train, y_train)#train data
grid_predictions = grid.predict(X_test) 

# print classification report 
print(classification_report(y_test, grid_predictions)) 
pickle.dump(grid, open('svc_new.pkl', 'wb'))
from androguard.core.bytecodes.apk import APK

###prediction APK file 
def predict(apk):
  vector = {}
  a = APK(apk)
  perm = a.get_permissions()
  print(perm)
  for d in perms:
    if d in perm:
      vector[d]=1
    else:
      vector[d]=0
  input = [ v for v in vector.values() ]
  print(input)
  print(grid.predict([input]))
###################Multilayer Perceptron (Simple Artificial Neural Network)######
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation, Embedding, LSTM
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix 
import pickle
import numpy as np
from sklearn.preprocessing import LabelEncoder
data=pd.read_csv("android_dataset-v1.csv")
Y = data['class']
X = data.drop(['class'], axis=1)

encoder = LabelEncoder().fit(Y)
Y = encoder.transform(Y)
print(encoder.transform(['malign']))
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
AN = Sequential()
AN.add(Dense(256, activation='relu', input_dim=409))
AN.add(Dropout(0.2))
AN.add(Dense(128, activation='relu'))
AN.add(Dropout(0.2))
AN.add(Dense(128, activation='relu'))
AN.add(Dropout(0.2))
AN.add(Dense(32, activation='relu'))
AN.add(Dropout(0.2))
AN.add(Dense(1, activation='sigmoid'))
AN.compile(optimizer='sgd', loss='binary_crossentropy', metrics=['accuracy'])

AN.fit(X_train, y_train, epochs=100, batch_size=32)# ANN train model


scores = AN.evaluate(X_test, y_test)
for i in range(len(scores)):
  print("\n%s: %.2f%%" % (AN.metrics_names[i], scores[i]*100))
from androguard.core.bytecodes.apk import APK

def predict(apk):
  vector = {}
  a = APK(apk)
  perm = a.get_permissions()
  print(perm)
  for d in perms:
    if d in perm:
      vector[d]=1
    else:
      vector[d]=0
  input = [ v for v in vector.values() ]
  print(input)
  print(AN.predict([[input]]))
pickle.dump(grid, open('ann_new.pkl', 'wb'))
