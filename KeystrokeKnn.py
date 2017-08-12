import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler


# Carrega os dados
data = pd.read_csv("DSL-StrongPasswordData.csv", header = 0)
# Reinicia o index
data = data.reset_index()
# Pega os IDs unicos campo 'subject'
unisub = list(data['subject'].unique())
# Cria ID numerico sequencial
mlist = [int(x) for x in range(len(unisub))]
# Vincula o Id numerico com o campo 'subject'
newvalue = dict(zip(unisub, mlist))
# Exibe dados tratados
data['subject'] = data['subject'].map(newvalue)


# import some data to play with


# Divide o conjunto de dados utilizando a proporção 80:20
train, test = train_test_split(data, test_size = 0.2)

#iris = datasets.load_iris()

features = list(data.columns[2:])

X = train[features].values
y = train['subject'].values


neigh = KNeighborsClassifier(n_neighbors=21, weights ='uniform', algorithm='brute',metric='manhattan',n_jobs =-1).fit(X, y) 

x_test = test[features]
y_test = test['subject']
# predict the output using the test data on the learned model
predicted_output = neigh.predict(x_test)

model_accuracy = metrics.accuracy_score(y_test, predicted_output)
model_accuracy

