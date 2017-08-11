import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import metrics
from sklearn import svm, datasets
from sklearn.model_selection import train_test_split

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

features = list(data.columns[2:])

X = train[features].values
y = train['subject'].values

# we create an instance of SVM and fit out data. We do not scale our
# data since we want to plot the support vectors
C = 3  # SVM regularization parameter
#svc = svm.SVC(kernel='linear', C=C).fit(X, y)
#rbf_svc = svm.SVC(kernel='rbf', C=C).fit(X, y)
poly_svc = svm.SVC(kernel='poly', degree=3, C=C, gamma=10)
#lin_svc = svm.LinearSVC(C=C).fit(X, y)


poly_svc.fit(X, y)

x_test = test[features]
y_test = test['subject']
# predict the output using the test data on the learned model
predicted_output = poly_svc.predict(x_test)

model_accuracy = metrics.accuracy_score(y_test, predicted_output)
model_accuracy