import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn import metrics

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

# Divide o conjunto de dados utilizando a proporção 80:20
train, test = train_test_split(data, test_size = 0.2)

features = list(data.columns[2:])

X = train[features].values
y = train['subject'].values

net = MLPClassifier(hidden_layer_sizes=(300, ), max_iter=600)

net.fit(X, y)

x_test = test[features]
y_test = test['subject']

predicted_output = net.predict(x_test)

model_accuracy = metrics.accuracy_score(y_test, predicted_output)
print(model_accuracy)
