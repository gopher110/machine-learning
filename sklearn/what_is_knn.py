import numpy as np
import pandas as pd
from sklearn import neighbors,metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

data = pd.read_csv('carData.csv')

X = data[[
    'buying',
    'maint',
    'safety'
]].values
y = data[['class']]

# converting the data
Le = LabelEncoder()
for i in range(len(X[0])):
    X[:, i] = Le.fit_transform(X[:, i])

# y
y = y.replace({
    'unacc': 0,
    'acc': 1,
    'good': 2,
    'vgood': 3
})

# create model
knn = neighbors.KNeighborsClassifier(n_neighbors=25, weights='uniform')
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

knn.fit(X_train, y_train.values.ravel())

prediction = knn.predict(X_test)

accuracy = metrics.accuracy_score(y_test, prediction)

print("predictions: ", prediction)
print("accuracy: ", accuracy)

a = 23
print('actual value: ', y['class'][a])
print('predicted value: ', knn.predict(X)[a])
