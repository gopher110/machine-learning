from sklearn.datasets import load_breast_cancer
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.preprocessing import scale
import pandas as pd
from time import time

bc = load_breast_cancer()

# 将每一列特征标准化为标准正太分布，注意，标准化是针对每一列而言的
X = scale(bc.data)

print(X)

y = bc.target

X_train, X_test, y_train, y_test = train_test_split(X, y)

model = KMeans(n_clusters=2, random_state=0)

model.fit(X_train)

predictions = model.predict(X_test)
labels = model.labels_

print(f'labels: {labels} \npredictions: {predictions}')
print(f'accuracy:{metrics.accuracy_score(y_test, predictions)}\n actual:{y_test}')
print(pd.crosstab(y_train, labels))

sample_size = 300


def bench_k_means(estimator, name, data):
    t0 = time()
    estimator.fit(data)
    print('%-9s\t%.2fs\t%i\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f'
          % (name, (time() - t0), estimator.inertia_,
             metrics.homogeneity_score(y, estimator.labels_),
             metrics.completeness_score(y, estimator.labels_),
             metrics.v_measure_score(y, estimator.labels_),
             metrics.adjusted_rand_score(y, estimator.labels_),
             metrics.adjusted_mutual_info_score(y,  estimator.labels_),
             metrics.silhouette_score(data, estimator.labels_,
                                      metric='euclidean',
                                      sample_size=sample_size)))
    pass


bench_k_means(model, name='1', data=X)
