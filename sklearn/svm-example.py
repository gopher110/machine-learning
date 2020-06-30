from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score
from joblib import dump, load

# sk learn 自带一些经典数据集 iris,digits,boston house price
# Iris数据集是常用的分类实验数据集，由Fisher, 1936收集整理。Iris也称鸢尾花卉数据集，是一类多重变量分析的数据集。数据集包含150个数据样本，分为3类，
# 每类50个数据，每个数据包含4个属性。可通过花萼长度，花萼宽度，花瓣长度，花瓣宽度4个属性
# 预测鸢尾花卉属于（Setosa，Versicolour，Virginica）三个种类中的哪一类。

iris = datasets.load_iris()

#  split it in features and labels
X = iris.data
y = iris.target

classes = ['Setosa', 'Versicolour', 'Virginica']


print(X.shape)
print(y.shape)

# hours of study vs good/bad grades
# 10 different students
# train with 8
# predict wit the remaining 2
# level of accuracy

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = svm.SVC()
model.fit(X_train, y_train)

filename = 'svm-example.sav'
dump(model, filename)
svmExampleModel = load(filename)

predictions = svmExampleModel.predict(X_test)
acc = accuracy_score(y_test, predictions)

print("prediction: ", predictions)
print("actual: ", y_test)
print("accuracy", acc)

for i in range(len(predictions)):
    print(classes[predictions[i]])