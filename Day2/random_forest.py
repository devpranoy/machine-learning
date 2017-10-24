from sklearn.ensemble import RandomForestClassifier

from sklearn.datasets import load_iris

iris= load_iris()

features = iris.data
labels = iris.target

from sklearn.cross_validation import train_test_split

X_train,X_test ,Y_train,Y_test=train_test_split(features,labels,test_size=.3)

#define classifier

clf=RandomForestClassifier()
clf.fit(X_train,Y_train)

p=clf.predict(X_test)

from sklearn.metrics import accuracy_score
print("Accuracy = ",accuracy_score(Y_test,p) )