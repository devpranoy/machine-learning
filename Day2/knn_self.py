from scipy.spatial import distance

def eucli(a,b):
	return distance.euclidean(a,b)

class myKNN():
	#training method fit
	def fit(self,X_train,Y_train):
		self.X_train= X_train		#Increased the scope of the variables to extend throughout the classes
		self.Y_train=Y_train
	def predict(self,X_test):
		predictions=[]
		for row in X_test:
			labels=self.closest(row)
			predictions.append(labels)
		return predictions
	def closest(self,row):
		best_dist=eucli(row,self.X_train[0])
		best_index=0
		for i in range(1,len(self.X_train)):
			dist=eucli(row,self.X_train[i])
			if dist<best_dist:
				best_dist=dist
				best_index=i
		return self.Y_train[best_index]

from sklearn.neighbors import KNeighborsClassifier

from sklearn.datasets import load_iris

iris= load_iris()

features = iris.data
labels = iris.target

from sklearn.cross_validation import train_test_split

X_train,X_test ,Y_train,Y_test=train_test_split(features,labels,test_size=.3)

#define classifier

clf=myKNN()
clf.fit(X_train,Y_train)

p=clf.predict(X_test)

from sklearn.metrics import accuracy_score
print("Accuracy = ",accuracy_score(Y_test,p) )