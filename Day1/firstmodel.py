from sklearn.tree import DecisionTreeClassifier
features=[[140,0],[130,0],[150,1],[170,1]] #smooth is 0 and bumpy is 1
labels=[0,0,1,1]

#initialise decision tree classifier
clf= DecisionTreeClassifier()
clf.fit(features,labels)

p=clf.predict([160,1]) #predicting a given value
print ("Prediction= ",p)
