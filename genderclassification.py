from sklearn.tree import DecisionTreeClassifier
features=[[181,80,44],[177,70,43],[160,60,38],[154,54,37],[166,65,40],[190,90,47],[175,64,39],[177,70,40],[159,55,37],[171,75,42],[185,85,43]]#smooth is 0 and bumpy is 1
labels=[1,1,-1,-1,1,1,-1,-1,-1,1,1]

#initialise decision tree classifier
clf= DecisionTreeClassifier()
clf.fit(features,labels)

p=clf.predict([190,70,43]) #predicting a given value
if p==1:
	print ("male")

else :
	print ("Female")