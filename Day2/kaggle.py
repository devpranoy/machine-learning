import numpy as np    
import matplotlib.pyplot as plt 
import pandas as pd 
from sklearn.tree import DecisionTreeClassifier


data = pd.read_csv("train.csv").as_matrix()
print("Data \n",data)
clf= DecisionTreeClassifier()
X_train=data[0:21000,1:]
Y_train=data[0:21000,0]

X_test=data[21000:,1:]
Y_test=data[21000:,0]

disp =X_test[8]
disp.shape=(28,28)
plt.imshow(255-disp,cmap="gray")
plt.show()

p=clf.predict([X_test[8]])
print("Prediction= " ,p)