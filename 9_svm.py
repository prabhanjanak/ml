import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
iris=pd.read_csv("Iris.csv")
x=iris.iloc[:,:-1]
y=iris.iloc[:,-1]
y=y.astype('category')
y=y.cat.codes
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.3)
svmclf=make_pipeline(StandardScaler(),LinearSVC(C=15))
svmclf.fit(xtrain,ytrain)
ypred=(svmclf.predict(xtest))
acc=accuracy_score(ytest,ypred)
print(acc)