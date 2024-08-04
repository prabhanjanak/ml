import pandas as pd
df=pd.read_csv("iris1.csv")
x=df.iloc[:,:-1]
y=df.iloc[:,-1]
y=y.astype('category')
y=y.cat.codes
from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.3)
from sklearn.naive_bayes import GaussianNB
g=GaussianNB()
g.fit(xtrain,ytrain)
ypred=g.predict(xtest)
from sklearn.metrics import accuracy_score
print(accuracy_score(ytest,ypred))