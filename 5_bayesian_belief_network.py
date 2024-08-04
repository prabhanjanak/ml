import pandas as pd
import numpy as np
from pgmpy.inference import VariableElimination
from pgmpy.models import BayesianModel
from pgmpy.estimators import MaximumLikelihoodEstimator
df=pd.read_csv("Medical Dataset.csv")
df=df.replace("?",np.nan)
model=BayesianModel([('age','heartdisease'),('sex','heartdisease'),
('exang','heartdisease'),('cp','heartdisease'),
('heartdisease','restecg'),('heartdisease','chol')])
model.fit(df,estimator=MaximumLikelihoodEstimator)
infer=VariableElimination(model)
q=infer.query(variables=['heartdisease'],evidence={'restecg':1})
print(q)