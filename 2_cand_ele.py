import pandas as pd
df=pd.read_csv("finds.csv",sep=",",header=None)
#Intialize S and G
S=[0,0,0,0,0,0]
G=list()
for i in range(len(df.columns)-1):
    G.append(['?','?','?','?','?','?'])
#Read samples
for i in range(len(df)):
    for j in range(len(df.columns)-1):
        if df.iloc[i,-1]=="Yes":
            if S[j]==0:
                S[j]=df.iloc[i,j]
            elif df.iloc[i,j]!=S[j]:
                S[j]="?"
            if G[j][j]!='?' and S[j]=='?':
                G[j][j]='?'
        else:
            if df.iloc[i,j]!=S[j] and S[j]!='?':
                G[j][j]=S[j]
print(S)
print(G)