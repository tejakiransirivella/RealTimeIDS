import sys
import pandas as pd
from ml import DecisionTree as dt
from ml import NeuralNetwork,ActivationFunction,LossFunction
import numpy as np
data_file = sys.argv[1]
model_file = sys.argv[2]
model=sys.argv[3]
tree=dt()
df=pd.read_csv(data_file)

limit=int(sys.argv[4]) if len(sys.argv)>4 else len(df)
def trainDecisionTree():
    X=df.iloc[:limit,:-2].values
    y=df.iloc[:limit,-2].values
    #print(sum([1 for i in y if i=="DoS"]))
    tree.train(X,y)
    print('Training complete')
    tree.saveModel(model_file)

def getUniqueLabels(y):
    labelSet=set()
    labels=[]
    cnt=0
    for label in y:
        if label not in labelSet:
            labelSet.add(label)
            labels.append(label)
            cnt+=1
    return labels
    

def trainNeuralNetwork():
    X=df.iloc[:limit,:-1].values
    y=df.iloc[:limit,-1]
    labels=getUniqueLabels(y)
    y=y.apply(lambda x: labels.index(x))
    y=np.eye(len(labels))[y]
    print(X.shape,labels)
    nn=NeuralNetwork(X.shape[1],[76],len(labels),ActivationFunction.RELU)
    nn.showLoss(True)
    nn.setLossFunction(LossFunction.CROSS_ENTROPY)
    nn.setBatchSize(100)
    nn.saveModel("untrained"+model_file)
    nn.train(X,y,1,0.1)
    nn.saveModel(model_file)

if __name__ == "__main__":
    if model=="dt":
        trainDecisionTree()
    else:
        trainNeuralNetwork()