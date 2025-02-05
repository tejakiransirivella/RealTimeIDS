
import numpy as np
import pandas as pd
import sys
from config import Config


config = Config()
sys.path.append(config.get_build())
from decision_tree import DecisionTree as dt
tree=dt()
df=pd.read_csv(config.get_data("columns.csv"))
columnSet=set(df.columns)
tree.loadModel(config.get_model("tree.model"))
  
    
def detect_intrusion(data):
    """
    Detects whether the flow is malicious or benign and classifies it accordingly.
    """
    flows=data['flows']
    arr=np.zeros((len(flows),len(df.columns)))
    for i,flow in enumerate(flows):
        for key in flow:
            if key in columnSet:
                col=df.columns.get_loc(key)
                arr[i,col]=flow[key]
    return tree.predict(arr)

# train()

