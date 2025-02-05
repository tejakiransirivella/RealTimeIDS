from decision_tree import DecisionTree as dt # type: ignore
from pathlib import Path

import numpy as np
import pandas as pd
tree=dt()
absolute_path = Path("../models/tree.model").resolve()
print(absolute_path)
tree.loadModel("../models/tree.model")
df=pd.read_csv("../data/train.csv")
limit=10
columnSet=set(df.columns)
X=df.iloc[:limit,:-1].values
y=df.iloc[:limit,-1].values
print(tree.test(X,y))