import os
import sys
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer

import utils as u
from ml import ml_analysis
from dl import dl_analysis

le = LabelEncoder()
si_mean = SimpleImputer(missing_values=np.nan, strategy="mean")

#####################################################################################################

def app():
    
    data = pd.read_csv(os.path.join(u.DATA_PATH, "data.csv"), engine="python")
    
    # u.breaker()
    # for col in data.columns:
    #     print(col + " - " + repr(data[col].nunique()))
    # u.breaker()
    # print(data.isnull().any())
    # u.breaker()
    # print(data.mean().mean())
    # print(data.std().std())
    # u.breaker()

    features = data.iloc[:, 3:].copy().values
    features[:, 2] = le.fit_transform(features[:, 2])
    features[:, 3] = le.fit_transform(features[:, 3])
    features = si_mean.fit_transform(features)

    targets  = data.iloc[:, 2].copy().values
    targets  = le.fit_transform(targets)

    args_1 = "--ml"
    args_2 = "--dl"

    if args_1 in sys.argv:
        ml_analysis(features=features, targets=targets)
    if args_2 in sys.argv:
        dl_analysis(features=features, targets=targets)

#####################################################################################################
