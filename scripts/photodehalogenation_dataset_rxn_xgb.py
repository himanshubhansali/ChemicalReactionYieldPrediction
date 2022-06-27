#!/usr/bin/env python

# date: 14 May 2022
# author: himanshubhansali978@gmail.com(himanshubhansali)
# purpose: Prediction of Photodehalogenation rxn yields
# revision:
# Copyright: (C) 2021 Caliche Pvt. Ltd. All right reserved.


#Importing Essential Libraries

import pandas as pd
import numpy as np
import rxnfp
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from drfp import DrfpEncoder
from rdkit import Chem
from rdkit.Chem import rdChemReactions

# Data Preprocessing

df=pd.read_excel('/content/Photodehalogenation.xlsx')
df

#Make_reaction smiles function used to create the reaction smiles from the photodehalogenation dataset

def make_reaction_smiles(i):
    precursors = f" {df['Reactant1'][i]}.{df['Reactant2'][i]}.{df['Catalyst1'][i]}.{df['Catalyst2'][i]}.{df['Catalyst3'][i]}.{df['Reagent'][i]}.{df['Solvent'][i]} "
    product = f" {df['Product1'][i]}.{df['Product2'][i]} "
    #print(precursors, product)
    return f"{precursors}>>{product}"
df['rxn']= [make_reaction_smiles(i) for i, row in df.iterrows()]
df['y'] = df['Yield']/ 100.
reactions_df = df[['rxn', 'y']]


# Encoding the Data

X, mapping = DrfpEncoder.encode(
            reactions_df.rxn.to_numpy(),
            n_folded_length=2048,
            radius=3,
            rings=True,
            mapping=True,
        )
 X = np.asarray(
            X,
            dtype=np.float32,
        )
 y = reactions_df.y.to_numpy()

X

# Importing XGB Regressor 

import pickle
from pathlib import Path
from typing import Tuple
from statistics import stdev
import numpy as np
from xgboost import XGBRegressor
from sklearn.metrics import r2_score

#Splitting the Dataset ito train,test and valid with 85% of data is used for traning the model and other 15% is uused for validation

X_df=X[:1000]
X_test=X[1000:]
y_df=y[:1000]
y_test=y[1000:]
X_train=X_df[:850]
y_train=y_df[:850]
X_valid=X_df[850:]
y_valid=y_df[850:]

X_test

# Model Training

#We used the same value for our hyperparameters for our XGB Regressor model ,i.e., n-estimators=999999, learning rate =0.1, max_depth=12,
#min_child_weight=8,colsample_bytree=0.6.


model = XGBRegressor(
                n_estimators=999999,
                learning_rate=0.1,
                max_depth=12,
                min_child_weight=8,
                colsample_bytree=0.6,
                subsample=0.8,
                random_state=42,
            )
 model.fit(
                X_train,
                y_train,
                eval_set=[(X_valid, y_valid)],
                early_stopping_rounds=20,
                verbose=False,
            )

y_pred = model.predict(X_test, ntree_limit=model.best_ntree_limit)

y_pred

#We achieved a R2 score of 0.9097 on our first run for prediction task

r_squared=r2_score(y_test,y_pred)
r_squared