#!/usr/bin/env python

# date: 18 April 2022 
# author: himanshubhansali978@gmail.com(himanshubhansali@github)
# purpose: Predicting Buchwald hartwig rxn yields
# revision:
# Copyright: (C) 2021 Caliche Pvt. Ltd. All right reserved.


# Importing Essential Libraries
!pip install drfp

import pandas as pd
import numpy as np

from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from drfp import DrfpEncoder
from rdkit import Chem
from rdkit.Chem import rdChemReactions

# Data Preprocessing
df=pd.read_excel('/content/Buchward Hartwig rxns.xlsx')

#We use the function MolToSmiles from rdkit library to make the useful columns into reaction smiles. Reactions smiles (Simplified molecular-input line-entry system) are the text-based representation of molecules and chemical reactions.
def canonicalize_with_dict(smi, can_smi_dict={}):
    if smi not in can_smi_dict.keys():
        return Chem.MolToSmiles(Chem.MolFromSmiles(smi))
    else:
        return can_smi_dict[smi]
    
def generate_buchwald_hartwig_rxns(df):
    df = df.copy()
    fwd_template = '[F,Cl,Br,I]-[c;H0;D3;+0:1](:[c,n:2]):[c,n:3].[NH2;D1;+0:4]-[c:5]>>[c,n:2]:[c;H0;D3;+0:1](:[c,n:3])-[NH;D2;+0:4]-[c:5]'
    methylaniline = 'Cc1ccc(N)cc1'
    pd_catalyst = Chem.MolToSmiles(Chem.MolFromSmiles('O=S(=O)(O[Pd]1~[NH2]C2C=CC=CC=2C2C=CC=CC1=2)C(F)(F)F'))
    methylaniline_mol = Chem.MolFromSmiles(methylaniline)
    rxn = rdChemReactions.ReactionFromSmarts(fwd_template)
    products = []
    for i, row in df.iterrows():
        reacts = (Chem.MolFromSmiles(row['Aryl halide']), methylaniline_mol)
        rxn_products = rxn.RunReactants(reacts)

        rxn_products_smiles = set([Chem.MolToSmiles(mol[0]) for mol in rxn_products])
        assert len(rxn_products_smiles) == 1
        products.append(list(rxn_products_smiles)[0])
    df['product'] = products
    rxns = []
    can_smiles_dict = {}
    for i, row in df.iterrows():
        aryl_halide = canonicalize_with_dict(row['Aryl halide'], can_smiles_dict)
        can_smiles_dict[row['Aryl halide']] = aryl_halide
        ligand = canonicalize_with_dict(row['Ligand'], can_smiles_dict)
        can_smiles_dict[row['Ligand']] = ligand
        base = canonicalize_with_dict(row['Base'], can_smiles_dict)
        can_smiles_dict[row['Base']] = base
        additive = canonicalize_with_dict(row['Additive'], can_smiles_dict)
        can_smiles_dict[row['Additive']] = additive

        reactants = f"{aryl_halide}.{methylaniline}.{pd_catalyst}.{ligand}.{base}.{additive}"
        rxns.append(f"{reactants}>>{row['product']}")
    return rxns

df['rxn']= generate_buchwald_hartwig_rxns(df)
reactions_df=df[['rxn','Output']]

reactions_df.columns = ['text', 'labels']
reactions_df['labels'] = reactions_df['labels'] / 100

# Encoding the Data

X, mapping = DrfpEncoder.encode(
            reactions_df.text.to_numpy(),
            n_folded_length=2048,
            radius=3,
            rings=True,
            mapping=True,
        )
 X = np.asarray(
            X,
            dtype=np.float32,
        )
 y = reactions_df.labels.to_numpy()


# Importing XGB Regressor 

import pickle
from pathlib import Path
from typing import Tuple
from statistics import stdev
import numpy as np
from xgboost import XGBRegressor
from sklearn.metrics import r2_score

#Splitting the Dataset into X_train,X_test and X_valid

X_df=X[:2767]
X_test=X[2767:]
y_df=y[:2767]
y_test=y[2767:]

from sklearn.model_selection import train_test_split
X_train,X_valid,y_train,y_valid= train_test_split(X_df,y_df,test_size=0.2)

#We used the same value for our hyperparameters for our XGB Regressor model ,i.e., n-estimators=999999, learning rate =0.1, max_depth=12,
min_child_weight=8,colsample_bytree=0.6.


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

#We achieved a R2 score of 0.9097 on our first run for prediction task

r_squared=r2_score(y_test,y_pred)
