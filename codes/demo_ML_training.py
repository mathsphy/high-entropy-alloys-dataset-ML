#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: kangming

A demo to show the XGBoost training on the formation energy dataset 
"""
#%%
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import cross_val_predict
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
import xgboost as xgb


model= Pipeline([
    ('scaler', StandardScaler()), # scaling does not actually matter for tree methods
    ('model', xgb.XGBRegressor(
                    n_estimators=500,
                    learning_rate=0.4,
                    reg_lambda=0.01,reg_alpha=0.1,
                    colsample_bytree=0.5,colsample_bylevel=0.7,
                    num_parallel_tree=6,
                    # tree_method='gpu_hist', gpu_id=0
                    tree_method = "hist", device = "cuda",                    
                    )
                    )
])

#%%
struct = 'structure'

dat_type = 'featurized' 
# df = pd.read_pickle(f'data/{struct}_{dat_type}.dat_reduced.pkl')
df = pd.read_csv(f'data/{struct}_{dat_type}.dat_all.csv',index_col=0)

#%%
if dat_type == 'featurized':
    nfeatures = 273
    cols_feat = df.columns[-nfeatures:]

    X_all = df[cols_feat]
    # drop features whose variance is zero
    X_all = X_all.loc[:,X_all.var()!=0]
else: 
    X_all = df[f'graphs_{struct}']

y_all = df['Ef_per_atom']

#%%
# Get the 5-fold cross-validation estimates for the whole dataset
cv = 5
y_pred = cross_val_predict(model, X_all, y_all, cv=cv)

#%%
df_y = pd.DataFrame({'Ef_true':y_all, 'Ef_pred':y_pred}, index=y_all.index)
cols2add = ['formula','lattice','NIONS']
df_y = pd.concat([df_y,df[cols2add]], axis=1)

#%%
# get the mae of Ef_true and Ef_pred
mad = np.mean(np.abs(df_y['Ef_true'] - df_y['Ef_true'].mean()))
mae = mean_absolute_error(df_y['Ef_true'], df_y['Ef_pred'])
print(f'MAD: {mad:.3f}')
print(f'MAE: {mae:.3f}')
# get the r2 score
r2 = r2_score(df_y['Ef_true'], df_y['Ef_pred'])
print(f'R2: {r2:.3f}')

#%% parity plot
fig, ax = plt.subplots(figsize=(5,5))
ax.scatter(df_y['Ef_true'], df_y['Ef_pred'], s=5)
lims = [-0.6, 0.6]
#diag line
ax.plot(lims,lims, 'k--', lw=1)
# set limits
ax.set_xlim(lims)
ax.set_ylim(lims)
ax.set_xlabel('DFT formation energy (eV/atom)')
ax.set_ylabel('Predicted formation energy (eV/atom)')
# add scores to fig
ax.text(0.05, 0.9, f'MAE: {mae:.3f} eV/atom', transform=ax.transAxes)
ax.text(0.05, 0.85, f'R2: {r2:.3f}', transform=ax.transAxes)

# %%
