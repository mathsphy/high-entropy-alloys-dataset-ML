#%%
import os 
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression,Lasso,Ridge
from sklearn.model_selection import GridSearchCV, cross_val_predict
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
from sklearn.model_selection import train_test_split
import random
from func_tmp import train_predict, parity_plot, plot_metrics_vs_size, perf_vs_size,eval_ood,get_mad_std
from distill import return_model

random_state = 1
overwrite = True # whether to overwrite the existing results

#%% Import data
struct='structure'
# dat_type = 'graphs' 
dat_type = 'featurized' 
df = pd.read_pickle(f'data/{struct}_{dat_type}.dat_reduced.pkl')
df = df.dropna()

csv_dir = f'csv/{struct}'
if not os.path.exists(csv_dir):
    os.makedirs(csv_dir)

#%% Define X and y
if dat_type == 'featurized':
    nfeatures = 273
    cols_feat = df.columns[-nfeatures:]
    # col2keep, col2drop = get_col2drop(df[cols_feat], cutoff=0.75,method='spearman')
    # cols_feat = col2keep


    X_all = df[cols_feat]
    # drop features whose variance is zero
    X_all = X_all.loc[:,X_all.var()!=0]
    # # standardize the features and turn it into a df by keeping the index and column names
    # X_std = pd.DataFrame((X_all-X_all.mean())/X_all.std(), index=X_all.index, columns=X_all.columns)
else: 
    X_all = df[f'graphs_{struct}']

y_all = df['Ef_per_atom']


#%% Get training test sets
X_train, X_test, y_train, y_test = train_test_split(X_all, y_all,   test_size=0.2, random_state=random_state)

pipe={}

#%% Hyperparameter tuning

csv_out = csv_dir + '/hypersearch.cv_results_RF.csv'
if not os.path.exists(csv_out) or overwrite:

    pipe['RF'] = Pipeline([
        ('scaler', StandardScaler()),
        ('model', RandomForestRegressor(n_jobs=-1, random_state=1))
    ])

    # Make a function that does the hyperparameter tuning for the model pipe['RF']
    # and returns the best hyperparameters
    hyperparams = {'model__bootstrap': [True, False],
                'model__n_estimators': [50, 100, 150, 200],
                'model__max_features': [0.45, 0.3, 0.2, 0.1],
                'model__max_depth': [5, 10, 15, 20, None],
                }
    
    # Use GridSearchCV to find the best hyperparameters based on MAE and the corresponding scores
    hypersearch = GridSearchCV(pipe['RF'], 
                               hyperparams, 
                               cv=5, 
                               scoring='neg_mean_absolute_error',
                               verbose=3).fit(X_all, y_all)
    best_params, best_scores = hypersearch.best_params_, hypersearch.best_score_

    # Save all the tested hyperparameters, scores, and the associated time to a csv file
    results = pd.DataFrame(hypersearch.cv_results_)
    results.to_csv(csv_out)

    # Save the best hyperparameters and the corresponding score to a csv file
    pd.DataFrame({'best_params': [best_params], 'best_scores': [best_scores]}).to_csv('csv/best_params_RF.csv')

#%%
csv_out = csv_dir + '/hypersearch.cv_results_XGB.csv'
if not os.path.exists(csv_out) or overwrite:
    pipe['XGB'] = Pipeline([
        ('scaler', StandardScaler()),
        ('model', xgb.XGBRegressor(
                        n_estimators=2000,
                        learning_rate=0.1,
                        reg_lambda=0, # L2 regularization
                        reg_alpha=0.1,# L1 regularization
                        num_parallel_tree=1, # set >1 for boosted random forest
                        tree_method='gpu_hist', gpu_id=0))
    ])

    # Make a function that does the hyperparameter tuning for the model pipe['XGB']
    # and returns the best hyperparameters
    hyperparams = {'model__n_estimators': [500, 1000, 2000, 3000],
                'model__learning_rate': [0.1, 0.2, 0.3, 0.4],
                'model__colsample_bytree': [0.3, 0.5, 0.7, 0.9],
                'model__colsample_bylevel': [0.3, 0.5, 0.7, 0.9],
                'model__num_parallel_tree': [4, 6, 8, 10],
                    }
    # Use GridSearchCV to find the best hyperparameters based on MAE and the corresponding scores
    hypersearch = GridSearchCV(pipe['XGB'], hyperparams, cv=5, scoring='neg_mean_absolute_error',verbose=3).fit(X_all, y_all)
    best_params, best_scores = hypersearch.best_params_, hypersearch.best_score_

    # Save all the tested hyperparameters, scores, and the associated time to a csv file
    results = pd.DataFrame(hypersearch.cv_results_)
    results.to_csv('csv/hypersearch.cv_results_XGB.csv')

    # Save the best hyperparameters and the corresponding score to a csv file
    pd.DataFrame({'best_params': [best_params], 'best_scores': [best_scores]}).to_csv('csv/best_params_XGB.csv')


