#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""



"""

#%%
import os 
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
from sklearn.model_selection import train_test_split
from func_tmp import plot_metrics_vs_size, perf_vs_size,eval_ood,get_mad_std


random_state = 1
overwrite = False # whether to overwrite the existing results


#Figure setting
figsize = (3.6,3.6)

#%% Define the models

pipe={}

# more expensive models
pipe['RF'] = Pipeline([
    ('scaler', StandardScaler()),
    ('model', RandomForestRegressor(n_estimators=100, 
                                    bootstrap=False,
                                    max_features = 1/3, 
                                    n_jobs=-1, random_state=random_state))
])

pipe['XGB'] = Pipeline([
    ('scaler', StandardScaler()),
    ('model', xgb.XGBRegressor(
                    n_estimators=500,
                    learning_rate=0.4,
                    reg_lambda=0.01,reg_alpha=0.1,
                    colsample_bytree=0.5,colsample_bylevel=0.7,
                    num_parallel_tree=6,
                    tree_method='gpu_hist', gpu_id=0)
                    )
])

epochs=50
modelname = f'alignn{epochs}'
pipe[modelname] = None #return_model(modelname,random_state,alignn_epoch=epochs)


#%% Import data
for struct in ['structure_ini']:
    # dat_type = 'graphs' 
    dat_type = 'featurized' 
    df = pd.read_pickle(f'data/{struct}_{dat_type}.dat_reduced.pkl')
    df = df.dropna()

    csv_dir = f'csv/{struct}'
    if not os.path.exists(csv_dir):
        os.makedirs(csv_dir)



    #%% Define X and y

    if dat_type == 'featurized':
        # drop features with high correlation
        # from myfunc import get_col2drop 
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

    get_mad_std(y_all)


    #%% define index

    # according to lattice
    id_bcc = df[df['lattice']=='bcc'].index.tolist()
    id_fcc = df[df['lattice']=='fcc'].index.tolist()

    print('bcc:',len(id_bcc))
    get_mad_std(y_all[id_bcc])
    print('fcc:',len(id_fcc))
    get_mad_std(y_all[id_fcc])

    # according to nelements
    id_nele = {}
    for nele in [2,3,4,5,6,7]:
        id_nele[nele] = df[df['nelements']==nele].index.tolist()

    id_loworder = df[df['nelements']<=3].index.tolist()
    id_highorder = df[df['nelements']>3].index.tolist()


    # according to NIONS
    id_all = df.index.tolist()
    id_small = df[df['NIONS']<=8].index.tolist()
    id_large = df[df['NIONS']>8].index.tolist()
    print('small:',len(id_small))   
    get_mad_std(y_all[id_small])
    print('large:',len(id_large))
    get_mad_std(y_all[id_large])

    #%%
    # calculate the composition based on reduced_formula
    from pymatgen.core.composition import Composition
    composition = df['reduced_formula'].apply(lambda x: Composition(x))

    def get_el_frac(x):
        el_frac_list = list(x.get_el_amt_dict().values())
        tot = sum(el_frac_list)
        el_frac = np.array([i/tot for i in el_frac_list])
        return el_frac

    el_frac = composition.apply(get_el_frac)

    # For each composition, calculate the max fractional concentration and min fractional concentration
    df['max_c'] = el_frac.apply(max)
    df['min_c'] = el_frac.apply(min)
    df['diff_c'] = df['max_c'] - df['min_c']
    df['std_c'] = el_frac.apply(np.std)


    #%%

    '''
    First, evaluate the interpolation performance 
    '''

    frac_list_alignn = [1,0.25,0.1,0.05,0.01]

    model_list = [f'alignn{epochs}','XGB','RF']
    # csv_dir_ = f'csv/{struct}'
    csv_dir_ = f'csv/structure_ini'


    for scope in ['all']: # ,'large','small'
        if scope == 'all':
            index = id_all
        elif scope == 'small':
            index = id_small
        elif scope == 'large':
            index = id_large

        test_size = 0.2

        X_pool, X_test, y_pool, y_test = train_test_split(
            X_all.loc[index], y_all.loc[index],
            test_size=test_size,
            random_state=random_state,
            )
            
        # get the performance vs. training set size
        metrics = {}
        for model_name in pipe.keys():
            csv_out = f'{csv_dir_}/size_effect_rand_split_{scope}_{model_name}.csv'
            # skip if the csv file already exists
            if os.path.exists(csv_out) and not overwrite:
                # read the csv file
                metrics[model_name] = pd.read_csv(csv_out, index_col=0)
                # MAD of the test set 
                mad = y_test.mad()
                metrics[model_name]['mae/mad'] = metrics[model_name]['mae']/mad
                metrics[model_name]['mae/mad_std'] = metrics[model_name]['mae_std']/mad
                continue

            if 'alignn' in model_name:
                frac_list = frac_list_alignn
                n_run_factor = 0.5
            else:
                frac_list = None
                n_run_factor = 1

            metrics[model_name] = perf_vs_size(
                pipe[model_name], 
                X_pool, y_pool, 
                X_test, y_test, 
                csv_out, 
                overwrite=overwrite,
                frac_list=frac_list,
                n_run_factor=n_run_factor,
                )
            
            # print the performance of full model
            mae = metrics[model_name].iloc[-1]['mae']
            r2 = metrics[model_name].iloc[-1]['r2']
            print(f'{scope} {model_name} {mae} {r2} ')


        fig, axs = plt.subplots(figsize=(3.25*2,2.5), ncols=2,
                                gridspec_kw={'wspace':0.325}
                                )
        ax = axs[0]
        fig, ax = plot_metrics_vs_size(metrics, 'mae/mad', X_pool.index,
                                       figsize=(3.5,3),
                                    ylims=[0.05,0.45],
                                    xlims=[5e-3,1],
                                    fig_ax=(fig,ax),
                                    )
        ax.set_xlabel('Fraction of training pool')
        ax.set_ylabel(f'MAE/MAD')
        ax.legend(
            # set legend label
            ['RF','XGB','ALIGNN']
        )
        ax.text(-0.15,1.05,'(a)',transform=ax.transAxes,fontsize=12)
        # add grid
        # ax.grid(which='both', axis='both')
        
        

        # ax.set_yticks(np.arange(0.05,0.5,0.05))

        ax = axs[1]
        fig, ax = plot_metrics_vs_size(metrics, 'r2', X_pool.index,
                                    ylims=[0.75,1],
                                    xlims=[5e-3,1],
                                    # ax_in=ax,
                                    fig_ax=(fig,ax),
                                    )
        ax.legend(
            # set legend label
            ['RF','XGB','ALIGNN']
        )

        ax.set_xlabel('Fraction of training pool')
        ax.set_ylabel(r'$R^2$')
        ax.text(-0.15,1.05,'(b)',transform=ax.transAxes,fontsize=12)

        fig.savefig(f'figs/{struct}_size_effect_rand_split_{scope}.pdf',bbox_inches='tight')


    #%%


    frac_list_alignn = [1,0.25,0.1,0.05,0.01]

    fig, axs = plt.subplots(figsize=(2.75*2,2.5), ncols=2,
                            gridspec_kw={'wspace':0.035}
                            )
    ax = axs[0]

    # csv_dir_ = f'csv/{struct}'
    scope = 'all'
    for i, csv_dir_ in enumerate(['csv/structure','csv/structure_ini']):
        metrics = {}
        for model_name in pipe.keys():
            if scope == 'all':
                csv_out = f'{csv_dir_}/size_effect_rand_split_{scope}_{model_name}.csv'
            else:
                csv_out = f'{csv_dir_}/size_effect_{scope}_{model_name}.csv'
            # skip if the csv file already exists
            if os.path.exists(csv_out) and not overwrite:
                # read the csv file
                metrics[model_name] = pd.read_csv(csv_out, index_col=0)
                # MAD of the test set 
                mad = y_test.mad()
                metrics[model_name]['mae/mad'] = metrics[model_name]['mae']/mad
                metrics[model_name]['mae/mad_std'] = metrics[model_name]['mae_std']/mad

        ax = axs[i]
        fig, ax = plot_metrics_vs_size(metrics, 'mae/mad', X_pool.index,
                                        figsize=(3.5,3),
                                    ylims=[0.05,0.425],
                                    xlims=[5e-3,1],
                                    fig_ax=(fig,ax),
                                    )
        ax.set_xlabel('Fraction of training pool')
        ax.legend(
            # set legend label
            ['RF','XGB','ALIGNN'],loc='upper right'
        )
        # ax.text(-0.15,1.05,f'({chr(ord("a")+i)})',transform=ax.transAxes,fontsize=12)
        if i == 0:
            label = 'Relaxed'
            ax.set_ylabel(f'MAE/MAD')

        else:
            label = 'Unrelaxed'
            # disable y label and y ticklabels
            ax.set_ylabel('')
            ax.set_yticklabels([])

        ax.text(0.05,0.05,f'({chr(ord("a")+i)}) {label}',transform=ax.transAxes,fontsize=12)
            
        ax.grid()
    fig.savefig('figs/size_effect_rand_split_all.pdf',bbox_inches='tight')







    #%%
    '''
    Next, evaluate the extrapolation performance 

    '''
    for scope in [
        'small2large',
        'low2high',
        'large2small',
        'high2low', # XGB is a bit strange

        # 'bcc2fcc',
        # 'fcc2bcc', # This is strange; need to figure out later
        ]:
        if scope == 'small2large':
            id_train = id_small
            id_test = id_large
        elif scope == 'large2small':
            id_train = id_large
            id_test = id_small
        elif scope == 'low2high':
            id_train = id_loworder
            id_test = id_highorder
        elif scope == 'high2low':
            id_train = id_highorder
            id_test = id_loworder
        elif scope == 'bcc2fcc':
            id_train = id_bcc
            id_test = id_fcc
        elif scope == 'fcc2bcc':
            id_train = id_fcc
            id_test = id_bcc

    
        metrics = {}

        # define the training and test sets
        X_pool, y_pool = X_all.loc[id_train], y_all.loc[id_train]
        X_test, y_test = X_all.loc[id_test], y_all.loc[id_test]

        mad = y_test.mad()
        print(f'{scope} MAD of the test set: {mad}')

        # get the performance vs. training set size
        for model_name in model_list:
            csv_out = f'{csv_dir}/size_effect_{scope}_{model_name}.csv'
            # skip if the csv file already exists
            # if os.path.exists(csv_out) and not overwrite:
            #     # read the csv file
            #     metrics[model_name] = pd.read_csv(csv_out, index_col=0)
            #     # MAD of the test set 
            #     mad = y_test.mad()
            #     metrics[model_name]['mae/mad'] = metrics[model_name]['mae']/mad
            #     metrics[model_name]['mae/mad_std'] = metrics[model_name]['mae_std']/mad
            #     continue

            if 'alignn' in model_name:
                frac_list = frac_list_alignn
                n_run_factor = 0.5
            else:
                frac_list = None
                n_run_factor = 1

            metrics[model_name] = perf_vs_size(
                pipe[model_name], 
                X_pool, y_pool, 
                X_test, y_test, 
                csv_out, 
                overwrite=overwrite,
                frac_list=frac_list,
                # frac_list = [0.5],
                n_run_factor=n_run_factor,
                )
            # print the performance of full model
            mae = metrics[model_name].iloc[-1]['mae']
            r2 = metrics[model_name].iloc[-1]['r2']
            print(f'{scope} {model_name} {mae} {r2} ')

        # plot performance vs. training set size
        # fig, ax = plot_metrics_vs_size(metrics, 'mae', X_pool.index,
        #                                 ylims=[0.0,0.07],
        #                                 )

        # fig, ax = plot_metrics_vs_size(metrics, 'r2', X_pool.index,
        #                             ylims=[0.5,1],
        #                             )
        # # add legend title
        # ax.legend(title=scope, loc='lower right')
        # ax.grid(which='both', axis='both', ls=':')










#%%

# show the accumulated count based on diff_c
# df['diff_c'].hist(cumulative=True, density=1, bins=1000)

for model_name in pipe.keys():
    model = pipe[model_name]

    for max_diff_c in [0.]:# ,0.15,0.2,0.3,0.4,0.5, 0.6
        id_train = df[df['diff_c']<=max_diff_c].index.tolist()
        id_test = df[df['diff_c']>max_diff_c].index.tolist()
        X_train, y_train = X_all.loc[id_train], y_all.loc[id_train]
        X_test, y_test = X_all.loc[id_test], y_all.loc[id_test]      
        # # MAD
        # mad = y_test.mad()
        # print(f'{model_name} {max_diff_c} {mad}')
        model.fit(X_train, y_train)
        y_pred = pd.Series(model.predict(X_test), index=y_test.index)

        y_err = (y_pred-y_test).abs()

        id_test_large = df[(df['diff_c']>max_diff_c) & (df['NIONS']>8)].index.tolist()
        id_test_small = df[(df['diff_c']>max_diff_c) & (df['NIONS']<=8)].index.tolist()

        # get mean absolute error
        mae = y_err.mean()
        # mae_large = y_err.loc[id_test_large].mean()
        # mae_small = y_err.loc[id_test_small].mean()
        # get r2
        r2 = r2_score(y_test, y_pred)
        # r2_large = r2_score(y_test.loc[id_test_large], y_pred.loc[id_test_large])
        # r2_small = r2_score(y_test.loc[id_test_small], y_pred.loc[id_test_small])

        print(f'{model_name} {max_diff_c} {mae}')
        # print(f'{model_name} {max_diff_c} {r2} {r2_large} {r2_small}')



