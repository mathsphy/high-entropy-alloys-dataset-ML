#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import time

random_state = 1

#%% Define functions


def train_predict(model, X_train, y_train, X_test, y_test,print_metrics=True):
    # record the time
    time_init = time.time()

    # fit the model on the training set
    model.fit(X_train, y_train)
    # predict the test set
    y_pred = pd.DataFrame(model.predict(X_test), index=y_test.index)

    # record the time
    time_elapsed = time.time() - time_init

    # Metrics
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    if print_metrics == True:
        print(f'rmse={rmse:.3f}, mae={mae:.3f}, r2={r2:.3f}, time={time_elapsed:.1f} s')
    metrics = {}
    metrics['rmse'] = rmse
    metrics['mae'] = mae
    metrics['r2'] = r2

    return model, y_pred, metrics

def parity_plot(y_test, y_pred, title=None, ax=None, metrics=None):
    # Figure
    fig, ax = plt.subplots(figsize=(4,4))
    ax.plot(y_test, y_pred,'r.')
    ax.plot(np.linspace(-10,10),np.linspace(-10,10),'k')
    ax.set_xlim([np.min(y_test)-0.1,np.max(y_test)+0.1])
    ax.set_ylim([np.min(y_test)-0.1,np.max(y_test)+0.1])
    ax.set_xlabel('DFT (eV/atom)')
    ax.set_ylabel('ML (eV/atom)')

    # add metrics
    if metrics is not None:
        rmse = metrics['rmse']
        mae = metrics['mae']
        r2 = metrics['r2']
        text = f'rmse={rmse:.3f}, mae={mae:.3f}, r2={r2:.3f}'
        ax.text(np.min(y_test),  np.max(y_test), text, ha="left", va="top", color="b") 
    plt.tight_layout()  
    if title is not None:
        plt.title(title)
    return ax




def perf_vs_size(model, X_pool, y_pool, X_test, y_test, csv_out, 
                 overwrite=False,frac_list=None,n_frac = 15, n_run_factor=1):
    '''
    model: a sklearn model
    X_pool, y_pool: the training pool
    X_test, y_test: the test set
    frac_list: the list of training set size as a fraction of the total training set
    overwrite: if True, overwrite the csv file
    n_frac: the number of training set size to consider
    '''
    
    # if csv_out exists, read it
    if os.path.exists(csv_out) and not overwrite:
        df = pd.read_csv(csv_out,index_col=0)
    # if csv_out does not exist, create it
    else:        
        df = pd.DataFrame(columns=['rmse','mae','r2','rmse_std','mae_std','r2_std'])

    if frac_list is None:
        # the list of training set size as a fraction of the total training set
        # set frac_list to be a list of fractions, equally spaced in log space, from 0.005 to 1
        frac_min = np.log10(100/X_pool.shape[0])
        frac_list = np.logspace(frac_min,0,n_frac)


    for frac in frac_list:
        skip = False
        # skip if frac is close to an existing frac
        for frac_ in df.index:
            if abs(frac - frac_)/frac_ < 0.25:
                skip = True
        if skip:
            continue

        if frac * X_pool.shape[0] < 80:
            continue

        # determine the number of runs based on frac
        if frac < 0.01:
            n_run = 20
        elif frac < 0.05:
            n_run = 10 # 20
        elif frac >= 0.05 and frac < 0.5:
            n_run = 6 #10
        elif frac >= 0.5 and frac < 1:
            n_run = 4 #5
        else:
            n_run = 1

        n_run = max(1, int(n_run * n_run_factor))

        print(f'frac={frac:.3f}, n_run={n_run}')

        metrics_ = {}
        for random_state_ in range(n_run):
            if frac == 1:
                X_train, y_train = X_pool, y_pool
            else:
                X_train, _, y_train, _ = train_test_split(X_pool, y_pool, train_size=frac, 
                                                            random_state=random_state_ * (random_state + 5) )
            # train and predict
            _, _, metrics_[random_state_] = train_predict(model, X_train, y_train, X_test, y_test)
            
        metrics_ = pd.DataFrame(metrics_).transpose()[['rmse','mae','r2']]
        means = metrics_.mean(axis=0)
        std = metrics_.std(axis=0)
        std.index = [f'{col}_std' for col in std.index]

        # add metrics_.mean(axis=1) and metrics_.std(axis=1) to metrics[model_name]
        df.loc[frac] = pd.concat([means,std])
        print(df.loc[frac])
        # save the metrics
        df.sort_index().to_csv(csv_out, index_label='frac')
        
    return df





def plot_metrics_vs_size(metrics, metrics_name,id_train,
                         xlims=None, ylims=None,
                         figsize=(4,4),
                         ax_in=None,
                         fig_ax = None,
                         markers=None
                         ):
    if fig_ax is not None: 
        fig, ax = fig_ax
    elif ax_in is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax_in.get_figure()
        # get second y axis
        ax = ax_in.twinx()


    if markers is None:
        markers = {'RF':'o',
                   'XGB':'s',
                   'alignn50':'^',}


    for model_name in metrics.keys():
        ax.errorbar(metrics[model_name].index, metrics[model_name][metrics_name], 
                    yerr=metrics[model_name][f'{metrics_name}_std'],
                    fmt=f'-{markers[model_name]}',markersize=5,label=model_name, capsize=3)
    
    if xlims is None:
        xlims = [100/len(id_train)*0.9, 1]
    ax.set_xlim(xlims)

    if ylims is None:
        ylims = [0.4, 1]
    ax.set_ylim(ylims)

    

    ax.set_xscale('log')
    ax.set_xlabel('Fraction of the full training set')

    # add the upper x axis for the number of training data
    ax2 = ax.twiny()
    xlims = ax.get_xlim()
    ax2.set_xlim([xlims[0]*len(id_train), xlims[1]*len(id_train)])
    ax2.set_xscale('log')
    ax2.set_xlabel('Training set size')
    ax2.tick_params(axis='x', which='major', pad=0)

    if metrics_name == 'r2':
        ax.set_ylabel('$R^2$')
    else:
        ax.set_ylabel(f'{metrics_name.upper()} (eV/atom)')

    if ax_in is None:
        ax.legend(loc='upper center')
        ax.grid(linewidth=0.1)
    return fig, ax



def eval_ood(model, X_train, y_train, X_test, y_test,title=None, id_small=None, id_large=None):
    model, y_pred, metrics = train_predict(model, X_train, y_train, X_test, y_test)
    # parity_plot(y_test, y_pred, title=None, metrics=metrics)
    # Figure
    fig, ax = plt.subplots(figsize=(4,4))

    index_small = list( set(y_test.index) & set(id_small) )
    index_large = list( set(y_test.index) & set(id_large) )
    if len(index_small) > 0:
        ax.plot(y_test.loc[index_small], y_pred.loc[index_small],'.', label='small')
    if len(index_large) > 0:
        ax.plot(y_test.loc[index_large], y_pred.loc[index_large],'.', label='large')
    ax.plot(np.linspace(-10,10),np.linspace(-10,10),'k')

    # xlim = [np.min(y_test)-0.1,np.max(y_test)+0.1]
    xlim = [-0.75,0.75]
    ylim = xlim

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_xlabel('DFT (eV/atom)')
    ax.set_ylabel('ML prediction (eV/atom)')
    ax.legend()
    rmse, mae, r2 = metrics['rmse'], metrics['mae'], metrics['r2']
    text = f'RMSE={rmse:.3f}, MAE={mae:.3f}, $R^2$={r2:.3f}'
    ax.text(min(xlim)+0.1,min(xlim), text, ha="left", va="bottom", color="k") 
    if title is not None:
        plt.title(title)
    plt.tight_layout()  


def get_mad_std(s):
    # calculate the mean absolute deviation and STD of the series s
    # mean absolute deviation
    mad = (s - s.mean()).abs().mean()
    print(f'mean absolute deviation: {mad:.4f}')
    # standard deviation
    print(f'standard deviation: {s.std():.4f}')
    


# %%
