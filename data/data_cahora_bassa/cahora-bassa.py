#!/usr/bin/python
# encoding: utf-8
#%%
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import scipy as sp
from sklearn.decomposition import FastICA, PCA

#%%
csv='cahora-bassa.csv'

X=pd.read_csv(csv,encoding='utf-8', sep=';')

X['Data'] = pd.to_datetime(X.index.values, origin=pd.Timestamp(X['Data'][0]), unit='D')
X.set_index('Data')
X.index = X['Data']
#X.index=[i for i in X.index]X.col

X['Ano']=[a.year for a in X.index]
X['Mes']=[a.month for a in X.index]
X['Dia']=[a.day for a in X.index]
X['Dia do Ano']=[a.dayofyear for a in X.index]
X['Dia Semana']=[a.day_name() for a in X.index]

X['Balanco Volumetrico (m3)'] = (X['Caudal Efluente (m3/s)'] - X['Caudal Afluente (m3/s)'])*60*60*24

cols_plot = X.columns.drop(['Data', 'Mes', 'Dia', 'Dia Semana', 'Ano',
                            'Dia do Ano'])
#%%
X=X.dropna()
#%%
c='Cota (m)'

for d in ['Caudal Afluente (m3/s)', 
          'Caudal Efluente (m3/s)',
          'Balanco Volumetrico (m3)',]:
    
    plt.figure()
    S = X[[c,d]];      
    s=MinMaxScaler(); 
    S=pd.DataFrame(s.fit_transform(S), columns=S.columns); 
    
    S.plot(figsize=(16,4))
    plt.show()
    
#%%
plt.close('all')
kwargs={'linewidth':0.5, 'marker':'.', 'alpha':0.5, 'linestyle':'-.',}
for c in cols_plot:
    plt.figure(figsize=(16,4))
    axes=X[c].plot(linewidth=0.5, marker='.', alpha=0.5, linestyle='-.', 
                              figsize=(27, 9), subplots=True, color='red')
    for ax in axes:
        ax.set_ylabel(c)

#%%
sns.pairplot(X, vars=cols_plot)
#%%
cols_daily=['Balanco Volumetrico (m3)']
#cols_daily=['Cota (m)']

X.groupby('Dia do Ano').agg(np.mean)[cols_daily].plot(linewidth=1.5, marker='.', 
                        linestyle='-.', color='red')
plt.axhline(0, color='k',linestyle='-.', alpha=0.5)
plt.show()

#%%

for x_data in ['Dia', 'Dia do Ano']:
    for c in cols_daily:    
        plt.figure()
        sns.catplot(data=X, x=x_data, y=c, kind='point',kwargs=kwargs)
        plt.axhline(0, color='red')
        for ax in axes:
            ax.set_ylabel(c)
        
        plt.show()
 
#%%

# https://scikit-learn.org/stable/auto_examples/decomposition/plot_ica_blind_source_separation.html

X_ = X[cols_plot].dropna().values

X_ = X[cols_plot].dropna().values

n_components =  len(cols_plot)
# Compute ICA
ica = FastICA(n_components=n_components)
S_ = ica.fit_transform(X_)  # Reconstruct signals
A_ = ica.mixing_  # Get estimated mixing matrix

# For comparison, compute PCA
pca = PCA(n_components=n_components)
H_ = pca.fit_transform(X_) 
#%%
plt.figure()

models = [X_, S_, H_]
names = ['Observations (mixed signal)',
         'ICA recovered signals',
         'PCA recovered signals']
colors = ['red', 'steelblue', 'orange']

for ii, (model, name) in enumerate(zip(models, names), 1):
    plt.subplot(len(names), 1, ii)
    plt.title(name)
    for sig, color in zip(model.T, colors):
        plt.plot(sig, color=color)

plt.subplots_adjust(0.09, 0.04, 0.94, 0.94, 0.26, 0.46)
plt.show()
#%%
S=pd.DataFrame(S_); S.index=X.index
S.columns=['ICA '+str(i) for i in S.columns]
for c in S.columns:
    plt.figure()
    axes=S[c].plot(linewidth=0.5, marker='.', alpha=0.5, linestyle='-.', 
                              figsize=(27, 9), subplots=True, color='red')
    #plt.plot(S[c].values,'-.')
    for ax in axes:
        ax.set_ylabel(c)

    plt.show()


#%%
for i in range(n_components):
    for j in range(n_components):
        r_=sp.stats.pearsonr(X_[:,i],S_[:,j])[0]
        if(abs(r_)>0.7):  
            plt.figure()
            plt.plot(X_[:,i],S_[:,j], '.')
            plt.xlabel(str(cols_plot[i]))
            plt.ylabel('ICA component '+str(j))
            plt.title(str(r_))
            plt.show()
            plt.show()

#%%
