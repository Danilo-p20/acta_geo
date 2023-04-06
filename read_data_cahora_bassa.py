#!/usr/bin/python
# -*- coding: utf-8 -*
import numpy as np
import scipy as sp
import pylab as pl
import pandas as pd
import seaborn as sns
import glob

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import (GridSearchCV, KFold, cross_val_predict, 
                                     TimeSeriesSplit, cross_val_score, 
                                     LeaveOneOut, KFold, StratifiedKFold,
                                     cross_val_predict,train_test_split)
#-------------------------------------------------------------------------------

#def split_sequences_multivariate_days_ahead(sequences, n_steps_in, n_steps_out):
#    # split a multivariate sequence into samples
#	X, y = list(), list()
#	for i in range(len(sequences)):
#		# find the end of this pattern
#		end_ix = i + n_steps_in
#		out_end_ix = end_ix + n_steps_out-1
#		# check if we are beyond the dataset
#		if out_end_ix+1 > len(sequences):
#			break
#		# gather input and output parts of the pattern
#		#seq_x, seq_y = sequences[i:end_ix, :-1], sequences[end_ix-1:out_end_ix, -1]
#		seq_x, seq_y = sequences[i:end_ix, :-1], sequences[end_ix:out_end_ix+1, -1]
#		X.append(seq_x)
#		y.append(seq_y)
#	return np.array(X), np.array(y)    
#-------------------------------------------------------------------------------
def split_sequences_multivariate(sequences, n_steps_in, n_steps_out):
    # split a multivariate sequence into samples
	X, y = list(), list()
	for i in range(len(sequences)):
		# find the end of this pattern
		end_ix = i + n_steps_in
		out_end_ix = end_ix + n_steps_out-1
		# check if we are beyond the dataset
		if out_end_ix+1 > len(sequences):
			break
		# gather input and output parts of the pattern
		#seq_x, seq_y = sequences[i:end_ix, :-1], sequences[end_ix-1:out_end_ix, -1]
		seq_x, seq_y = sequences[i:end_ix, :], sequences[end_ix:out_end_ix+1, -1]
		X.append(seq_x)
		y.append(seq_y)
	return np.array(X), np.array(y)    
#-------------------------------------------------------------------------------
def read_cahora_bassa(
            filename='./data/data_cahora_bassa/cahora-bassa.csv',
            look_back=21,  look_forward=7, 
            expand_features=False,
        ):
    #%%
    #look_back=7; look_forward=1; kind='ml'; unit='day'; roll=False; window=7; scale=False; expand_features=False;
    filename='./data/data_cahora_bassa/cahora-bassa-raw.csv'
    df= pd.read_csv(filename,  delimiter=';', decimal='.' )
    df.index=df['Data']
    df.drop('Data', axis=1, inplace=True)
    aux = [dict(zip(('month','day','year'),x.split('/'))) for x in df.index.values]
    dt=pd.DataFrame(aux)
    idx=pd.to_datetime(dt, yearfirst=True)    
    df.index =idx
    for c in df.columns:
       #print(c)
       df[c].interpolate(method='linear', inplace=True)
    
    df.sort_index(inplace=True)
    idx = df.index < '2015-12-31'
    idx = df.index < '2011-12-31'
    df=df[idx]
    df.drop(['Cota (m)', 'Caudal Efluente (m3/s)','Volume Evaporado Mm3',], axis=1, inplace=True)
    df.columns=['Q','R','E','H']
    target='Q'
    df[target]/=1e3
    aux=df[target]; df.drop(target,axis=1, inplace=True); df[target]=aux
    
    n_steps_in, n_steps_out = look_back, look_forward
    X, y = split_sequences_multivariate(df.values, n_steps_in, n_steps_out)
    y = y[:,-1].reshape(-1,1); n_steps_out = y.shape[1] # only the last day
    dates=df.index[look_forward+look_back-1::]
    X = np.array([list(X[i].T.ravel()) for i in range(len(X))])
 
    X = pd.DataFrame(X, index=dates)
    y = pd.DataFrame(y, index=dates)
    
    var_names = ['x'+str(i+1) for i in range(X.shape[1])]
    X.columns=var_names
    if expand_features:
        for i in range(len(var_names)):
            a = var_names[i]
            #s='log('+str(a)+')'
            #X[s] = np.log10(X[a])
            s='sin('+str(a)+')'
            X[s] = np.sin(X[a])
            s='sig('+str(a)+')'
            X[s] = 1/(1+np.exp(X[a]))
            for j in range(i,len(var_names)):
                b = var_names[j]
                if a!=b:
                    s=str(a)+str(' * ')+str(b)
                    X[s] = X[a]*X[b]
                    s=str(a)+str(' ** ')+str(b)
                    X[s] = X[a]**X[b]
                    s=str(b)+str(' ** ')+str(a)
                    X[s] = X[b]**X[a]
                    ##s=str(a)+str(' / ')+str(b)
                    ##X[s] = X[a]/X[b]
                    ##s=str(b)+str(' / ')+str(a)
                    ##X[s] = X[b]/X[a]
                    ##s=str(b)+str(' / ')+str(a)
                    ##X[s] = X[b]/X[a]
                
    variable_names = X.columns
        
    train_date = X.index <= '2009-06-30'
    test_date  = X.index >  '2009-06-30'
    
    X_train, X_test, y_train, y_test = X[train_date], X[test_date], y[train_date], y[test_date] 
    
    y_train.colums=['Train']; y_test.colums=['Test']
    ax=y_train.plot(); y_test.plot(ax=ax, figsize=(10,5),)
    #%%
    k=0
    col_range=[]
    col_dict={}
    for j in range(look_back):
        for i in df.columns:
            k+=1
            j0=look_back-j-1
            j1='t-'+str(j0) if j0!=0 else 't' 
            #cr='x'+str(k)+'$'+i+'{'+j1+'}$'
            cr={'x'+str(k):'$'+i+'_{'+j1+'}$'}
            print(cr)
            col_dict['x'+str(k)]='$'+i+'_{'+j1+'}$'
            col_range.append(cr)
    #%%
    pl.rc('text', usetex=True); pl.rc('font', family='serif',  serif='Times')
    dataframe=X_train.copy()
    target='$Q_{t+1}$'
    dataframe.columns = [col_dict[i] for i in dataframe.columns]
    dataframe[target]=y_train      
    pl.figure(figsize=(1, 10))
    heatmap = sns.heatmap(dataframe.corr()[[target]].sort_values(by=target, ascending=False), vmin=-1, vmax=1, annot=True, cmap='BrBG')
    heatmap.set_yticklabels(labels=heatmap.get_yticklabels(), rotation=0)
    pl.savefig('target_corr.png', dpi=300,  bbox_inches='tight')
    
    n_samples, n_features = X_train.shape
    #variable_names=['X'+str(i) for i in range(len(X_train.columns))]
    if look_back==1:
        variable_names=df.columns
    
    data_description = ['x'+str(i+1) for i in range(X_train.shape[1])]
    sn =filename.split('-')[0].split('/')[-1]
    dataset=  {
      'task':'forecast',
      'name':'CB '+str(look_back)+' '+str(look_forward),
      'feature_names':variable_names,
      'target_names':[target],
      'n_samples':n_samples, 'n_features':n_features,
      'X_train':X_train.values,
      'X_test':X_test.values,
      'y_train':np.array([y_train.values.ravel()]),
      'y_test':np.array([y_test.values.ravel()]),      
      'true_labels':None,
      'predicted_labels':None,
      'descriptions':data_description,
      'items':None,
      'reference':"",      
      'normalize': 'MinMax',
      }   
    #%%
    # #
    # #filename='./data/data_cahora_bassa/cahora-bassa-imputed.csv'
    # #df= pd.read_csv(filename,  delimiter=';', decimal=',' )
    # #df.index = pd.DatetimeIndex(data=df['Data'].values, dayfirst=True)
    # #
    # df['year']=[a.year for a in df.index]
    # df['month']=[a.month for a in df.index]
    # #df['week']=[a.week for a in df.index]
    # #
    # if unit=='day':
    #     df['day']=[a.day for a in df.index]
    #     #df = df.groupby(['day', 'month', 'year']).agg(np.mean)
    #     #dt = pd.DataFrame([ {'year':int(y), 'month':int(m), 'day':d} for (d,m,y) in df.index.values] )        
    #     dt=df.index
    #     df.index=pd.to_datetime(dt, yearfirst=True)
    # elif unit=='month':   
    #     df_std = df.groupby(['month', 'year']).agg(np.std)
    #     df = df.groupby(['month', 'year']).agg(np.mean)
    #     dt = pd.DataFrame([ {'year':int(y), 'month':int(m), 'day':15} for (m,y) in df.index.values] )
    #     df.index=pd.to_datetime(dt, yearfirst=True)
    # else:
    #     sys.exit('Time slot is not defined: day or month')
        
    # # #>>df['Time']= [a.year+a.dayofyear/366 for a in df.index]   
    # df.sort_index(inplace=True)
    # #
    # idx = df.index < '2013-12-31'
    # idx = df.index < '2015-12-31'
    # idx = df.index < '2018-12-31'
    # idx = df.index < '2011-12-31'

    # df=df[idx]
    
    # c='Caudal Afluente (m3/s)'
    # df[c]/=1e3
    # out_seq=df[c] 
    # #aux=df.rolling(window=5, min_periods=1, win_type=None).sum()
    # #df['Prec. Acum. (mm)']=aux['Precipitacao (mm)']
        
    # #df['smooth']=smooth(df[c].values, window_len=10)
    # #if unit=='day':
    # #    df[c]=smooth(df[c].values, window_len=10)
        
    # clstdrp=[]#['Precipitacao (mm)', 'Evaporacao (mm)','Humidade Relativa (%)',]
    # if unit=='day':
    #     cols_to_drop = clstdrp+['Cota (m)', 'Caudal Efluente (m3/s)', 'Volume Evaporado Mm3',  'year', 'month', 'day']
    # elif unit=='month':   
    #     cols_to_drop = clstdrp+['Cota (m)', 'Caudal Efluente (m3/s)', 'Volume Evaporado Mm3', ]
    # else:
    #     sys.exit('Time slot is not defined: day or month')
    
    # df.drop(cols_to_drop, axis=1, inplace=True) 
    
    # #df.drop(df.columns, axis=1, inplace=True); df[c]=out_seq
    # df.columns=['Q', 'R', 'E', 'H', ]

    # feature_names=df.columns    
    # df['Target']=out_seq 
    # target_names=[c]
    # dates = df.index

    # # if unit=='month':
    # #     pl.plot(df.index, df[target_names].values)    
    # #     pl.fill_between(df.index, 
    # #                     (df[target_names].values - df_std[target_names].values).ravel(), 
    # #                     (df[target_names].values + df_std[target_names].values).ravel(), 
    # #              alpha=0.2, color='k')

    # if scale:       
    #     scaler=MinMaxScaler()
    #     scaled=scaler.fit_transform(df)
    #     scaled=pd.DataFrame(data=scaled, columns=df.columns, index=df.index)
    # else:
    #     scaled=df
    
    # #ds = df.values
    # ds = scaled.values
    # n_steps_in, n_steps_out = look_back, look_forward
    # X, y = split_sequences_multivariate_days_ahead(ds, n_steps_in, n_steps_out)
    # y = y[:,-1].reshape(-1,1); n_steps_out = y.shape[1] # inly the last day
    # dates=df.index[look_forward+look_back-1::]
    
    # if roll:
    #      y_roll=pd.DataFrame(y).rolling(window=window, min_periods=1, win_type=None).mean().values
    #      y=y_roll
    
    # if ecological:
    #      y_eco=pd.DataFrame(y).rolling(window=7, min_periods=1, win_type=None).min().values
    #      y=y_eco
    
    # train_set_date = '2014-12-31' 
    # train_set_date = '2009-06-30' 
    # train_size, test_size = sum(dates <= train_set_date), sum(dates > train_set_date) 
    # X_train, X_test = X[0:train_size], X[train_size:len(dates)]
    # y_train, y_test = y[0:train_size], y[train_size:len(dates)]
    # #y_std_train, y_std_test = df_std[target].values[0:train_size], df_std[target].values[train_size:len(dates)]
        
    # pl.figure(figsize=(16,4)); 
    # pl.plot([a for a in y_train]+[None for a in y_test]);
    # pl.plot([None for a in y_train]+[a for a in y_test]); 
    # pl.show()

    # mnth=[a.month for a in dates]
    # n_samples, _, n_features = X_train.shape
    # if kind=='ml':        
    #     X_train = np.array([list(X_train[i].T.ravel()) for i in range(len(X_train))])
    #     X_test  = np.array([list(X_test[i].T.ravel()) for i in range(len(X_test))])
    #     y_train, y_test = y_train.T, y_test.T 
    #     n_features = n_features+look_back
    #     #X_train=np.c_[X_train, mnth[:train_size]]
    #     #X_test=np.c_[X_test, mnth[train_size:]]
    #     n_samples, n_features = X_train.shape
    #     feature_names=np.array([ str(i)+'_{-'+str(j)+'}' for i in feature_names for j in range(look_back)])
    
    # data_description = np.array(['var_'+str(i) for i in range(n_features)])
    # dataset=  {
    #   'task'            : 'regression',
    #   'name'            : 'Cahora Bassa '+str(look_back)+' '+unit+'s back '+str(look_forward)+' '+unit+'s ahead',
    #   'feature_names'   : feature_names,
    #   'target_names'    : target_names,
    #   'n_samples'       : n_samples, 
    #   'n_features'      : n_features,
    #   'X_train'         : X_train,
    #   'X_test'          : X_test,
    #   'y_train'         : y_train,
    #   'y_test'          : y_test,      
    #   'targets'         : target_names,
    #   'true_labels'     : None,
    #   'predicted_labels': None,
    #   'descriptions'    : data_description,
    #   'items'           : None,
    #   'reference'       : "Alfeu",      
    #   'normalize'       : 'MinMax',
    #   'date_range'      : {'train': dates[0:train_size], 'test': dates[train_size:]},
    #   }
    #%%
    return dataset
#------------------------------------------------------------------------------
#
#
#
if __name__ == "__main__":
    datasets = [                 
                #read_data_xingu_sequence(look_back=1, look_forward=1, kind='ml' , roll=True, window=7),
                #read_data_alameer_sequence(look_back=1, look_forward=1, kind='ml', roll=False,),
                #read_data_naula(model=0, kind='lstm'),
                #read_data_vietnam(station='centralhighlands', period='daily',),
                #read_data_vietnam(station='centralhighlands', period='hourly',)
                #read_data_pakistan(station='islamabad',),
                read_cahora_bassa(look_back=7, look_forward=1,)
            ]
    for D in datasets:
        print('='*80+'\n'+D['name']+'\n'+'='*80)
        print(D['reference'])
        print(D['y_train'])
        print('\n')

    x=np.c_[range(0,10),range(10,20),range(20,30),range(30,40),]
    z=x.sum(axis=1)
    Z=pd.DataFrame(x, columns=['a','b','c','d']); 
    df=Z.copy(); df['target']=z
    X,y=aux=split_sequences_multivariate(df.values, 3, 1)
    
#%%-----------------------------------------------------------------------------
    x=np.c_[range(10),range(-10,0),range(100,110)]
    X,y=split_sequences_multivariate(x, 3, 1)
#%%-----------------------------------------------------------------------------

# x_1 = P_{-1}, x2 =E_{-1}, x3 =H_{-1}, x4 =Q_{-1}
# x_5 = P_{-2}, x6 =E_{-2}, x7 =H_{-2}, x8 =Q_{-2}
# x_9 = P_{-3}, x10=E_{-3}, x11=H_{-3}, x12=Q_{-3}
# x_13= P_{-4}, x14=E_{-4}, x15=H_{-4}, x16=Q_{-4}
# x_17= P_{-5}, x18=E_{-5}, x19=H_{-5}, x20=Q_{-5}
# x_21= P_{-6}, x22=E_{-6}, x23=H_{-6}, x24=Q_{-6}
# x_25= P_{-7}, x26=E_{-7}, x27=H_{-7}, x28=Q_{-7}

# x_1 = P_{-0}, x2 =E_{-0}, x3 =H_{-0}, x4 =Q_{-0}
# x_5 = P_{-1}, x6 =E_{-1}, x7 =H_{-1}, x8 =Q_{-1}
# x_9 = P_{-2}, x10=E_{-2}, x11=H_{-2}, x12=Q_{-2}
# x_13= P_{-3}, x14=E_{-3}, x15=H_{-3}, x16=Q_{-3}
# x_17= P_{-4}, x18=E_{-4}, x19=H_{-4}, x20=Q_{-4}
# x_21= P_{-5}, x22=E_{-5}, x23=H_{-5}, x24=Q_{-5}
# x_25= P_{-6}, x26=E_{-6}, x27=H_{-6}, x28=Q_{-6}
