
import numpy as np
import pandas as pd
import seaborn as sns
from glob import glob
from util.metrics import rrmse, agreementindex,  lognashsutcliffe,  nashsutcliffe
from sklearn.metrics import  r2_score, mean_squared_error, mean_absolute_error
from scipy import stats
import hydroeval as he


def fmt(x): 
    if (type(x) == str or type(x) == tuple or type(x) == list):
        return str(x)
    else:
      if (abs(x)>0.001 and abs(x)<1e0):
        return '%1.3f' % x   
      else:
        return '%1.2f' % x #return '%1.3f' % x
  
def fstat(x):
  #m,s= '{:1.4g}'.format(np.mean(x)), '{:1.4g}'.format(np.std(x))
  #m,s, md= fmt(np.mean(x)), fmt(np.std(x)), fmt(np.median(x)) 
  m,s, md= np.mean(x), np.std(x), np.median(x) 
  #text=str(m)+'$\pm$'+str(s)
  s = '--' if s<1e-8 else s
  text=fmt(m)+' ('+fmt(s)+')'#+' ['+str(md)+']'
  return text
#%%
basename='cb__'
path='./pkl_cb*'
pkl_list  = []
for (k,p) in enumerate(glob(path)):
    pkl_list += glob(p+'/'+'*.pkl')


pkl_list.sort()
A=[]
for pkl in pkl_list:
    #print(pkl)
    df = pd.read_pickle(pkl)       
    A.append(df)
A = pd.concat(A, sort=False)

#%%
steps=[
       #'train', 
       'test',
       ]

C = []
for step in steps:
    for k in range(len(A)):
        df=A.iloc[k]
        y_true = pd.DataFrame(df['y_'+step+'_true'], columns=[df['output']])#['0'])
        y_pred = pd.DataFrame(df['y_'+step+'_pred'], columns=[df['output']])#['0'])
        
        run = df['run']
        ds_name = df['dataset_name']
        var_names = y_true.columns

        if len(y_true)>0:
            for v in var_names:
                _mape    = abs((y_true[v] - y_pred[v])/y_true[v]).mean()*100
                #_vaf     = VAF(y_true[v], y_pred[v])
                _r2      = r2_score(y_true[v], y_pred[v])
                _mae     = mean_absolute_error(y_true[v], y_pred[v])
                _mse     = mean_squared_error(y_true[v], y_pred[v])
                _rrmse   = rrmse(y_true[v], y_pred[v])
                _wi      = agreementindex(y_true[v], y_pred[v])
                _r       = stats.pearsonr(y_true[v], y_pred[v])[0]
                #_nse     = he.nse(y_true.values, y_pred.values)[0]
                _nse     = nashsutcliffe(y_true.values, y_pred.values)
                #_lnse    = lognashsutcliffe(y_true.values, y_pred.values)
                _rmse    = he.rmse(y_true.values, y_pred.values)[0]
                #_rmsekx  = rmse_lower(y_true.values, y_pred.values, 'Kx')
                #_rmseq   = rmse_lower(y_true.values, y_pred.values, 'Q')
                _kge     = he.kge(y_true.values, y_pred.values)[0][0]
                _mare    = he.mare(y_true.values, y_pred.values)[0]
                dic     = {'Run':run, 'Output':v, 'MAPE':_mape, 'R$^2$':_r2, 'MSE':_mse,
                           #'Active Features':s2,
                          'Seed':df['seed'], 
                          'Dataset':ds_name, 'Phase':step, 'SI':None,
                          'NSE': _nse, 'MARE': _mare, 'MAE': _mae, #'VAF': _vaf, 
                          #'Active Variables': ', '.join(df['ACTIVE_VAR_NAMES']),
                          #'RMSELKX':rmsekx, 'RMSELQ':rmseq, 
                          #'Scaler': df['SCALER'], 'KGE': _kge,
                          'RMSE':_rmse, 'R':_r, 
                          #'Parameters':df['EST_PARAMS'],
                          'NDEI':_rmse/np.std(y_true.values),
                          'WI':_wi, 'RRMSE':_rrmse,
                          'y_true':y_true.values.ravel(), 
                          'y_pred':y_pred.values.ravel(),
                          'selected_indices':df['selected_indices'],
                          'expression':df['expression'],
                          #'Optimizer':df['ALGO'].split(':')[0], #A['ALGO'].iloc[0].split(':')[0],
                          #'Accuracy':accuracy_log(y_true.values.ravel(), y_pred.values.ravel()),
                          'Estimator':df['estimator'],
                          'Parameters':df['params']
                          }
                for d in df['params']:
                    #print(d, ':',df['params'][d])
                    dic[d] = df['params'][d]

                C.append(dic)

C = pd.DataFrame(C)
C = C.reindex(sorted(C.columns), axis=1)
    
#%%

metrics=[
        #'R', 
        #'R$^2$', 
        'WI',
        'RRMSE',
        #'RMSELKX', 'RMSELQ', 
        #'RMSE$(K_x<100)$', 'RMSE$(B/H<50)$', 
        #'RMSE', #'NDEI', 
        'MAE', #'Accuracy', 
        'MAPE',
        'NSE', #'LNSE', 
        'KGE',
        #'MARE',
        #'VAF', 'MAE (MJ/m$^2$)', 'R',  'RMSE (MJ/m$^2$)',
        ]

metrics_max =  ['NSE', 'VAF', 'R', 'Accuracy','R$^2$', 'KGE', 'WI']    

#%%    
from sklearn import metrics  
import scipy  as sp
import pylab as pl
#pl.rc('text', usetex=True)
pl.rc('font', family='serif',  serif='Times')

id_rmse_min=C['RMSE'].argmin()
df=C.iloc[id_rmse_min]

y_test, y_pred = df['y_true'],df['y_pred']
rmse, r2 = metrics.mean_squared_error(y_test, y_pred)**.5, metrics.r2_score(y_test, y_pred)
r        = sp.stats.pearsonr(y_test.ravel(), y_pred.ravel())[0] 
#tit      = dataset_name+' -- '+target+'\nRMSE = '+str(rmse)+', '+'R$^2$ = '+str(r2)+', '+'R = '+str(r)
tit      = 'RMSE = '+fmt(rmse)+', '+'R$^2$ = '+fmt(r2)+', '+'R = '+fmt(r)

fig = pl.figure(figsize=[12,4])
pl.plot(y_test, 'r-', y_pred,'b-'); pl.legend(['Observed', 'Predicted'])
pl.legend(); pl.title(tit); #pl.xlabel(p)
pl.show()

pl.figure(figsize=(4,4)); 
pl.plot(y_test, y_pred, 'bo', y_test, y_test, 'k-', alpha=0.5)
pl.legend(); pl.title(tit); 
pl.xlabel('Measured streamflow')
pl.ylabel('Predicted streamflow')
pl.savefig('scatter_best_model.png',  bbox_inches='tight', dpi=300)
pl.show()

#%%
residuals = (y_test-y_pred)
pl.figure()
pl.plot(residuals)
pl.xlabel('')
pl.ylabel('Residuals')
pl.savefig('residuals_best_model.png',  bbox_inches='tight', dpi=300)
pl.show()

sns.kdeplot(residuals)
pl.show()

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
pl.figure()
plot_acf(residuals, lags=20)
plot_pacf(residuals, lags=20)
pl.show()



#%%  
