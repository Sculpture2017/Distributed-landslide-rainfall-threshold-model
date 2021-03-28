#!/usr/bin/env python

import sklearn
import pandas as pd
import numpy as np

from sklearn import linear_model
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
from joblib import dump

from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import LinearSVR
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor

def distr_model():
    # directory
    path = 'D:/climate/model/'
    folder = 'data/48-1-10-8/'
    outpath = path+'fitting/'
    
    # prepare train data
    #model_data(path,folder)
    
    # read train data
    traindata = pd.read_csv(path+folder+'fitdata.csv')
    traindata = traindata[traindata['delay']<240]
    traindata = traindata[['re75','re90','sp75','sp90','tem','artiG','cropG','grassG','herbG','shrubG','snowG','treeG','waterG','D','E']]
    
    # slope scaling
    traindata['sp75'] = traindata['sp75']/100
    traindata['sp90'] = traindata['sp90']/100
    traindata['D'] = np.log10(traindata['D'])
    traindata['E'] = np.log10(traindata['E'])
    traindata.to_csv(outpath+'traindata.csv',index=False)
    
    # Split datasets
    X = traindata.iloc[:,:-1]
    Y = traindata.iloc[:,-1]

    
    
    # train models
    # model_mlr(outpath,X,Y)
    # model_dtr(outpath,X,Y)
    # model_rfr(outpath,X,Y)
    # model_mlp(outpath,X,Y)
    
    # fit and test
    fit_test(outpath,X,Y)
    
    
# Module II: model_train, Multi-layer Perceptron (MLP) 
def model_mlp(path,X,Y):
    # train model
    regr = MLPRegressor(hidden_layer_sizes=(6,6,6), alpha=0.0029,random_state=1,max_iter=400)
    regr.fit(X, Y)
    y_pred = regr.predict(X)
    trainE = list(Y)
    predE =  list(y_pred)
    trainD = list(X['D'])
    traindata = pd.DataFrame(
    {'trainD': trainD,
     'trainE': trainE,
     'predE': predE
    })
    traindata.to_csv(path+'mlptrain.csv',index=False)
    
    # save
    dump(regr, path+'mlpmodel.gz')  

# Module II: model_train, Random Forest Regressor (RFR)
def model_rfr(path,X,Y):
    # train model
    regr = RandomForestRegressor(n_estimators=89)
    regr.fit(X, Y)
    y_pred = regr.predict(X)
    trainE = list(Y)
    predE =  list(y_pred)
    trainD = list(X['D'])
    traindata = pd.DataFrame(
    {'trainD': trainD,
     'trainE': trainE,
     'predE': predE
    })
    traindata.to_csv(path+'rfrtrain.csv',index=False)
    
    # save
    dump(regr, path+'rfrmodel.gz')  
     

# Module II: model_train, Decision Tree Regression (DTR)
def model_dtr(path,X,Y):
    # train model
    regr = DecisionTreeRegressor(max_depth=4)
    regr.fit(X, Y)
    y_pred = regr.predict(X)
    trainE = list(Y)
    predE =  list(y_pred)
    trainD = list(X['D'])
    traindata = pd.DataFrame(
    {'trainD': trainD,
     'trainE': trainE,
     'predE': predE
    })
    traindata.to_csv(path+'dtrtrain.csv',index=False)
    
    # save
    dump(regr, path+'dtrmodel.gz')    
    

# Module II: model_train, Multiple Linear Regression (MLR)
def model_mlr(path,X,Y):    
    # variables selection based on P-value
    del X['cropG']
    del X['sp75']
    del X['shrubG']
 
    # sklearn linear regression
    regr = linear_model.LinearRegression()
    regr.fit(X, Y)
    y_pred = regr.predict(X)
    trainE = list(Y)
    predE =  list(y_pred)
    trainD = list(X['D'])
    traindata = pd.DataFrame(
    {'trainD': trainD,
     'trainE': trainE,
     'predE': predE
    })
    traindata.to_csv(path+'mlrtrain.csv',index=False)
    
    # save
    dump(regr, path+'mlrmodel.gz')        

# Module I: model_data, preparation based on envi and event data
def model_data(path,folder):
    # read data
    envi = pd.read_csv(path+'slide.csv')
    event = pd.read_csv(path+folder+'LRE.csv',sep=' ')
    rain = event[['ID','D_L','E_L','dh_RE_lan']]
    
    # prepare train data
    nm = envi.shape
    num = rain.shape
    size = (nm[0],num[1])
    tmp = np.empty(size)
    tmp[:]=np.nan
    NO = list(envi['ID'])
    for ir in range(nm[0]):
       slt = rain[rain['ID']==NO[ir]]
       if slt.shape[0]:
           tmp[ir,0] = slt.iloc[0,0]
           tmp[ir,1] = slt.iloc[0,1]
           tmp[ir,2] = slt.iloc[0,2]
           tmp[ir,3] = slt.iloc[0,3] 
    tmpdata = pd.DataFrame(tmp,columns=['ID','D','E','delay'])
    fitdata = pd.concat([envi,tmpdata],axis=1)
    #fitdata.dropna() 
    fitdata = fitdata.loc[~pd.isnull(fitdata).any(axis=1)]
    fitdata.to_csv(path+folder+'fitdata.csv',index=False)

def fit_test(path,X,Y):
    # MLR parameters
    mse_fit = np.zeros((500,8))
    r2_fit = np.zeros((500,8))
    
    nstate = 500
    for istate in range(nstate):
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=istate)
        
        # MLP model
        regr = MLPRegressor(hidden_layer_sizes=(6,6,6), alpha=0.0029,random_state=1,max_iter=400)
        regr.fit(X_train, y_train)
        y_pred = regr.predict(X_train)
        mse_fit[istate,0] = mean_squared_error(y_train, y_pred)
        r2_fit[istate,0] = r2_score(y_train, y_pred)
        y_pred = regr.predict(X_test)
        mse_fit[istate,1] = mean_squared_error(y_test, y_pred)
        r2_fit[istate,1] = r2_score(y_test, y_pred)      
        
        # RFR model
        regr = RandomForestRegressor(n_estimators=89)
        regr.fit(X_train, y_train)
        y_pred = regr.predict(X_train)
        mse_fit[istate,2] = mean_squared_error(y_train, y_pred)
        r2_fit[istate,2] = r2_score(y_train, y_pred)
        y_pred = regr.predict(X_test)
        mse_fit[istate,3] = mean_squared_error(y_test, y_pred)
        r2_fit[istate,3] = r2_score(y_test, y_pred)
        
        # DTR model
        regr = DecisionTreeRegressor(max_depth=4)
        regr.fit(X_train, y_train)
        y_pred = regr.predict(X_train)
        mse_fit[istate,4] = mean_squared_error(y_train, y_pred)
        r2_fit[istate,4] = r2_score(y_train, y_pred)
        y_pred = regr.predict(X_test)
        mse_fit[istate,5] = mean_squared_error(y_test, y_pred)
        r2_fit[istate,5] = r2_score(y_test, y_pred)
        
        # MLR model
        del X_train['cropG']; del X_train['sp75']; del X_train['shrubG']
        del X_test['cropG']; del X_test['sp75']; del X_test['shrubG']
        regr = linear_model.LinearRegression()
        regr.fit(X_train, y_train)
        y_pred = regr.predict(X_train)
        mse_fit[istate,6] = mean_squared_error(y_train, y_pred)
        r2_fit[istate,6] = r2_score(y_train, y_pred)
        y_pred = regr.predict(X_test)
        mse_fit[istate,7] = mean_squared_error(y_test, y_pred)
        r2_fit[istate,7] = r2_score(y_test, y_pred)
        
        print('iteration '+str(istate)+' is finished!')
    mse_data = pd.DataFrame(mse_fit, columns = ['mlptrain','mlptest','rfrtrain','rfrtest','dtrtrain','dtrtest','mlrtrain','mlrtest'])
    r2_data = pd.DataFrame(r2_fit, columns = ['mlptrain','mlptest','rfrtrain','rfrtest','dtrtrain','dtrtest','mlrtrain','mlrtest'])
    mse_data.to_csv(path+'msedata.csv',index=False)
    r2_data.to_csv(path+'r2_data.csv',index=False)
        
if __name__ == '__main__':
    distr_model()
