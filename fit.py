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
    
    # fit and test
    test(outpath,X,Y)
    

def test(path,X,Y):
    # parameters
    mse_fit = np.zeros((500,8))
    r2_fit = np.zeros((500,8))
    
    # mlp 
    hiden = np.arange(2,22,2)
    a1 = np.arange(0.001,0.005,0.0001)
    a2 = np.array([0.005,0.006,0.007,0.008,0.009,0.01,0.02,0.03,0.04,0.05])
    alpha = np.concatenate((a1, a2))
    mlpCe = [(a,b) for a in hiden for b in alpha]
    
    # rfr
    rfrCe = np.repeat(np.arange(1,101,1), 5)
    
    # dtr
    dtrCe = np.repeat(np.arange(1,101,1), 5)
    
    nstate = 500
    for istate in range(nstate):
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=istate)
        
        # MLP model
        lyr = mlpCe[istate][0]
        alp = mlpCe[istate][1]
        regr = MLPRegressor(hidden_layer_sizes=(lyr,lyr,lyr), alpha=alp,random_state=1,max_iter=400)
        regr.fit(X_train, y_train)
        y_pred = regr.predict(X_train)
        mse_fit[istate,0] = mean_squared_error(y_train, y_pred)
        r2_fit[istate,0] = r2_score(y_train, y_pred)
        y_pred = regr.predict(X_test)
        mse_fit[istate,1] = mean_squared_error(y_test, y_pred)
        r2_fit[istate,1] = r2_score(y_test, y_pred)      
        
        # RFR model
        regr = RandomForestRegressor(n_estimators=rfrCe[istate])
        regr.fit(X_train, y_train)
        y_pred = regr.predict(X_train)
        mse_fit[istate,2] = mean_squared_error(y_train, y_pred)
        r2_fit[istate,2] = r2_score(y_train, y_pred)
        y_pred = regr.predict(X_test)
        mse_fit[istate,3] = mean_squared_error(y_test, y_pred)
        r2_fit[istate,3] = r2_score(y_test, y_pred)
        
        # DTR model
        regr = DecisionTreeRegressor(max_depth=dtrCe[istate])
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
        
        print('iteration'+str(istate)+'is finished!')
    mse_data = pd.DataFrame(mse_fit, columns = ['mlptrain','mlptest','rfrtrain','rfrtest','dtrtrain','dtrtest','mlrtrain','mlrtest'])
    r2_data = pd.DataFrame(r2_fit, columns = ['mlptrain','mlptest','rfrtrain','rfrtest','dtrtrain','dtrtest','mlrtrain','mlrtest'])
    mse_data.to_csv(path+'msefit.csv',index=False)
    r2_data.to_csv(path+'r2_fit.csv',index=False)
        
if __name__ == '__main__':
    distr_model()
