#!/usr/bin/env python3

import grass.script as gscript
import os

## import data: event and duration
def importdata():
    path  = 'D:/climate/model/data/'
    os.chdir(path)
    gscript.run_command('v.in.ogr', input='event.shp',output='event')
    gscript.run_command('v.in.ogr', input='duration.shp',output='duration')

    gscript.run_command('g.region', raster='dem')
    gscript.run_command('v.to.rast', input='duration',type='point',use='attr',attribute_column='duration',output='duration')

## train and predict data
def predictdata():
    gscript.run_command('r.mask', raster='relief',overwrite=True)
    gscript.run_command('g.region', raster='dem')
    morvar = ['reav','re25','re75','re90','spav','sp25','sp75','sp90','tem']
    landvar = ['artiG','bareG','cropG','grassG','herbG','mangG','shrubG','snowG','sparseG','treeG','waterG']
    
    # group train data
    variable = morvar+landvar
    variable.append('duration')
    gscript.run_command('i.group', group='train',input=variable,overwrite=True)
    
    # group predict data
    dura = [1,3,24,72,168,720]
    for ir in range(6):
        period = 'duration'+str(dura[ir])
        gscript.mapcalc('$dr=$a',dr=period,a=dura[ir])
        variable = morvar+landvar
        variable.append(period)
        gscript.run_command('i.group', group='pred'+str(dura[ir]),input=variable)
        
## Linear Regression model 
def mlr_model():
    model = 'mlr'
    path = 'D:/climate/model/'
    os.chdir(path)
    gscript.run_command('g.region', raster='dem')
    
    # train model
    mfile = 'model'+model+'.gz'
    #gscript.run_command('r.learn.train', group='train',training_points='event',n_estimators=100,field='rainfall',save_model=mfile,model_name='LinearRegression')
    
    # predict model
    dura = [3,24,72,168,720]
    for ir in range(5):
        predata = 'pred'+str(dura[ir])
        thresh = 'thre_'+model+'_'+str(dura[ir])
        thre10 = 'thre_'+model+str(dura[ir])
        #gscript.run_command('r.learn.predict', group=predata,load_model=mfile,output=thresh)
        gscript.mapcalc('$th=int($a*10)',th=thre10,a=thresh)
        gscript.run_command('g.remove', type='raster', name=thresh, flags='f', quiet=True)
        gscript.run_command('r.out.gdal', input=thre10,output='threshold/'+thre10+'.tif',format='GTiff')

## Decision Tree Regression model 
def dtr_model():
    model = 'dtr'
    path = 'D:/climate/model/'
    os.chdir(path)
    gscript.run_command('g.region', raster='dem')
    
    # train model
    mfile = 'model'+model+'.gz'
    #gscript.run_command('r.learn.train', group='train',training_points='event',n_estimators=100,field='rainfall',save_model=mfile,model_name='DecisionTreeRegressor')
    
    # predict model
    dura = [1,3,24,72,168,720]
    for ir in range(6):
        predata = 'pred'+str(dura[ir])
        thresh = 'thre_'+model+'_'+str(dura[ir])
        thre10 = 'thre_'+model+str(dura[ir])
        gscript.run_command('r.learn.predict', group=predata,load_model=mfile,output=thresh)
        gscript.mapcalc('$th=int($a*10)',th=thre10,a=thresh)
        gscript.run_command('g.remove', type='raster', name=thresh, flags='f', quiet=True)
        gscript.run_command('r.out.gdal', input=thre10,output='threshold/'+thre10+'.tif',format='GTiff')

## Support Vector Machine model 
def svr_model():
    model = 'svr'
    path = 'D:/climate/model/'
    os.chdir(path)
    gscript.run_command('g.region', raster='dem')
    
    # train model
    mfile = 'model'+model+'.gz'
    gscript.run_command('r.learn.train', group='train',training_points='event',n_estimators=100,field='rainfall',save_model=mfile,model_name='SVR')
    
    # predict model
    dura = [1,3,24,72,168,720]
    for ir in range(6):
        predata = 'pred'+str(dura[ir])
        thresh = 'thre_'+model+'_'+str(dura[ir])
        thre10 = 'thre_'+model+str(dura[ir])
        gscript.run_command('r.learn.predict', group=predata,load_model=mfile,output=thresh)
        gscript.mapcalc('$th=int($a*10)',th=thre10,a=thresh)
        gscript.run_command('g.remove', type='raster', name=thresh, flags='f', quiet=True)
        gscript.run_command('r.out.gdal', input=thre10,output='threshold/'+thre10+'.tif',format='GTiff')

## Multi-layer perceptron algorithm model 
def mlp_model():
    model = 'mlp'
    path = 'D:/climate/model/'
    os.chdir(path)
    gscript.run_command('g.region', raster='dem')
    
    # train model
    mfile = 'model'+model+'.gz'
    gscript.run_command('r.learn.train', group='train',training_points='event',n_estimators=100,field='rainfall',save_model=mfile,model_name='MLPRegressor')
    
    # predict model
    dura = [1,3,24,72,168,720]
    for ir in range(6):
        predata = 'pred'+str(dura[ir])
        thresh = 'thre_'+model+'_'+str(dura[ir])
        thre10 = 'thre_'+model+str(dura[ir])
        gscript.run_command('r.learn.predict', group=predata,load_model=mfile,output=thresh)
        gscript.mapcalc('$th=int($a*10)',th=thre10,a=thresh)
        gscript.run_command('g.remove', type='raster', name=thresh, flags='f', quiet=True)
        gscript.run_command('r.out.gdal', input=thre10,output='threshold/'+thre10+'.tif',format='GTiff')

if __name__ == '__main__':
    mlp_model()
