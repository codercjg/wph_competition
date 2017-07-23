# -- coding: utf-8 --
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import gc  
import datetime
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from sklearn import metrics
import datetime
import pickle
from sklearn import metrics
import xgboost as xgb

import operator

#用于xgboost计算f1
def xgb_f1(y,t):
    t = t.get_label()
    y_bin = [1. if y_cont >= 0.5 else 0. for y_cont in y] # binaryzing your output
    return 'f1',metrics.f1_score(t,y_bin)

#画出xgboost特征重要性
def get_xgb_feature_importance(model, x):
    features = [f for f in x.columns]  
    #ceate_feature_map(features)  
    outfile = open('xgb.fmap', 'w')  
    i = 0  
    for feat in features:  
        outfile.write('{0}\t{1}\tq\n'.format(i, feat))  
        i = i + 1  
    outfile.close()
    
    importance = model.get_fscore(fmap='xgb.fmap')  
    importance = sorted(importance.items(), key=operator.itemgetter(1))  
  
    df = pd.DataFrame(importance, columns=['feature', 'fscore'])  
    df['fscore'] = df['fscore'] / df['fscore'].sum()  
    df.to_csv("feat_importance.csv", index=False)  
  
    plt.figure()  
    df.plot(kind='barh', x='feature', y='fscore', legend=False, figsize=(6, 10))  
    plt.title('XGBoost Feature Importance')  
    plt.xlabel('relative importance')  
    plt.show()
    return df
	
#xgboost train
def xgb_model_train(x, y, test_x=None, test_y=None):
    xgb_params = {
        'max_depth':4, 
        'learning_rate':0.05,
        'n_estimators':1000, 
        'silent':True, 
        'objective':'binary:logistic', 
        'booster':'gbtree', 
        'n_jobs':1, 
        'nthread':0, 
        'gamma':0.4, 
        'min_child_weight':5, 
        'max_delta_step':0, 
        'subsample':0.8, 
        'colsample_bytree':0.9, 
        'reg_alpha':5, 
        'scale_pos_weight':1, 
        'eval_metric': 'rmse',
        'random_state':1
    }
    
    dtrain = xgb.DMatrix(x, y)
    dtest = xgb.DMatrix(test_x, test_y)
    watchlist = [(dtrain, 'train'),(dtest, 'test')]

    #model = xgb.train(dict(xgb_params, silent=0), dtrain, early_stopping_rounds=30, num_boost_round=1000, evals=watchlist,feval=xgb_f1)
    model = xgb.train(dict(xgb_params, silent=0), dtrain, early_stopping_rounds=30, num_boost_round=331,evals=watchlist)

    print 'xgb pred: %s'%datetime.datetime.now() 
    scores = model.predict(dtrain)

    model_save(model, 'xgb.txt')
    return (model,scores)
	
#xgboost predict
def xgb_model_predict(model, x):
    dtrain = xgb.DMatrix(x)
    scores = model.predict(dtrain)
    return scores
	
def model_save(obj, path):
    with open(path, 'wb') as handle:
        pickle.dump(obj, handle)
    
def model_load(path):
    with open(path, 'rb') as handle:
        obj = pickle.load(handle)
    return obj
