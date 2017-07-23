# -- coding: utf-8 --
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import gc  
import datetime

from feature_extract import *
from xgb_0416 import *
from pltlib import *

#run on centos7, python2.7

#从前三个月用户行为数据提取特征训练模型，预测下一周用户购买物品概率
#唯品会用户购买行为预测
#赛题详细信息: http://www.datafountain.cn/#/competitions/260/intro
if __name__ == '__main__':

	# load data
	print 'load data: %s'%datetime.datetime.now()  
	test_items=pd.read_csv('./input/user_action_test_items.txt', delimiter='\t', header=None)
	test_items.columns=['uid', 'spu_id', 'label']
	goods=pd.read_csv('./input/goods_train.txt', delimiter='\t', header=None)
	goods.columns=['spu_id','brand_id', 'cate_id']
	actions=pd.read_csv('./input/user_action_train.txt', delimiter='\t', header=None)
	actions.columns=['uid', 'spu_id','action_type', 'date']
	actions['buy']=actions.action_type
	actions['click']=1
	actions.drop('action_type', axis=1,inplace=True)
	print 'load finish: %s'%datetime.datetime.now() 
	
	#get train feature, label
	train_label=get_label(actions, '03-17','03-24')
	print 'get x,y: %s'%datetime.datetime.now() 
	train_x,train_y=get_x_y(actions, goods, '01-01','03-17', train_label)
	print train_x.shape
	
	#save feature, label
	#train_x.to_csv('./tmp/train_x_full.csv')
	#train_label.to_csv('./tmp/train_y_full.csv')

	#get test feature, label
	print 'get x,y: %s'%datetime.datetime.now() 
	test_label=get_label(actions, '03-24','03-31')
	test_x,test_y=get_x_y(actions, goods, '01-01','03-24', test_label)
	print test_x.shape

	#save feature, label
	#test_x.to_csv('./tmp/test_x_full.csv', index=None)
	#test_label.to_csv('./tmp/test_y_full.csv', index=None)
	
	#训练模型
	model,train_scores=xgb_model_train(train_x, train_y, test_x, test_y)
	display_score(train_y, train_scores)
	
	#验证模型
	test_scores=xgb_model_predict(model, test_x)
	display_score(test_y, test_scores)
	
	#内存不足分开预测概率，最后合并成一个文件用于提交评测
	print 'get x,y: %s'%datetime.datetime.now() 
	x1,y1=get_x_y(actions, goods, '01-01','03-31', test_items[:2000000])
	pred_y1=xgb_model_predict(model,x1)
	pred1=pd.DataFrame(pred_y1)

	x2,y2=get_x_y(actions, goods, '01-01','03-31', test_items[2000000:4000000])
	pred_y2=xgb_model_predict(model,x2)
	pred2=pd.DataFrame(pred_y2)

	x3,y3=get_x_y(actions, goods, '01-01','03-31', test_items[4000000:])
	pred_y3=xgb_model_predict(model,x3)
	pred3=pd.DataFrame(pred_y3)

	pred=pd.concat([pred1,pred2,pred3])
	pred.columns=['score']
	scores=pred.score.tolist()
	result=['%0.03f'%s for s in scores]
	commit=pd.DataFrame(result)
	commit.to_csv('./output/pred_xgb.txt', index=False, header=False)
