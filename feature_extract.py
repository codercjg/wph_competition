# -- coding: utf-8 --
import pandas as pd
import numpy as np
import gc  
import datetime
from sklearn.utils import shuffle 

#两个日期间隔的天数
def delt_days(dateA,dateB):
    d1 = datetime.datetime.strptime(dateA,'%m-%d')
    d2 = datetime.datetime.strptime(dateB,'%m-%d')
    return ((d2-d1).days)
	
#通过间隔的天数计算日期
def get_date_by_days(date, days):
    d=datetime.datetime.strptime(date,'%m-%d');
    return (d - datetime.timedelta(days=days)).strftime('%m-%d')

#提取user,item, user-item等特征，实际比赛中，因为内存不足放弃了一部分特征，最后保留了其中的300多个。
def get_full_global_feat(actions, goods, begin_date,end_date,label):

    predict_date=get_date_by_days(end_date,-1)
    
    if begin_date>'01-01':
        never_buy_date=get_date_by_days(begin_date,1)
        never_buy_delt_day=delt_days(never_buy_date,end_date)
    else:
        never_buy_delt_day=90
        
    print "generate features %s"%begin_date    
    act=actions[(actions.date>begin_date) & (actions.date<=end_date)]    
    print "generate user features "
    xact=pd.merge(act, goods, how='left', on=['spu_id'])
    #用户最近购买日期
    user_last_buy_date=xact.date.groupby(xact[xact.buy==1].uid).max().reset_index()
    #用户最近点击日期
    user_last_click_date=xact.date.groupby(xact.uid).max().reset_index()
    
    #将日期转化成与要预测的日期相隔的天数
    user_last_buy_date['buy_delt_day']=user_last_buy_date.date.apply(lambda date: delt_days(date, predict_date))
    user_last_click_date['click_delt_day']=user_last_click_date.date.apply(lambda date: delt_days(date, predict_date))
    
    #not_buy没有购买记录
    user_feat=pd.merge(user_last_click_date[['uid','click_delt_day']],user_last_buy_date[['uid','buy_delt_day']], how='left',on=['uid'])
       
    del user_last_buy_date
    del user_last_click_date
    user_feat['buy_delt_day'].fillna(never_buy_delt_day, inplace=True)
    user_feat['never_buy']=0
    user_feat.loc[user_feat.buy_delt_day==never_buy_delt_day, 'never_buy']=1
    
    #用户购买商品的天数
    user_buy_days=xact[['uid', 'date', 'buy']].drop_duplicates().groupby(['uid']).sum().reset_index()
    user_buy_days.columns=['uid','buy_days']
    #用户点击物品的天数
    user_click_days=xact[['uid', 'date', 'click']].drop_duplicates().groupby(['uid']).sum().reset_index()
    user_click_days.columns=['uid','click_days']
    user_feat=pd.merge(user_feat,user_buy_days, how='left',on=['uid'])
    user_feat=pd.merge(user_feat,user_click_days, how='left',on=['uid'])
    
    del user_buy_days
    del user_click_days
    gc.collect()
    
    #用户某天购买的商品数
    user_buy_by_day=xact[['uid', 'date', 'buy']].groupby(['uid','date']).sum().reset_index()
    #用户某天点击物品的商品数
    user_click_by_day=xact[['uid', 'date', 'click']].groupby(['uid','date']).sum().reset_index()
    #用户某天点击、购买的商品数
    user_by_day=xact[['uid', 'date', 'buy','click']].groupby(['uid','date']).sum().reset_index()
    #用户某天转化率
    user_by_day['frac']=user_by_day.buy/(user_by_day.click*1.0)
    user_click_by_day=user_by_day[['uid', 'date', 'click']]
    user_buy_by_day=user_by_day[['uid', 'date', 'buy']]
    user_frac_by_day=user_by_day[['uid', 'date', 'frac']]
    #用户购买最多的商品数的那一天的商品点击量
    user_buy_max_by_day=user_buy_by_day.buy.groupby(user_buy_by_day.uid).max().reset_index()
    #用户点击最多的商品数的那一天的商品购买量
    user_click_max_by_day=user_click_by_day.click.groupby(user_click_by_day.uid).max().reset_index()
    #用户转化率最高的一天的转化率
    user_frac_max_by_day=user_frac_by_day.frac.groupby(user_frac_by_day.uid).max().reset_index()

    #用户点击最多的那一天日期
    user_click_max_day=pd.merge(user_click_max_by_day, user_click_by_day, how='left', on=['uid','click'],left_index=True, right_index=True)
    #用户购买最多的那一天日期
    user_buy_max_day=pd.merge(user_buy_max_by_day, user_buy_by_day, how='left', on=['uid','buy'],left_index=True, right_index=True)
    #用户转化率最高的那一天日期
    user_frac_max_day=pd.merge(user_frac_max_by_day, user_frac_by_day, how='left', on=['uid','frac'],left_index=True, right_index=True)

    del user_click_by_day
    del user_click_max_by_day
    del user_buy_by_day
    del user_buy_max_by_day
    del user_frac_by_day
    del user_frac_max_by_day
    gc.collect()
    
    user_click_max_day['max_click_delt_days']=user_click_max_day.date.apply(lambda date: delt_days(date, predict_date))
    user_click_max_day.drop('date',axis=1, inplace=True)
    user_click_max_day.columns=['uid','max_click','max_click_delt_days']

    user_buy_max_day['max_buy_delt_days']=user_buy_max_day.date.apply(lambda date: delt_days(date, predict_date))
    user_buy_max_day.drop('date',axis=1, inplace=True)
    user_buy_max_day.columns=['uid','max_buy','max_buy_delt_days']

    user_frac_max_day['max_frac_delt_days']=user_frac_max_day.date.apply(lambda date: delt_days(date, predict_date))
    user_frac_max_day.drop('date',axis=1, inplace=True)
    user_frac_max_day.columns=['uid','max_frac','max_frac_delt_days']

    user_feat=pd.merge(user_feat,user_click_max_day, how='left',on=['uid'])
    user_feat=pd.merge(user_feat,user_buy_max_day, how='left',on=['uid'])
    user_feat=pd.merge(user_feat,user_frac_max_day, how='left',on=['uid'])

    del user_click_max_day
    del user_buy_max_day
    del user_frac_max_day
    gc.collect()

    
    #用户总点击数和总购买数，同一件商品可能被多次购买，算多次
    click_buy_count_by_user=xact[['uid','click', 'buy']].groupby('uid').sum().reset_index()
    #用户转化率
    click_buy_count_by_user['frac']=click_buy_count_by_user.buy/(click_buy_count_by_user.click*1.0)
    #click_buy_count_by_user['sub']=click_buy_count_by_user.click-click_buy_count_by_user.buy

    #用户点击商品数和购买商品数，同一件商品被多次购买算一次
    user_item=xact[['uid','spu_id','click', 'buy']].drop_duplicates().groupby(['uid']).sum().reset_index()
    user_item.drop('spu_id', axis=1,inplace=True)
    #用户点击商品数和购买商品数的转化率
    user_item['frac'] = user_item.buy/(user_item.click*1.0)
    #user_item['sub'] = user_item.click-user_item.buy
    #用户点击分类数和购买分类数
    user_cate=xact[['uid','cate_id','click', 'buy']].drop_duplicates().groupby(['uid']).sum().reset_index()
    user_cate.drop('cate_id', axis=1,inplace=True)

    #用户点击分类数和购买分类数的转化率
    user_cate['frac'] = user_cate.buy/(user_cate.click*1.0)
    #user_cate['sub']=user_cate.click-user_cate.buy
    #用户点击品牌数和购买品牌数
    user_brand=xact[['uid','brand_id','click','buy']].drop_duplicates().groupby(['uid']).sum().reset_index()
    user_brand.drop('brand_id', axis=1,inplace=True)

    #用户对某品牌的转化率
    user_brand['frac'] = user_brand.buy/(user_brand.click*1.0)
    #user_brand['sub'] = user_brand.click-user_brand.buy
    click_buy_count_by_user.columns=['uid','total_click_item_count','total_buy_item_count','total_user_frac']
    user_item.columns=['uid','click_item_count','buy_item_count','user_item_frac']
    user_cate.columns=['uid','click_cate_count','buy_cate_count','user_cate_frac']
    user_brand.columns=['uid','click_brand_count','buy_brand_count','user_brand_frac']
  
    user_feat=pd.merge(user_feat, click_buy_count_by_user, how='left', on=['uid'])
    user_feat=pd.merge(user_feat, user_item, how='left', on=['uid'])
    user_feat=pd.merge(user_feat, user_cate, how='left', on=['uid'])
    user_feat=pd.merge(user_feat, user_brand, how='left', on=['uid'])
    del click_buy_count_by_user
    del user_item
    del user_cate
    del user_brand
    gc.collect()

    #构造组合特征
    print "generate user sub features "
    user_feat['sub_click_buy_day']=user_feat.click_delt_day-user_feat.buy_delt_day
    user_feat['sub_max_click_buy_day']=user_feat.max_click_delt_days-user_feat.max_buy_delt_days
    user_feat['sub_last_buy_max_buy_day']=user_feat.buy_delt_day-user_feat.max_buy_delt_days
    user_feat['sub_last_click_max_buy_day']=user_feat.click_delt_day-user_feat.max_buy_delt_days
    user_feat['sub_last_buy_max_click_day']=user_feat.buy_delt_day-user_feat.max_click_delt_days
    user_feat['sub_last_click_max_click_day']=user_feat.click_delt_day-user_feat.max_click_delt_days
    user_feat['sub_total_click_buy']=user_feat.total_click_item_count-user_feat.total_buy_item_count
    user_feat['sub_cate_click_buy']=user_feat.click_cate_count-user_feat.buy_cate_count
    user_feat['sub_brand_click_buy']=user_feat.click_brand_count-user_feat.click_brand_count
    user_feat['sub_item_click_buy']=user_feat.click_item_count-user_feat.buy_item_count
    user_feat['sub_item_click_cate_buy']=user_feat.click_item_count-user_feat.buy_cate_count
    user_feat['sub_item_buy_cate_buy']=user_feat.buy_item_count-user_feat.buy_cate_count
    
    print "generate item features "
    #商品某天点击、购买数
    item_by_day=xact[['spu_id', 'date', 'buy','click']].groupby(['spu_id','date']).sum().reset_index()
    #某商品某天转化率
    item_by_day['frac']=item_by_day.buy/(item_by_day.click*1.0)

    item_click_by_day=item_by_day[['spu_id', 'date', 'click']]
    item_buy_by_day=item_by_day[['spu_id', 'date', 'buy']]
    item_frac_by_day=item_by_day[['spu_id', 'date', 'frac']]

    #商品购买最多的那一天购买量
    item_buy_max_by_day=item_buy_by_day.buy.groupby(item_buy_by_day.spu_id).max().reset_index()
    #商品点击最多的那一天点击量
    item_click_max_by_day=item_click_by_day.click.groupby(item_click_by_day.spu_id).max().reset_index()
    #商品点击最多的那一天点击量
    item_frac_max_by_day=item_frac_by_day.frac.groupby(item_frac_by_day.spu_id).max().reset_index()

    #商品点击最多的一天日期
    item_click_max_day=pd.merge(item_click_max_by_day, item_click_by_day, how='left', on=['spu_id','click'],left_index=True, right_index=True)
    #商品购买最多的一天日期
    item_buy_max_day=pd.merge(item_buy_max_by_day, item_buy_by_day, how='left', on=['spu_id','buy'],left_index=True, right_index=True)
    #商品转化率最高的一天日期
    item_frac_max_day=pd.merge(item_frac_max_by_day, item_frac_by_day, how='left', on=['spu_id','frac'],left_index=True, right_index=True)

    item_click_max_day['max_click_delt_days']=item_click_max_day.date.apply(lambda date: delt_days(date, predict_date))
    
    item_buy_max_day['max_buy_delt_days']=item_buy_max_day.date.apply(lambda date: delt_days(date, predict_date))
    item_frac_max_day['max_frac_delt_days']=item_frac_max_day.date.apply(lambda date: delt_days(date, predict_date))
                                                                         
    
    item_click_max_day.drop('date',axis=1, inplace=True)
    item_click_max_day.columns=['spu_id','item_max_click','item_max_click_delt_days']
    item_buy_max_day.drop('date',axis=1, inplace=True)
    item_buy_max_day.columns=['spu_id','item_max_buy','item_max_buy_delt_days']
    item_frac_max_day.drop('date',axis=1, inplace=True)
    item_frac_max_day.columns=['spu_id','item_max_frac','item_max_frac_delt_days']
                                                                     
    item_feat=pd.merge(goods,item_click_max_day, how='left',on=['spu_id'])
    item_feat=pd.merge(item_feat,item_buy_max_day, how='left',on=['spu_id'])
    item_feat=pd.merge(item_feat,item_frac_max_day, how='left',on=['spu_id'])
    del item_click_max_day
    del item_buy_max_day
    del item_frac_max_day
    gc.collect()
    
    item_feat.item_max_click.fillna(0, inplace=True)
    item_feat.item_max_buy.fillna(0, inplace=True)
    item_feat.item_max_frac.fillna(0, inplace=True)
    item_feat.item_max_click_delt_days.fillna(never_buy_delt_day, inplace=True)
    item_feat.item_max_buy_delt_days.fillna(never_buy_delt_day, inplace=True)
    item_feat.item_max_frac_delt_days.fillna(never_buy_delt_day, inplace=True)
                                                                     
    #某分类某天点击、购买的商品数
    cate_by_day=xact[['cate_id', 'date', 'buy','click']].groupby(['cate_id','date']).sum().reset_index()
    #某分类某天转化率
    cate_by_day['frac']=cate_by_day.buy/(cate_by_day.click*1.0)

    cate_click_by_day=cate_by_day[['cate_id', 'date', 'click']]
    cate_buy_by_day=cate_by_day[['cate_id', 'date', 'buy']]
    cate_frac_by_day=cate_by_day[['cate_id', 'date', 'frac']]
                                                                     
    #分类购买最多的那一天的购买量
    cate_buy_max_by_day=cate_buy_by_day.buy.groupby(cate_buy_by_day.cate_id).max().reset_index()
    #分类点击最多的那一天的点击量
    cate_click_max_by_day=cate_click_by_day.click.groupby(cate_click_by_day.cate_id).max().reset_index()
    #分类转化率最高的那一天的转化率
    cate_frac_max_by_day=cate_frac_by_day.frac.groupby(cate_frac_by_day.cate_id).max().reset_index()
                                                                     
    #分类点击最多的一天日期
    cate_click_max_day=pd.merge(cate_click_max_by_day, cate_click_by_day, how='left', on=['cate_id','click'],left_index=True, right_index=True)
    #分类购买最多的一天日期
    cate_buy_max_day=pd.merge(cate_buy_max_by_day, cate_buy_by_day, how='left', on=['cate_id','buy'],left_index=True, right_index=True)
    #分类转化率最高的那一天的日期
    cate_frac_max_day=pd.merge(cate_frac_max_by_day, cate_frac_by_day, how='left', on=['cate_id','frac'],left_index=True, right_index=True)
    
    del cate_click_by_day
    del cate_buy_by_day
    del cate_frac_by_day
    gc.collect()
    
    cate_click_max_day['max_click_delt_days']=cate_click_max_day.date.apply(lambda date: delt_days(date, predict_date))
    cate_buy_max_day['max_buy_delt_days']=cate_buy_max_day.date.apply(lambda date: delt_days(date, predict_date))
    cate_frac_max_day['max_frac_delt_days']=cate_frac_max_day.date.apply(lambda date: delt_days(date, predict_date))
                                                                     
    cate_click_max_day.drop('date',axis=1, inplace=True)
    cate_click_max_day.columns=['cate_id','cate_max_click','cate_max_click_delt_days']
    cate_buy_max_day.drop('date',axis=1, inplace=True)
    cate_buy_max_day.columns=['cate_id','cate_max_buy','cate_max_buy_delt_days']
    cate_frac_max_day.drop('date',axis=1, inplace=True)
    cate_frac_max_day.columns=['cate_id','cate_max_frac','cate_max_frac_delt_days']
            
    item_feat=pd.merge(item_feat,cate_click_max_day, how='left',on=['cate_id'])
    item_feat=pd.merge(item_feat,cate_buy_max_day, how='left',on=['cate_id'])
    item_feat=pd.merge(item_feat,cate_frac_max_day, how='left',on=['cate_id'])
    item_feat.cate_max_click.fillna(0, inplace=True)
    item_feat.cate_max_buy.fillna(0, inplace=True)
    item_feat.cate_max_frac.fillna(0, inplace=True)
    item_feat.cate_max_click_delt_days.fillna(never_buy_delt_day, inplace=True)
    item_feat.cate_max_buy_delt_days.fillna(never_buy_delt_day, inplace=True)
    item_feat.cate_max_frac_delt_days.fillna(never_buy_delt_day, inplace=True)
    del cate_click_max_day
    del cate_buy_max_day
    del cate_frac_max_day
    gc.collect()
                                                                 
    #某品牌某天点击、购买的商品数
    brand_by_day=xact[['brand_id', 'date', 'buy','click']].groupby(['brand_id','date']).sum().reset_index()
    #某品牌某天转化率
    brand_by_day['frac']=brand_by_day.buy/(brand_by_day.click*1.0)

    brand_click_by_day=brand_by_day[['brand_id', 'date', 'click']]
    brand_buy_by_day=brand_by_day[['brand_id', 'date', 'buy']]
    brand_frac_by_day=brand_by_day[['brand_id', 'date', 'frac']]
                                                                     
    #品牌购买最多的那一天的购买量
    brand_buy_max_by_day=brand_buy_by_day.buy.groupby(brand_buy_by_day.brand_id).max().reset_index()
    #品牌点击最多的那一天的点击量
    brand_click_max_by_day=brand_click_by_day.click.groupby(brand_click_by_day.brand_id).max().reset_index()
    #品牌转化率最高的那一天的转化率
    brand_frac_max_by_day=brand_frac_by_day.frac.groupby(brand_frac_by_day.brand_id).max().reset_index()
                                                                     
    #品牌点击最多的一天日期
    brand_click_max_day=pd.merge(brand_click_max_by_day, brand_click_by_day, how='left', on=['brand_id','click'],left_index=True, right_index=True)
    #品牌购买最多的一天日期
    brand_buy_max_day=pd.merge(brand_buy_max_by_day, brand_buy_by_day, how='left', on=['brand_id','buy'],left_index=True, right_index=True)
    #品牌转化率最高的一天日期
    brand_frac_max_day=pd.merge(brand_frac_max_by_day, brand_frac_by_day, how='left', on=['brand_id','frac'],left_index=True, right_index=True)
                                                                     
    brand_click_max_day['max_click_delt_days']=brand_click_max_day.date.apply(lambda date: delt_days(date, predict_date))
    brand_buy_max_day['max_buy_delt_days']=brand_buy_max_day.date.apply(lambda date: delt_days(date, predict_date))
    brand_frac_max_day['max_frac_delt_days']=brand_frac_max_day.date.apply(lambda date: delt_days(date, predict_date))
    brand_click_max_day.drop('date',axis=1, inplace=True)
    brand_click_max_day.columns=['brand_id','brand_max_click','brand_max_click_delt_days']
    brand_buy_max_day.drop('date',axis=1, inplace=True)
    brand_buy_max_day.columns=['brand_id','brand_max_buy','brand_max_buy_delt_days']
    brand_frac_max_day.drop('date',axis=1, inplace=True)
    brand_frac_max_day.columns=['brand_id','brand_max_frac','brand_max_frac_delt_days']
                                                                     
    item_feat=pd.merge(item_feat,brand_click_max_day, how='left',on=['brand_id'])
    item_feat=pd.merge(item_feat,brand_buy_max_day, how='left',on=['brand_id'])
    item_feat=pd.merge(item_feat,brand_frac_max_day, how='left',on=['brand_id'])
    item_feat.brand_max_click.fillna(0, inplace=True)
    item_feat.brand_max_buy.fillna(0, inplace=True)
    item_feat.brand_max_frac.fillna(0, inplace=True)
    item_feat.brand_max_click_delt_days.fillna(never_buy_delt_day, inplace=True)
    item_feat.brand_max_buy_delt_days.fillna(never_buy_delt_day, inplace=True)
    item_feat.brand_max_frac_delt_days.fillna(never_buy_delt_day, inplace=True)
    del brand_click_max_day
    del brand_buy_max_day
    del brand_frac_max_day
    gc.collect()
                                                                 
     #所有商品的总点击数和购买数
    total_buy_click_count=xact[['click', 'buy']].sum()
    #点击、购买转化率
    total_buy_click_frac=total_buy_click_count.buy/(total_buy_click_count.click*1.0)
    #某个品牌商品的点击数和购买数
    click_buy_count_in_brand=xact[['brand_id','click', 'buy']].groupby('brand_id').sum().reset_index()
    #某个品牌的转换率
    click_buy_count_in_brand['frac']=click_buy_count_in_brand.buy/(click_buy_count_in_brand.click*1.0)
    #有过交互的品牌数
    act_brand_count = click_buy_count_in_brand.shape[0]
    #某个分类商品的点击数和购买数
    click_buy_count_in_cate=xact[['cate_id','click', 'buy']].groupby('cate_id').sum().reset_index()
    #某个分类的转换率
    click_buy_count_in_cate['frac']=click_buy_count_in_cate.buy/(click_buy_count_in_cate.click*1.0)
    act_cate_count=click_buy_count_in_cate.shape[0]

    #商品点击数和购买数
    click_buy_count_by_item=xact[['spu_id','click', 'buy']].groupby('spu_id').sum().reset_index()
    #商品转化率
    click_buy_count_by_item['frac']=click_buy_count_by_item.buy/(click_buy_count_by_item.click*1.0)
    #有过交互的商品数
    act_item_count=click_buy_count_by_item.shape[0]
    #某个品牌下某个商品的点击数和购买数
    click_buy_item_count_by_brand=xact[['brand_id','spu_id','click', 'buy']].groupby(['brand_id','spu_id']).sum().reset_index()
    #某个品牌下某个商品的转化率
    click_buy_item_count_by_brand['frac']=click_buy_item_count_by_brand.buy/(click_buy_item_count_by_brand.click*1.0)
    #某个分分类下某个商品的点击数和购买数
    click_buy_item_count_by_cate=xact[['cate_id','spu_id','click', 'buy']].groupby(['cate_id','spu_id']).sum().reset_index()
    #某个分类下某个商品的转化率
    click_buy_item_count_by_cate['frac']=click_buy_item_count_by_cate.buy/(click_buy_item_count_by_cate.click*1.0)

    #某个分类下某个品牌的点击数和购买数
    click_buy_brand_count_by_cate=xact[['brand_id','cate_id','click', 'buy']].groupby(['cate_id','brand_id']).sum().reset_index()
    click_buy_brand_count_by_cate['frac']=click_buy_brand_count_by_cate.buy/(click_buy_brand_count_by_cate.click*1.0)
                                                                     
    click_buy_count_in_brand.columns=['brand_id','brand_click','brand_buy','brand_frac']
    click_buy_count_in_cate.columns=['cate_id','cate_click','cate_buy','cate_frac']
    click_buy_count_by_item.columns=['spu_id','item_click','item_buy','item_frac']
    click_buy_item_count_by_brand.columns=['brand_id','spu_id','item_in_brand_click','item_in_brand_buy','item_in_brand_frac']
    click_buy_item_count_by_cate.columns=['cate_id','spu_id','item_in_cate_click','item_in_cate_buy','item_in_cate_frac']
    click_buy_brand_count_by_cate.columns=['cate_id','brand_id','item_click_in_cb','item_buy_in_cb','item_frac_in_cb']

    print "merge item  features "
    item_feat=pd.merge(item_feat, click_buy_count_in_brand, how='left', on=['brand_id'])
    item_feat=pd.merge(item_feat, click_buy_count_in_cate, how='left', on=['cate_id'])
    item_feat=pd.merge(item_feat, click_buy_count_by_item, how='left', on=['spu_id'])
    item_feat=pd.merge(item_feat, click_buy_item_count_by_brand, how='left', on=['brand_id','spu_id'])
    item_feat=pd.merge(item_feat, click_buy_item_count_by_cate, how='left', on=['cate_id','spu_id'])
    item_feat=pd.merge(item_feat, click_buy_brand_count_by_cate, how='left', on=['brand_id','cate_id'])
    item_feat.fillna(0, axis=1,inplace=True)
    item_feat.drop(['cate_id', 'brand_id'], axis=1, inplace=True)
    del click_buy_count_in_brand
    del click_buy_count_in_cate
    del click_buy_count_by_item
    del click_buy_item_count_by_brand
    del click_buy_item_count_by_cate
    del click_buy_brand_count_by_cate
    gc.collect()

    print "generate user item features "
    '''
    #用户购买某商品的最后日期
    user_buy_item_last_date=xact[xact.buy>0][['uid','spu_id','date']].sort_values('date', ascending=False).groupby(['uid','spu_id'], as_index=False).first()
    #用户点击某商品的最后日期
    user_click_item_last_date=xact[xact.click>0][['uid','spu_id','date']].sort_values('date', ascending=False).groupby(['uid','spu_id'], as_index=False).first()
    #用户购买某分类的最后日期
    user_buy_cate_last_date=xact[xact.buy>0][['uid','cate_id','date']].sort_values('date', ascending=False).groupby(['uid','cate_id'], as_index=False).first()
    #用户点击某商品的最后日期
    user_click_cate_last_date=xact[xact.click>0][['uid','cate_id','date']].sort_values('date', ascending=False).groupby(['uid','cate_id'], as_index=False).first()
    #用户购买某品牌的最后日期
    user_buy_brand_last_date=xact[xact.buy>0][['uid','brand_id','date']].sort_values('date', ascending=False).groupby(['uid','brand_id'], as_index=False).first()
    #用户点击某品牌的最后日期
    user_click_brand_last_date=xact[xact.click>0][['uid','brand_id','date']].sort_values('date', ascending=False).groupby(['uid','brand_id'], as_index=False).first()
    
    #把日期转化为与预测日期间隔的天数
    user_buy_item_last_date['buy_item_last_date_delt']=user_buy_item_last_date.date.apply(lambda date: delt_days(date, predict_date))
    user_click_item_last_date['click_item_last_date_delt']=user_click_item_last_date.date.apply(lambda date: delt_days(date, predict_date))

    user_buy_cate_last_date['buy_cate_last_date_delt']=user_buy_cate_last_date.date.apply(lambda date: delt_days(date,predict_date))

    user_click_cate_last_date['click_cate_last_date_delt']=user_click_cate_last_date.date.apply(lambda date: delt_days(date, predict_date))

    user_buy_brand_last_date['buy_brand_last_date_delt']=user_buy_brand_last_date.date.apply(lambda date: delt_days(date, predict_date))

    user_click_brand_last_date['click_brand_last_date_delt']=user_click_brand_last_date.date.apply(lambda date: delt_days(date, predict_date))

    user_buy_item_last_date.drop('date', axis=1, inplace=True)
    user_click_item_last_date.drop('date', axis=1, inplace=True)
    user_buy_cate_last_date.drop('date', axis=1, inplace=True)
    user_click_cate_last_date.drop('date', axis=1, inplace=True)
    user_buy_brand_last_date.drop('date', axis=1, inplace=True)
    user_click_brand_last_date.drop('date', axis=1, inplace=True)
    
    print "merge user item features "
    user_item_feat=pd.merge(label[['uid','spu_id']], goods, how='left', on=['spu_id'])
    user_item_feat=pd.merge(user_item_feat, user_buy_item_last_date, how='left', on=['uid','spu_id'])
    user_item_feat=pd.merge(user_item_feat, user_click_item_last_date, how='left', on=['uid','spu_id'])
    user_item_feat=pd.merge(user_item_feat, user_buy_cate_last_date, how='left', on=['uid','cate_id'])
    user_item_feat=pd.merge(user_item_feat, user_click_cate_last_date, how='left', on=['uid','cate_id'])
    user_item_feat=pd.merge(user_item_feat, user_buy_brand_last_date, how='left', on=['uid','brand_id'])
    user_item_feat=pd.merge(user_item_feat, user_click_brand_last_date, how='left', on=['uid','brand_id'])
    del user_buy_item_last_date
    del user_click_item_last_date
    del user_buy_cate_last_date
    del user_click_cate_last_date
    del user_buy_brand_last_date
    del user_click_brand_last_date
    gc.collect()
    '''
    #用户对某商品的点击天数和购买次数
    user_act_item=xact[['uid','spu_id','click', 'buy']].groupby(['uid','spu_id']).sum().reset_index()
    #用户对某商品的转化率
    user_act_item['frac'] = user_act_item.buy/(user_act_item.click*1.0)
    #用户对某分类的点击天数和购买次数
    user_act_cate=xact[['uid','cate_id','click', 'buy']].groupby(['uid','cate_id']).sum().reset_index()
    #用户对某分类的转化率
    user_act_cate['frac'] = user_act_cate.buy/(user_act_cate.click*1.0)
    #user_act_cate['sub']= user_act_cate.click-user_act_cate.buy

    #用户对某品牌的点击天数和购买次数
    user_act_brand=xact[['uid','brand_id','click','buy']].groupby(['uid','brand_id']).sum().reset_index()
    #用户对某品牌的转化率
    user_act_brand['frac'] = user_act_brand.buy/(user_act_brand.click*1.0)
    #user_act_brand['sub']= user_act_brand.click-user_act_brand.buy
    
    user_act_item.columns=['uid','spu_id','user_act_item_click','user_act_item_buy','user_act_item_frac']
    user_act_cate.columns=['uid','cate_id','user_act_cate_click','user_act_cate_buy','user_act_cate_frac']
    user_act_brand.columns=['uid','brand_id','user_act_brand_click','user_act_brand_buy','user_act_brand_frac']
    
    print "merge user item features"
    #uis=pd.merge(label[['uid','spu_id']],act, how='inner', on=['uid','spu_id'])
    user_item_feat=pd.merge(label[['uid','spu_id']], goods, how='left', on=['spu_id'])
    user_item_feat=pd.merge(user_item_feat, user_act_item, how='left', on=['uid','spu_id'])
    user_item_feat=pd.merge(user_item_feat, user_act_cate, how='left', on=['uid','cate_id'])
    user_item_feat=pd.merge(user_item_feat, user_act_brand, how='left', on=['uid','brand_id'])
    user_item_feat.fillna(0, inplace=True)
    user_item_feat.drop(['cate_id', 'brand_id'], axis=1, inplace=True)
    del user_act_item
    del user_act_cate
    del user_act_brand
    gc.collect()

    
    print "generate user item sub features "
    '''
    user_item_feat['sub_item_click_buy_day']=user_item_feat.click_item_last_date_delt-user_item_feat.buy_item_last_date_delt
    user_item_feat['sub_cate_click_buy_day']=user_item_feat.click_cate_last_date_delt-user_item_feat.buy_cate_last_date_delt
    user_item_feat['sub_brand_click_buy_day']=user_item_feat.click_brand_last_date_delt-user_item_feat.buy_brand_last_date_delt
    user_item_feat['sub_item_click_cate_buy_day']=user_item_feat.click_item_last_date_delt-user_item_feat.buy_cate_last_date_delt
    user_item_feat['sub_item_click_brand_buy_day']=user_item_feat.buy_item_last_date_delt-user_item_feat.buy_brand_last_date_delt
    '''
    user_item_feat['sub_item_click_buy']=user_item_feat.user_act_item_click-user_item_feat.user_act_item_buy
    user_item_feat['sub_cate_click_buy']=user_item_feat.user_act_cate_click-user_item_feat.user_act_cate_buy
    user_item_feat['sub_brand_click_buy']=user_item_feat.user_act_brand_click-user_item_feat.user_act_brand_buy
    user_item_feat['sub_cate_click_item_buy']=user_item_feat.user_act_cate_click-user_item_feat.user_act_item_buy
    user_item_feat['sub_brand_click_item_buy']=user_item_feat.user_act_brand_click-user_item_feat.user_act_item_buy
    user_item_feat['sub_item_buy_cate_buy']=user_item_feat.user_act_item_buy-user_item_feat.user_act_cate_buy

    del xact
    gc.collect()
    print "return features "
    return (item_feat, user_feat, user_item_feat)

#提取预测日期前1天，前2天，前3天，前一周，前一个月和从一月份开始用户行为数据的特征
def get_all_feat(actions, goods, feat_begin_date, feat_end_date, label):
    day1=get_date_by_days(feat_end_date, 1)
    day2=get_date_by_days(feat_end_date, 2)
    day3=get_date_by_days(feat_end_date, 3)
    week1=get_date_by_days(feat_end_date, 7)
    month1=get_date_by_days(feat_end_date, 30)

    print day1, day2, day3, week1, month1
       
    print 'all days feature'
    ifeat_all,ufeat_all,uifeat_all= get_full_global_feat(actions, goods, feat_begin_date,feat_end_date, label)
    ifeat_1month,ufeat_1month,uifeat_1month= get_full_global_feat(actions, goods, month1,feat_end_date,label)

    ifeat_1day,ufeat_1day,uifeat_1day= get_full_global_feat(actions, goods, day1,feat_end_date,label)
    ifeat_3day,ufeat_3day,uifeat_3day= get_full_global_feat(actions, goods, day3,feat_end_date,label)
    ifeat_1week,ufeat_1week,uifeat_1week= get_full_global_feat(actions, goods, week1,feat_end_date,label)
         
    print 'merge item feature'        
    item_feat=pd.merge(ifeat_all, ifeat_1month, how='left', on=['spu_id'],suffixes=('_all','_1month'))
    item_feat=pd.merge(item_feat, ifeat_1week, how='left', on=['spu_id'],suffixes=('','_1week'))

    item_feat=pd.merge(item_feat, ifeat_3day, how='left', on=['spu_id'],suffixes=('','_3day'))
    item_feat=pd.merge(item_feat, ifeat_1day, how='left', on=['spu_id'],suffixes=('','_1day'))
    print ifeat_all.shape,ifeat_1week.shape,ifeat_3day.shape,ifeat_1day.shape,  item_feat.shape,

    item_feat.fillna(0, inplace=True)
        
    del ifeat_all
    del ifeat_1month
    del ifeat_1week
    del ifeat_3day
    del ifeat_1day
    gc.collect()
    gc.collect()

    print 'merge user feature'
    user_feat=pd.merge(ufeat_all, ufeat_1month, how='left', on=['uid'],suffixes=('_all','_1month'))
    user_feat=pd.merge(user_feat, ufeat_1week, how='left', on=['uid'],suffixes=('','_1week'))
    user_feat=pd.merge(user_feat, ufeat_3day, how='left', on=['uid'],suffixes=('','_3day'))
    user_feat=pd.merge(user_feat, ufeat_1day, how='left', on=['uid'],suffixes=('','_1day'))
    user_feat.fillna(0, inplace=True)
    del ufeat_all
    del ufeat_1month
    del ufeat_1week
    del ufeat_3day
    del ufeat_1day

    gc.collect()

    print 'merge user item pair feature'
    ui_feat=pd.merge(uifeat_all, uifeat_1month, how='left', on=['uid','spu_id'],suffixes=('_all','_1month'))

    ui_feat=pd.merge(ui_feat, uifeat_1week, how='left', on=['uid','spu_id'],suffixes=('','_1week'))

    ui_feat=pd.merge(ui_feat, uifeat_3day, how='left', on=['uid','spu_id'],suffixes=('','_3day'))
    ui_feat=pd.merge(ui_feat, uifeat_1day, how='left', on=['uid','spu_id'],suffixes=('','_1day'))
    ui_feat.fillna(0, inplace=True)
    del uifeat_all
    del uifeat_1month
    del uifeat_1week
    del uifeat_1day
    del uifeat_3day
    gc.collect()
    return(item_feat,user_feat,ui_feat)
	
#构建训练集的user-item label, 用户行为数据正负样本1：100，需对负样本进行采样，使正负样本在1：2左右	
def get_label(actions, begin_date,end_date, sample_rate=0.02):
    
    act=actions[(actions.date>begin_date) & (actions.date<=end_date)]

    buy_pair_label=act[['uid', 'spu_id','buy']].groupby(['uid', 'spu_id']).sum().reset_index()
    buy_pair_label.loc[buy_pair_label.buy>0,'label']=1
    buy_pair_label.fillna(0, inplace=True)
    buy_pair_label.drop('buy', axis=1, inplace=True)
   
    n_label=buy_pair_label[buy_pair_label.label==0]
    p_label=buy_pair_label[buy_pair_label.label==1]

    #对负样本采样
    if sample_rate < 1.0:
        n_label=n_label.sample(frac=sample_rate)
        print n_label.shape
        
    label=pd.concat([p_label, n_label])
    label.fillna(0, inplace=True)
    label.drop_duplicates(inplace=True)
    print label.label.value_counts()
    label = shuffle(label)
    
    del act
    gc.collect()
    
    return label

#创建用于模型训练的特征集和label	
def get_x_y(actions, goods, begin_date, end_date, label):
    print 'generate feat.'
    item_feat, user_feat, ui_feat=get_all_feat(actions, goods, begin_date, end_date, label)
    
    print 'generate x,y.'

    x=pd.merge(label, item_feat, how='left', on=['spu_id'],left_index=True)
    del item_feat
    gc.collect()
    
    x=pd.merge(x, user_feat, how='left', on=['uid'],left_index=True)
    del user_feat
    gc.collect()
   
    x=pd.merge(x, ui_feat, how='left', on=['uid','spu_id'], left_index=True)
    del ui_feat
    gc.collect()
 
    x.fillna(0, inplace=True)
    x.drop(['uid','spu_id','label'], axis=1,inplace=True)
    y=label.label.tolist()
    
    print 'generate ok'
    print x.shape
    return (x,y)
