# -- coding: utf-8 --
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import gc  
import datetime
from sklearn import metrics
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score

#根据阈值转化为0,1	
def get_pred_from_threshold(pred, threshold):
    pred_df=pd.DataFrame(pred.copy())
    pred_df.columns=['y']
    pred_df.loc[pred_df.y<threshold, 'y']=0
    pred_df.loc[pred_df.y>=threshold, 'y']=1
    print pred_df.y.value_counts()
    return pred_df.y.tolist()
	
#画出prc、roc曲线
def display_score(y, scores, t=0.5, draw=False):

	if draw:
		precision, recall, thresholds = precision_recall_curve( y, scores)
		average_precision = average_precision_score(y, scores, average="micro")
    
		lw = 2
        
		# Plot Precision-Recall curve
		plt.clf()
		plt.plot(recall, precision, lw=lw, color='navy',
			label='Precision-Recall curve')
		plt.xlabel('Recall')
		plt.ylabel('Precision')
		plt.ylim([0.0, 1.05])
		plt.xlim([0.0, 1.0])
		plt.title('Precision-Recall: ROC={0:0.5f}'.format(average_precision))
		plt.legend(loc="lower left")
		plt.show()

	fpr, tpr, thresholds = metrics.roc_curve(y, scores, pos_label=1)
	auc=metrics.auc(fpr, tpr)
	print auc

	if draw:
		plt.figure()
		plt.plot(fpr, tpr, color='darkorange',lw=lw, label='auc curve (area = %0.5f)' % auc)
		plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
		plt.xlim([0.0, 1.0])
		plt.ylim([0.0, 1.05])
		plt.xlabel('False Positive Rate')
		plt.ylabel('True Positive Rate')
		plt.title('Receiver operating characteristic (area = %0.5f)'% auc)
		plt.legend(loc="lower right")
		plt.show()

	print 'rmse: %0.5f'%np.sqrt(metrics.mean_squared_error(y,scores))
    
	pred_label=get_pred_from_threshold(scores, t)
	print 'label rmse: %0.5f'%np.sqrt(metrics.mean_squared_error(y,pred_label))
	print 'accuracy: %0.5f'%metrics.accuracy_score(y, pred_label)
	print 'precision: %0.5f'%metrics.precision_score(y, pred_label)
	print 'recall: %0.5f'%metrics.recall_score(y, pred_label)
	print 'f1: %0.5f'%metrics.f1_score(y, pred_label)
    