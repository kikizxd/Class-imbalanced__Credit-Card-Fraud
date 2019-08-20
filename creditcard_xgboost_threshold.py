# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score,classification_report,roc_curve,auc,confusion_matrix
import itertools
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

#读取信用卡数据
data = pd.read_csv("creditcard.csv",encoding='utf-8')	#文末附有数据

# 数据归一化
from sklearn.preprocessing import StandardScaler
data['normAmount'] = StandardScaler().fit_transform(data['Amount'].values.reshape(-1, 1))

data = data.drop(['Time','Amount'],axis = 1)
#数据划分成LABEL列和特征列
X_data = data.drop(['Class'],axis = 1)	#删除LABEL及其它非特征列
y = data.Class

#查看样本类别分布
from collections import Counter
print(Counter(y))
# Counter({0: 37027, 1: 227})

#更直观地，可以绘制饼图
plt.axes(aspect='equal')
counts = data.Class.value_counts() #统计LABEL中各类别的频数
plt.pie(x = counts, #绘制数据
        labels = pd.Series(counts.index).map({0:'normal',1:'cheat'}),  #添加文字标签
        autopct='%.2f%%'    #设置百分比的格式，这里保留两位小数
        )
plt.show()  #显示图形

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X_data,y,test_size=0.3,
                                                 random_state=123) 
Train,Test = train_test_split(data,test_size=0.3,random_state=123) 



"""======================XGBoost模型分类==================="""
import xgboost
##	原数据结果	##
xgb = xgboost.XGBClassifier()  #添加系数scale_pos_weight=10或100即为调整类别的权
# 使用非平衡的训练数据集拟合模型
xgb.fit(X_train,y_train)
# 基于拟合的模型对测试数据集进行预测
y_pred = xgb.predict(X_test)
# 返回模型的预测效果
print('模型的准确率为：\n',accuracy_score(y_test, y_pred))
print('模型的评估报告：\n',classification_report(y_test, y_pred))
# 计算用户流失的概率值
y_score = xgb.predict_proba(X_test)[:,1]
fpr,tpr,threshold = roc_curve(y_test, y_score)
# 计算AUC的值
roc_auc = auc(fpr,tpr)
print('ROC curve (area = %0.2f)' % roc_auc)

y_pred_proba = xgb.predict_proba(X_test) #之后阈值移动要用到

#绘制混淆矩阵图
def plot_confusion_matrix(cm, classes,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=0)
    plt.yticks(tick_marks, classes)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')



#阈值移动方法
#默认阈值一般为0.5
#thresholds = [0.45,0.47,0.49,0.51,0.53,0.55]
thresholds = [0.4,0.5,0.6,0.7,0.8,0.9,0.95]
plt.figure(figsize=(10,10))

m = 1
for i in thresholds:
    y_test_predictions_high_recall = y_pred_proba[:,1] > i
    
    plt.subplot(3,3,m)
    m += 1
    
    cnf_matrix = confusion_matrix(y_test,y_test_predictions_high_recall)
    np.set_printoptions(precision=2)

    print ("Recall:{}".format(cnf_matrix[1,1]/(cnf_matrix[1,0]+cnf_matrix[1,1])))
   
    class_names = [0,1]
    plot_confusion_matrix(cnf_matrix
                          , classes=class_names
                          , title='Threshold >= %s'%i) 