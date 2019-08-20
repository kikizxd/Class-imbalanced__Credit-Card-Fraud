# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from sklearn.metrics import recall_score,confusion_matrix 
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
import itertools
import warnings
warnings.filterwarnings("ignore")

#读取信用卡数据
data = pd.read_csv("creditcard.csv",encoding='utf-8')


# 数据归一化
from sklearn.preprocessing import StandardScaler
data['normAmount'] = StandardScaler().fit_transform(data['Amount'].values.reshape(-1, 1))


#数据划分成LABEL列和特征列
X_data = data.drop(['Class','Time','Amount'],axis = 1)	#删除LABEL及其它非特征列
y = data.Class

#查看样本类别分布
from collections import Counter
print(Counter(y))
# Counter({0: 37027, 1: 227})

#更直观地，可以绘制饼图
import matplotlib.pyplot as plt

plt.axes(aspect='equal')
counts = data.Class.value_counts() #统计LABEL中各类别的频数
plt.pie(x = counts, #绘制数据
        labels = pd.Series(counts.index).map({0:'normal',1:'cheat'}),  #添加文字标签
        autopct='%.2f%%'    #设置百分比的格式，这里保留两位小数
        )
plt.show()  #显示图形

#数据划分成训练集和测试集
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X_data,y,test_size=0.3) 

#解决类别不均衡问题，SMOTE算法
from imblearn.over_sampling import SMOTE
sm = SMOTE(random_state=123)
X_train_sm, y_train_sm = sm.fit_sample(X_train, y_train)

X_train_sm = pd.DataFrame(X_train_sm)
y_train_sm = pd.DataFrame(y_train_sm)

#查看过采样之后的类别分布
print(Counter(y_train_sm))
#Counter({0: 37027, 1: 37027})


#========================逻辑回归模型=====================================
# 指定不同的惩罚系数，利用交叉验证找到最合适的参数，打印每个结果
def printing_Kfold_scores(X_train_data,Y_train_data):
    fold = KFold(5,shuffle=False)
    print(fold)
    c_param_range = [0.01,0.1,1,10,100]

    results_table = pd.DataFrame(index=range(len(c_param_range),2),
                                 columns=['C_Parameter','Mean recall score'])
    results_table['C_Parameter'] = c_param_range
    j=0
    for c_param in c_param_range:
        print('c_param:',c_param)
        recall_accs = []
        for iteration,indices in enumerate(fold.split(X_train_data)):
            lr = LogisticRegression(C = c_param, penalty = 'l1')
            lr.fit(X_train_data.iloc[indices[0],:],
                   Y_train_data.iloc[indices[0],:].values.ravel())
            
            Y_pred = lr.predict(X_train_data.iloc[indices[1],:].values)

            recall_acc = recall_score(Y_train_data.iloc[indices[1],:].values,Y_pred)
            recall_accs.append(recall_acc)

            print ('Iteration:',iteration,'recall_acc:',recall_acc)

        print ('Mean recall score',np.mean(recall_accs))
        results_table.loc[j,'Mean recall score'] = np.mean(recall_accs)
        print('----------')
        j+=1
    print(type(results_table['Mean recall score']))
    print(results_table['Mean recall score'])
    results_table['Mean recall score'] = results_table['Mean recall score'].astype('float64')

    best_c = results_table.loc[results_table['Mean recall score'].idxmax()]['C_Parameter']
    print ('best_c is :',best_c)
    return best_c

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
    
    
best_c = printing_Kfold_scores(X_train_sm,y_train_sm)
lr = LogisticRegression(C = best_c, penalty = 'l2')
lr.fit(X_train_sm,y_train_sm.values.ravel())
y_pred = lr.predict_proba(X_test.values)

## Compute confusion matrix
#cnf_matrix = confusion_matrix(y_test,y_pred)
#np.set_printoptions(precision=2)
#print("Recall metric in the testing dataset: ", float(cnf_matrix[1,1])/(cnf_matrix[1,0]+cnf_matrix[1,1]))
#
#class_names = [0,1]
#plot_confusion_matrix(cm=cnf_matrix,classes=class_names,title='Confusion matrix')


#阈值移动方法
#默认阈值一般为0.5
thresholds = [0.5,0.6,0.7,0.8,0.9]
plt.figure(figsize=(10,10))

m = 1
for i in thresholds:
    y_test_predictions_high_recall = y_pred[:,1] > i
    
    plt.subplot(3,3,m)
    m += 1
    
    cnf_matrix = confusion_matrix(y_test,y_test_predictions_high_recall)
    np.set_printoptions(precision=2)

    print ("Recall:{}".format(cnf_matrix[1,1]/(cnf_matrix[1,0]+cnf_matrix[1,1])))
   
    class_names = [0,1]
    plot_confusion_matrix(cnf_matrix
                          , classes=class_names
                          , title='Threshold >= %s'%i) 
