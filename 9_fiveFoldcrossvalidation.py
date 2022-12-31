import matplotlib.pyplot as plt
from sklearn.metrics import auc,roc_curve
import numpy as np
#from get_data import get_train_data
from sklearn.utils import shuffle
from sklearn import metrics
from sklearn.model_selection import KFold
from xgboost import XGBClassifier
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import precision_recall_curve
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.ensemble import  AdaBoostClassifier,VotingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import tree 
plt.style.use('_classic_test')
plt.figure(dpi=300,figsize=(8,5))
def train(X, Y):
    X, Y = shuffle(X, Y, random_state=1)
    clf1 = SVC(C = 5.0, probability=True)
    clf2 = GradientBoostingClassifier(n_estimators=500,learning_rate=0.1)
    clf3 = RandomForestClassifier(n_estimators=800, random_state=1)
    clf4 = XGBClassifier(n_estimators=300, learning_rate=0.1, random_state=1)
    clf = VotingClassifier(estimators=[
    ('svm',clf1),    
    ('gb',clf2), 
    ('rf',clf3),
    ('xgboost',clf4),    
    ],    
    voting='soft')
    kf = KFold(n_splits=5)    
    AUC_list = []
    p_list = []
    r_list = []
    f1_list = []
    AUPR_list = []    
    fp = []
    tp = []
    ax = plt.subplot(1,1,1)
    for train_index, test_index in kf.split(X, Y):
        X_train = X[train_index]
        Y_train = Y[train_index]
        X_test = X[test_index]
        Y_test = Y[test_index]
        clf.fit(X_train, Y_train)
        predict_value = clf.predict_proba(X_test)[:, 1]
        AUC = metrics.roc_auc_score(Y_test, predict_value)
        fpr,tpr,thresholds = roc_curve(Y_test, predict_value)
        fp.append(fpr)
        tp.append(tpr)        
        ax.plot(fpr,tpr,lw=0.8,alpha=0.7,label="ROC Fold %d (area=%.4f)"%(i,AUC))
        precision, recall, _ = precision_recall_curve(Y_test, predict_value)
        AUCPR = auc(recall, precision)
        AUPR_list.append(AUCPR)
        p = precision_score(Y_test, predict_value.round())
        p_list.append(p)
        r = recall_score(Y_test, predict_value.round())
        r_list.append(r)
        f1 = f1_score(Y_test, predict_value.round())
        f1_list.append(f1)
        AUC_list.append(AUC)        
    ax.axis([0, 1, 0, 1.05])
    ax.legend(loc = 'lower right',fontsize=12)
    axins = ax.inset_axes((0.2, 0.45, 0.4, 0.3))
    for i in range(len(fp)):        
        axins.plot(fp[i],tp[i],lw=0.8,alpha=0.7)
    axins.axis([0.1, 0.35, 0.7, 0.9])
    return AUC_list,AUPR_list,p_list,r_list,f1_list
if __name__ == "__main__": 
    data = np.load('trainSamples.npz')
    X = data['X']
    y = data['y'] 
    X[np.isnan(X)]=0    
    AUC,AUPR,p,r,f1=train(X,y)


