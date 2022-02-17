#!/usr/bin/env python
# coding: utf-8

# Machine Learning Project : Classification on four grades of liver cancer (RN&LOW, High, HCC in DN, Progressed HCC).     
# From 300 high-resolution pathology images, extract 1024*1024 patches based on binary masks following the way of computer vision.     
# Quantitive cell features from the patches were obtained through two softwares - CellProfiler and QuPath.     

# Load raw data

# In[1]:


path_data = '/data5/cy_SGER/project/case_raw.csv'


# In[2]:


import pandas as pd
df = pd.read_csv(path_data)
df.head()


# The statistical analysis : correlation

# In[4]:


df = df.dropna()
df = df.rename(columns={'Class':'label'})
df.replace('hcc', 4, inplace=True)
df.replace('dn', 3, inplace=True)
df.replace('hg', 2, inplace=True)
df.replace('rnlow', 1, inplace=True)
df.replace('normal', 1, inplace=True)


# In[5]:


corr = df.corr()
corr_top10 = corr.nlargest(10, 'label')
corr_top10 = corr_top10[list(corr_top10.index)]

# In[9]:


import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
sns.set(style="darkgrid")

mask = np.triu(np.ones_like(corr_top10, dtype=bool))

f, ax = plt.subplots(figsize=(10, 10))
cmap = sns.diverging_palette(220, 10, as_cmap=True)

sns.heatmap(corr_top10, cmap=cmap, ax=ax, mask=mask,
            annot=True,
            linewidths=.9,
           annot_kws={"size": 10},
           square = True, vmin =-0.5 , vmax = 1)
plt.show()


# Standard Scaler : mean -->0,          
# variance --> unit variance

# In[12]:


import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler


# In[13]:


dt_tot = df.dropna()


# In[16]:


temp_x = dt_tot[dt_tot.columns[2:]]
temp_x.index.name ='idx'
scaler = StandardScaler()
scaler.fit(temp_x)
transformed_x = scaler.transform(temp_x)


# In[17]:


transformed_d = pd.DataFrame(transformed_x, columns=dt_tot.columns[2:])


# In[18]:


temp_cols = dt_tot[dt_tot.columns[0:2]]


# In[19]:


scaled = pd.merge(temp_cols, transformed_d, left_index=True, right_index=True, how='left')


# In[20]:


scaled.describe()


# Feature Selection : offers a effective way to overcome this challenge by eliminating redundant and irrelevant data

# In[24]:


from sklearn.metrics import classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression

data = scaled
data.head()


# In[22]:


data.describe()


# In[23]:


X_df = data.values[:,2:]
y_df = data['label']


# In[26]:


col_path = '/data5/cy_SGER/project/col_name.csv'
col_li = pd.read_csv(col_path)
cols_name = col_li['col_nm']


# In[32]:


from sklearn.feature_selection import SelectFromModel
selector = SelectFromModel((LogisticRegression(max_iter=9999)),threshold=None, prefit=False, norm_order=1, max_features=None).fit(X_df,y_df)
selector.estimator_.coef_
sel_cols = selector.get_support()


# In[55]:


cnt=0
sel = sel_cols
for i in sel:
    print(np.array(cols_name)[cnt], i) #print selected features
    cnt+=1


# In[52]:


data = data[['label','PID','Mean_Nucleus_AreaShape_Compactness',	'Mean_Nucleus_AreaShape_Eccentricity',	'Mean_Nucleus_AreaShape_Extent',	'Mean_Nucleus_AreaShape_FormFactor',	'Mean_Nucleus_AreaShape_Orientation',	'Mean_Nucleus_AreaShape_Perimeter',	'Mean_Nucleus_AreaShape_Solidity',	'Mean_area_Parent_Nucleus',	'Median_Nucleus_AreaShape_Compactness',	'Median_Nucleus_AreaShape_Eccentricity',	'Median_Nucleus_AreaShape_Extent',	'Median_Nucleus_AreaShape_FormFactor',	'Median_Nucleus_AreaShape_MajorAxisLength',	'Median_Nucleus_AreaShape_MaxFeretDiameter',	'Median_Nucleus_AreaShape_MaximumRadius',	'Median_Nucleus_AreaShape_MeanRadius',	'Median_Nucleus_AreaShape_MedianRadius',	'Median_Nucleus_AreaShape_MinFeretDiameter',	'Median_Nucleus_AreaShape_MinorAxisLength',	'Median_Nucleus_AreaShape_Orientation',	'Median_Nucleus_AreaShape_Perimeter',	'Median_area_Parent_Nucleus',	'StDev_Nucleus_AreaShape_Compactness',	'StDev_Nucleus_AreaShape_Eccentricity',	'StDev_Nucleus_AreaShape_Extent',	'StDev_Nucleus_AreaShape_FormFactor',	'StDev_Nucleus_AreaShape_MajorAxisLength',	'StDev_Nucleus_AreaShape_MaxFeretDiameter',	'StDev_Nucleus_AreaShape_MaximumRadius',	'StDev_Nucleus_AreaShape_MeanRadius',	'StDev_Nucleus_AreaShape_MedianRadius',	'StDev_Nucleus_AreaShape_Orientation',	'StDev_Nucleus_AreaShape_Solidity',	'StDev_area_Parent_Nucleus',	'M_Nucleus_Eccentricity',	'M_Nucleus_Eosin_OD_sum',	'M_Cell_Eccentricity',	'M_Cytoplasm_Eosin_OD_max',	'Q1_Nucleus_Area',	'Q1_Nucleus_Circularity',	'Q1_Nucleus_Max_caliper',	'Q1_Nucleus_Eccentricity',	'Q1_Nucleus_Hematoxylin_OD_range',	'Q1_Nucleus_Eosin_OD_sum',	'Q1_Nucleus_Eosin_OD_min',	'Q1_Cell_Max_caliper',	'Q1_Cell_Min_caliper',	'Q1_Cell_Eccentricity',	'Q1_Cell_Hematoxylin_OD_min',	'Q1_Cell_Eosin_OD_std_dev',	'Q1_Cytoplasm_Hematoxylin_OD_std_dev',	'Q1_Cytoplasm_Hematoxylin_OD_min',	'Q1_Cytoplasm_Eosin_OD_mean',	'Q1_Cytoplasm_Eosin_OD_std_dev',	'Q1_Cytoplasm_Eosin_OD_max',	'Q1_Cytoplasm_Eosin_OD_min',	'Q1_Nucleus_Cell_area_ratio',	'Q3_Nucleus_Circularity',	'Q3_Nucleus_Eccentricity',	'Q3_Nucleus_Hematoxylin_OD_std_dev',	'Q3_Nucleus_Hematoxylin_OD_min',	'Q3_Nucleus_Eosin_OD_sum',	'Q3_Cell_Area',	'Q3_Cell_Circularity',	'Q3_Cell_Min_caliper',	'Q3_Cell_Eccentricity',	'Q3_Cell_Eosin_OD_min',	'Q3_Cytoplasm_Eosin_OD_max',	'Q3_Nucleus_Cell_area_ratio',	'S_Nucleus_Area',	'S_Nucleus_Circularity',	'S_Nucleus_Max_caliper',	'S_Nucleus_Min_caliper',	'S_Nucleus_Eccentricity',	'S_Nucleus_Hematoxylin_OD_mean',	'S_Nucleus_Hematoxylin_OD_sum',	'S_Nucleus_Hematoxylin_OD_std_dev',	'S_Nucleus_Hematoxylin_OD_min',	'S_Nucleus_Eosin_OD_sum',	'S_Nucleus_Eosin_OD_std_dev',	'S_Nucleus_Eosin_OD_max',	'S_Nucleus_Eosin_OD_min',	'S_Cell_Circularity',	'S_Cell_Min_caliper',	'S_Cell_Eccentricity',	'S_Cell_Hematoxylin_OD_mean',	'S_Cell_Hematoxylin_OD_std_dev',	'S_Cell_Eosin_OD_max',	'S_Cell_Eosin_OD_min',	'S_Cytoplasm_Hematoxylin_OD_mean',	'S_Cytoplasm_Hematoxylin_OD_std_dev',	'S_Cytoplasm_Hematoxylin_OD_max',	'S_Cytoplasm_Eosin_OD_mean',	'S_Cytoplasm_Eosin_OD_std_dev',	'S_Cytoplasm_Eosin_OD_max',	'S_Cytoplasm_Eosin_OD_min',	'S_Nucleus_Cell_area_ratio'
]]


# Machine Learning

# split train and test set randomly

# In[56]:


X= data.values[:,2:]
y= data['label']


# In[57]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, 
                                                    y, 
                                                    test_size=0.33, 
                                                    random_state=0)


# In[58]:


pd.DataFrame(y_test)


# In[59]:


pd.DataFrame(X_test)


# 5-fold Cross-Validation    
# StratifiedKFold :  preserving the percentage of samples for each classs

# In[60]:


from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score

from sklearn.svm import LinearSVC
model = LinearSVC()

skf = StratifiedKFold(n_splits=5,
           shuffle=True,
            random_state =1)

scores = cross_val_score(model, X, y, cv = skf)
print("Scores: ", scores)
print("Mean of the scores: ", scores.mean())


# Grid Search : Tuning the hyper-parameters by the parameter grid and cross-validation

# In[61]:


from sklearn.svm import LinearSVC
from sklearn.model_selection import GridSearchCV


param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100],
    'penalty': ['l1', 'l2'],
    'loss': ['hinge', 'squared_hinge'],
    'dual': [True, False],
    'multi_class': ['ovr', 'crammer_singer']
   }


# In[62]:


grid_search = GridSearchCV(LinearSVC(), param_grid=param_grid, n_jobs=-1,cv=skf)


# In[63]:


grid_search.fit(X_train, y_train)


# In[64]:


print("best parameters:", grid_search.best_params_)
print("best score: {:.4f}".format(grid_search.best_score_)) 
print("best estimator:\n", grid_search.best_estimator_)


# In[65]:


pd.set_option('display.max_columns', None)
cv_results = np.transpose(pd.DataFrame(grid_search.cv_results_))
cv_results 


# In[66]:


print("test score: {:.4f}".format(grid_search.score(X_test, y_test)))


# In[71]:


best_Classifier = grid_search.best_estimator_
best_Classifier.fit (X_train, y_train)
print ("test score by grid search: {:.4f} ".format(best_Classifier.score(X_test, y_test)))


# In[72]:


from sklearn.metrics import classification_report, confusion_matrix
best_Classifier.score(X_test, y_test)
Y_Pred = best_Classifier.predict(X_test)
cm = confusion_matrix(y_test, Y_Pred)
print (cm)
print (classification_report(y_test,Y_Pred))


# In[76]:


clf = LinearSVC()
clf.fit(X_train, y_train)
print ("test score by basic model: {:.4f}".format(clf.score(X_test, y_test)))


# In[77]:


clf.score(X_test, y_test)
Y_Pred2 = clf.predict(X_test)
cm = confusion_matrix(y_test, Y_Pred2)
print (cm)
print (classification_report(y_test,Y_Pred2))


# In[78]:


dfl_path = '/data5/cy_SGER/dfl_test_list.csv'
dfl = pd.read_csv(dfl_path)
df_test_PID = dfl['PID']
GTlist_= dfl['class']


# In[79]:


cnt=0
ret = clf.predict(X_test)
for ret_ in ret:
    print (cnt, np.array(df_test_PID)[cnt], ', {}, %d'.format(np.array(GTlist_)[cnt]) % (int(ret_)))
    cnt+=1


# OneVsRestClassifier : simplify the classification and visualizing

# In[80]:


from sklearn.datasets import make_classification
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

n_classes = 4

X = (data.values[:,2:])
y = data['label']
y = label_binarize(y, classes=[1,2,3,4])
X_train, X_test, y_train, y_test = train_test_split(X, 
                                                    y, 
                                                    test_size=0.33, 
                                                    random_state=0)

from sklearn.svm import SVC


# classifier
clf = OneVsRestClassifier(SVC(probability=True))
clf.fit(X_train, y_train)
y_score = clf.predict_proba(X_test)
#probability=True

# ROC & AUC
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Plot of a ROC curve for a specific class
plt.figure(figsize=(15, 5))
for idx, i in enumerate(range(n_classes)):
    plt.subplot(151+idx)
    plt.plot(fpr[i], tpr[i], label='ROC curve (area = %0.2f)' % roc_auc[i])
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('SVM : Class %0.0f' % (idx+1) )
    plt.legend(loc="lower right")
plt.show()

print("roc_auc_score: ", roc_auc_score(y_test, y_score, multi_class='raise'))


# In[82]:


import numpy as np
import matplotlib.pyplot as plt
from itertools import cycle
n_classes = 4
lw = 3
from sklearn import svm, datasets
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from scipy import interp
from sklearn.metrics import roc_auc_score
fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

# First aggregate all false positive rates
all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

# Then interpolate all ROC curves at this points
mean_tpr = np.zeros_like(all_fpr)
for i in range(n_classes):
    mean_tpr += interp(all_fpr, fpr[i], tpr[i])

# Finally average it and compute AUC
mean_tpr /= n_classes

fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

# Plot all ROC curves
plt.figure()

import matplotlib.pylab as plt

colors = cycle(['gold','yellowgreen','teal','royalblue','midnightblue'])

for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=lw,
             label='ROC curve of class {0} (area = {1:0.2f})'
             ''.format(i+1, roc_auc[i]))
plt.rcParams["figure.figsize"] = (20,20)
plt.xticks(fontsize =30)
plt.yticks(fontsize =30)
plt.plot([0,1 ], [0, 1], 'k--', lw=lw)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate', fontsize = 30)
plt.ylabel('True Positive Rate', fontsize = 30)
plt.title('Receiver Operating Characteristics to multi-class : SVM',fontsize = 40, pad=20 )
plt.legend(loc="lower right", fontsize = 30)
plt.show()

