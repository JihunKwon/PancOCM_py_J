'''
#This script shows scikit-learn example of Classification by Logistic Regression and plotting ROC curve, for breast cancer data.

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
import matplotlib.pyplot as plt

logreg = LogisticRegression()
cancer = load_breast_cancer()
print(cancer.target)
X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, stratify=cancer.target, random_state=42)
logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_test)
print("Training set score: {:.3f}".format(logreg.score(X_train,y_train)))
print("Test set score: {:.3f}".format(logreg.score(X_test,y_test)))

#Draw ROC curve
from sklearn.metrics import roc_curve
y_pred_prob = logreg.predict_log_proba(X_test)[:,1]
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
plt.plot([0,1], [0,1], 'k--')
plt.plot(fpr, tpr, label='Logistic Regression')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Logistic Regression ROC Curve')
plt.show()
'''

from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as spio

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve

####### My original code ##########
plt.style.use('ggplot')

mat = spio.loadmat('C:\\Users\\Kwon\Documents\\MATLAB\\PCA\\OCM_Analysis\\Scatter_Standard_each_fromRaw.mat', squeeze_me=True)

mri_A = mat['stdz_mriA_sub']
mri_B = mat['stdz_mriB_sub']
mri_C = mat['stdz_mriC_sub']
ocm_A = mat['stdz_ocmA_sub']
ocm_B = mat['stdz_ocmB_sub']
ocm_C = mat['stdz_ocmC_sub']


#Connect data
mri_BC = np.append(mri_B, mri_C)
ocm_BC = np.append(ocm_B, ocm_C)

all_BC = np.zeros((2,mri_BC.shape[0]))
all_BC[0,:] = mri_BC[:]
all_BC[1,:] = ocm_BC[:]
all_BC = all_BC.T
#Create Answer
#phase_BC = [0] * mri_BC.shape[0]
phase_BC = np.ones(100)
for i in range(mri_BC.shape[0]):
    if i < mri_B.shape[0]:
        phase_BC[i] = 1
    else:
        phase_BC[i] = 0
########################################

np.random.seed(seed=0)
X_0 = np.random.multivariate_normal( [2,2],  [[2,0],[0,2]],  50 )
y_0 = np.zeros(len(X_0))

X_1 = np.random.multivariate_normal( [6,7],  [[3,0],[0,3]],  50 )
y_1 = np.ones(len(X_1))

X = np.vstack((X_0, X_1))
y = np.append(y_0, y_1)

X_train, X_test, y_train, y_test = train_test_split(X, phase_BC, test_size=0.3)

a = X_train[y_train==0, 0]
plt.scatter(X_train[y_train==0, 0], X_train[y_train==0, 1], c='red', marker='x', label='train 0')
plt.scatter(X_train[y_train==1, 0], X_train[y_train==1, 1], c='blue', marker='x', label='train 1')
plt.scatter(X_test[y_test==0, 0], X_test[y_test==0, 1], c='red', marker='o', s=60, label='test 0')
plt.scatter(X_test[y_test==1, 0], X_test[y_test==1, 1], c='blue', marker='o', s=60, label='test 1')

plt.legend(loc='upper left')
plt.show()