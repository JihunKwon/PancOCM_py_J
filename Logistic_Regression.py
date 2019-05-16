import numpy as np
import matplotlib.pyplot  as plt
import scipy.io as spio

from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.model_selection import train_test_split

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
phase_BC = np.ones(len(mri_BC))
for i in range(mri_BC.shape[0]):
    if i < mri_B.shape[0]:
        phase_BC[i] = 1
    else:
        phase_BC[i] = 0


X_train, X_test, y_train, y_test = train_test_split(all_BC, phase_BC,test_size=0.2)

#Start training
logreg = LogisticRegression()
logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_test)
print("Training set score: {:.3f}".format(logreg.score(X_train,y_train)))
print("Test set score: {:.3f}".format(logreg.score(X_test,y_test)))

print(logreg.intercept_)
print(logreg.coef_)

w_0 = logreg.intercept_[0]
w_1 = logreg.coef_[0,0]
w_2 = logreg.coef_[0,1]

fig = plt.figure(figsize=(6,6))
x_range = [0,1.0]
plt.plot(x_range, list(map(lambda x: (-w_1 * x - w_0)/w_2, x_range)),'k--')
plt.scatter(mri_A, ocm_A, c='blue', marker='s', s=60, label='Before')
plt.scatter(X_train[y_train==1,0], X_train[y_train==1,1], c='red', marker='x', label='Shortly after, train')
plt.scatter(X_test[y_test==1, 0], X_test[y_test==1, 1], c='red', marker='o', s=60, label='Shortly after, test')
plt.scatter(X_train[y_train==0,0],X_train[y_train==0,1], c='green', marker='x', label='10min after, train')
plt.scatter(X_test[y_test==0, 0], X_test[y_test==0, 1], c='green', marker='o', s=60, label='10min after, test')
plt.xlim(-2.2,2.2)
plt.ylim(-2.2,2.2)
plt.xlabel('MRI, DVF')
plt.ylabel('OCM')
plt.title('Boundary Calculated by Logistic Regression')
plt.legend(loc='upper left')
#plt.show()
plt.savefig('Boundary.png')

#Draw ROC curve
fig = plt.figure(figsize=(6,6))
y_pred_prob = logreg.predict_log_proba(X_test)[:,1]
fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred_prob)
auc = metrics.auc(fpr, tpr)

plt.plot([0,1], [0,1], 'k--')
plt.plot(fpr, tpr, label='ROC curve (area = %.2f)'%auc)
plt.legend()
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Logistic Regression ROC Curve')
#plt.show()
plt.savefig('ROC.png')