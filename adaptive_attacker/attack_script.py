import random

import scipy.io as sio
from secml.adv.attacks import CAttackPoisoningSVM
from secml.data import CDataset
from secml.ml.classifiers import CClassifierSVM
from secml.ml.kernels import CKernelRBF
from secml.ml.peval.metrics import CMetricAccuracy
from sklearn import preprocessing, svm
from sklearn.datasets import load_svmlight_file
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from joblib import Memory
from adaptive_attacker.c_attack_poisoning_svm_lid import CAttackPoisoningSVMwithLID

import numpy as np

###########
# Variables
from utilities import get_lids_random_batch, get_kde, weight_calculation

C = 0.39685026299204973  # SVM regularization parameter
GAMMA = 0.7937005259840995
DATASET = 'omnet'
ATTACK = 'poison'
CV_SET = 1
###########
random_state = 1234
random_state = 4
random.seed(random_state)
np.random.seed(random_state)

print('starting')

n_tr = 280  # Number of training set samples
n_val = 53  # Number of validation set samples
n_ts = 53  # Number of test set samples

# poison 10%
poison_percentage = 0.1

LID_GAMMA = 0.1
BATCH_SIZE = 64

if DATASET == 'mnist_9_1' or DATASET == 'omnet':
    xtr_arrays = []
    ytr_arrays = []
    for i in range(1):
        mat_contents = sio.loadmat('../../data/{}/{}/training/training_{}.mat'.format(DATASET, ATTACK, i + 1))
        # training data
        xtr_arrays.append(mat_contents['Xtr'])
        # training labels
        ytr_arrays.append(mat_contents['Ytr'])
        # testing data
        xtr_arrays.append(mat_contents['Xtt'])
        # testing labels
        ytr_arrays.append(mat_contents['Ytt'])

    xtr = np.vstack(xtr_arrays)
    ytr = np.vstack(ytr_arrays)
else:
    # for lib-svm datasets
    def get_data():
        data = load_svmlight_file('../../data/libsvm-dataset/{}'.format(DATASET))
        return data[0], data[1]


    xtr, ytr = get_data()
    xtr = xtr.toarray()

idx = np.random.choice(xtr.shape[0], n_tr + n_val + n_ts, replace=False)
xtr = xtr[idx, :]
ytr = ytr[idx]

ytr = (ytr + 1) / 2
min_max_scaler = preprocessing.MinMaxScaler()
xtr = min_max_scaler.fit_transform(xtr)

x_train, xtt, y, ytt = train_test_split(xtr, ytr, test_size=n_ts, train_size=n_tr + n_val, random_state=random_state)
x_train, x_val, y, y_val = train_test_split(x_train, y, test_size=n_val, train_size=n_tr, random_state=random_state)

training_data = CDataset(x_train, y)
validation_data = CDataset(x_val, y_val)
test_data = CDataset(xtt, ytt)

del xtr
del ytr

metric = CMetricAccuracy()

clf = CClassifierSVM(kernel=CKernelRBF(gamma=GAMMA), C=C)

# We can now fit the classifier
clf.fit(training_data.X, training_data.Y)
print("Training of classifier complete!")
# Compute predictions on a test set
y_pred = clf.predict(test_data.X)

lb, ub = validation_data.X.min(), validation_data.X.max()  # Bounds of the attack space. Can be set to `None` for unbounded
n_poisoning_points = int(n_tr * poison_percentage)  # Number of poisoning points to generate

# Should be chosen depending on the optimization problem
solver_params = {
    'eta': 0.05,
    'eta_min': 0.05,
    'eta_max': None,
    'max_iter': 100,
    'eps': 1e-6
}
# Non-adaptive attacker #################################################################################
pois_attack = CAttackPoisoningSVM(classifier=clf,
                                  training_data=training_data,
                                  val=validation_data,
                                  lb=lb, ub=ub,
                                  solver_params=solver_params,
                                  random_seed=random_state)

pois_attack.n_points = n_poisoning_points
# Run the poisoning attack
print("Attack started...")
pois_y_pred, pois_scores, pois_ds, f_opt = pois_attack.run(test_data.X, test_data.Y)
print("Attack complete!")

# Evaluate the accuracy of the original classifier
acc = metric.performance_score(y_true=test_data.Y, y_pred=y_pred)
# Evaluate the accuracy after the poisoning attack
pois_acc = metric.performance_score(y_true=test_data.Y, y_pred=pois_y_pred)

print("Original accuracy on test set: {:.2%}".format(acc))
print("Accuracy after non adaptive attack on test set: {:.2%}".format(pois_acc))

# Adaptive attacker #################################################################################
clf = CClassifierSVM(kernel=CKernelRBF(gamma=GAMMA), C=C)
# We can now fit the classifier
clf.fit(training_data.X, training_data.Y)

pois_attack = CAttackPoisoningSVMwithLID(classifier=clf,
                                         training_data=training_data,
                                         val=validation_data,
                                         lb=lb, ub=ub,
                                         solver_params=solver_params,
                                         random_seed=random_state,
                                         lid_k=10,
                                         lid_cost_coefficient=0.2)
pois_attack.n_points = n_poisoning_points

# Run the poisoning attack
print("Attack started...")
pois_y_pred, pois_scores, pois_ds, f_opt, x_indices = pois_attack.run(test_data.X, test_data.Y)
print("Attack complete!")

# Evaluate the accuracy of the original classifier
acc = metric.performance_score(y_true=test_data.Y, y_pred=y_pred)
# Evaluate the accuracy after the poisoning attack
pois_acc = metric.performance_score(y_true=test_data.Y, y_pred=pois_y_pred)

print("Original accuracy on test set: {:.2%}".format(acc))
print("Accuracy after adaptive attack on test set: {:.2%}".format(pois_acc))

# Defender #################################################################################
WEIGHT_LB = 0.1
pos_density_normal = None
pos_density_fl = None
neg_density_normal = None
neg_density_fl = None

# distorted data
x_att = pois_ds.X.tondarray()

x_train[x_indices, :] = x_att

pos_indices = np.where(y == 1)[0]
neg_indices = np.where(y == 0)[0]

pos_data = x_train[pos_indices, :]
neg_data = x_train[neg_indices, :]

pos_lids = get_lids_random_batch(pos_data, LID_GAMMA, lid_type='kernel', k=10, batch_size=BATCH_SIZE)
neg_lids = get_lids_random_batch(neg_data, LID_GAMMA, lid_type='kernel', k=10, batch_size=BATCH_SIZE)

row_count = x_train.shape[0]
# initialize the weights to 0
lids = np.zeros(row_count)
# insert LIDs at the correct index
lids[pos_indices] = pos_lids
lids[neg_indices] = neg_lids

# get the indices of rows that are not flipped
normal_idx = list(set(range(row_count)) - set(x_indices))

fl_lids = lids[x_indices]
normal_lids = lids[normal_idx]
flipped_labels = y[x_indices]
normal_labels = np.squeeze(y[normal_idx])

pos_indices = np.where(normal_labels == 1)[0]
neg_indices = np.where(normal_labels == 0)[0]

pos_normal_lids = normal_lids[pos_indices]
neg_normal_lids = normal_lids[neg_indices]

pos_fl_indices = np.where(flipped_labels == 1)[0]
neg_fl_indices = np.where(flipped_labels == 0)[0]

pos_fl_lids = fl_lids[pos_fl_indices]
neg_fl_lids = fl_lids[neg_fl_indices]

# placeholders
weights_pos_normal = np.ones((len(pos_normal_lids),))
weights_pos_fl = np.ones((len(pos_fl_lids),))
weights_neg_normal = np.ones((len(neg_normal_lids),))
weights_neg_fl = np.ones((len(neg_fl_lids),))

# If there are labels flipped to positive
if pos_fl_lids.size > 1:
    pos_density_normal = get_kde(pos_normal_lids, type='normal')
    pos_density_fl = get_kde(pos_fl_lids, type='flipped')

    lr_pos_normal, lr_pos_fl = weight_calculation(pos_normal_lids, pos_fl_lids, pos_density_normal,
                                                  pos_density_fl,
                                                  WEIGHT_LB)

    tmp_lid_values = np.concatenate((pos_normal_lids, pos_fl_lids), axis=0)
    # obtain the weights from the fitted function
    weights_pos_normal = lr_pos_normal
    weights_pos_fl = lr_pos_fl

# If there are labels flipped to negative
if neg_fl_lids.size > 1:
    neg_density_normal = get_kde(neg_normal_lids, type='normal')
    neg_density_fl = get_kde(neg_fl_lids, type='flipped')

    lr_neg_normal, lr_neg_fl = weight_calculation(neg_normal_lids, neg_fl_lids, neg_density_normal,
                                                  neg_density_fl,
                                                  WEIGHT_LB)

    tmp_lid_values = np.concatenate((neg_normal_lids, neg_fl_lids), axis=0)
    weights_neg_normal = lr_neg_normal
    weights_neg_fl = lr_neg_fl

weights_fl = np.zeros((len(fl_lids),))
weights_fl[pos_fl_indices] = weights_pos_fl
weights_fl[neg_fl_indices] = weights_neg_fl

weights_normal = np.zeros((len(normal_lids),))
weights_normal[pos_indices] = weights_pos_normal
weights_normal[neg_indices] = weights_neg_normal

weights = np.zeros((row_count,))
weights[x_indices] = weights_fl
weights[normal_idx] = weights_normal

model = svm.SVC(kernel='rbf', gamma=GAMMA, C=C)
model.fit(x_train, y.ravel(), sample_weight=weights)
y_hat = model.predict(xtt)
pois_acc = accuracy_score(ytt, y_hat)
print(pois_acc)
