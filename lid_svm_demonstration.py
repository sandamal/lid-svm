import random

import scipy.io as sio
from scipy import optimize
from sklearn import svm

from utilities import *

seed = 123
random.seed(seed)
np.random.seed(seed)

# Variables
C = 1  # SVM regularization parameter
GAMMA = 0.5  # gamma of the SVM
DATASET = 'toy_2'
ATTACK = 'ALFA'

# find gamma with the highest KL divergence - offline
LID_GAMMA = 0.012
BATCH_SIZE = 50
###########

XMIN = 0
XMAX = 6
YMIN = 0
YMAX = 1.2
WEIGHT_LB = 0.1

# load the data
mat_contents = sio.loadmat('data/toy/{}.mat'.format(DATASET))

# training data
xtr = mat_contents['Xtr']
# training labels
ytr = mat_contents['Ytr']
# testing data
xtt = mat_contents['Xtt']
# testing labels
ytt = mat_contents['Ytt']

X = xtr
y = ytr
row_count = len(X)

n_sv = []

###############################################################
# No attack
###############################################################
model = svm.SVC(kernel='rbf', gamma=GAMMA, C=C)
model.fit(X, y.ravel())
y_hat = model.predict(xtt)

draw_contours(X, ytr, model, [], 'SVM - no attack', 'no_attack')

###############################################################
# Attack
###############################################################

# flipped indices
fi = mat_contents['fi']
# correct matlab 1 indexing
fi = fi - 1
# flipped labels
fl = mat_contents['fl']
y = fl

model = svm.SVC(kernel='rbf', gamma=GAMMA, C=C)
model.fit(X, y.ravel())
y_hat = model.predict(xtt)

draw_contours(X, ytr, model, fi, 'SVM - alfa attack', 'alfa_attack')

###############################################################
# Test defense - LID
###############################################################
pos_density_normal = None
pos_density_fl = None
neg_density_normal = None
neg_density_fl = None

y = fl

pos_indices = np.where(y == 1)[0]
neg_indices = np.where(y == -1)[0]

pos_data = xtr[pos_indices, :]
neg_data = xtr[neg_indices, :]

pos_lids = get_lids_random_batch(pos_data, LID_GAMMA, lid_type='kernel', k=10, batch_size=BATCH_SIZE)
neg_lids = get_lids_random_batch(neg_data, LID_GAMMA, lid_type='kernel', k=10, batch_size=BATCH_SIZE)

# initialize the weights to 0
lids = np.zeros(row_count)
# insert LIDs at the correct index
lids[pos_indices] = pos_lids
lids[neg_indices] = neg_lids

# LIDs w.r.t to the opposite class
pos_lids_opp = get_cross_lids_random_batch(pos_data, neg_data, LID_GAMMA, lid_type='kernel', k=10,
                                           batch_size=BATCH_SIZE)
neg_lids_opp = get_cross_lids_random_batch(neg_data, pos_data, LID_GAMMA, lid_type='kernel', k=10,
                                           batch_size=BATCH_SIZE)
# Cross LID values
pos_cross_lids = np.divide(pos_lids, pos_lids_opp)
neg_cross_lids = np.divide(neg_lids, neg_lids_opp)
#
lids_opp = np.zeros(len(X))
lids_cross = np.zeros(len(X))
lids_opp[pos_indices] = pos_lids_opp
lids_opp[neg_indices] = neg_lids_opp

lids_cross[pos_indices] = pos_cross_lids
lids_cross[neg_indices] = neg_cross_lids

original_lids = lids
lids = lids_cross

fi = fi[:, 0]
# get the indices of rows that are not flipped
normal_idx = list(set(range(row_count)) - set(fi))

fl_lid_values = lids[fi]
normal_lid_values = lids[normal_idx]
flipped_labels = y[fi]
normal_labels = np.squeeze(y[normal_idx])

pos_indices = np.where(normal_labels == 1)[0]
neg_indices = np.where(normal_labels == -1)[0]

pos_normal_lids = normal_lid_values[pos_indices]
neg_normal_lids = normal_lid_values[neg_indices]

pos_fl_indices = np.where(flipped_labels == 1)[0]
neg_fl_indices = np.where(flipped_labels == -1)[0]

pos_fl_lids = fl_lid_values[pos_fl_indices]
neg_fl_lids = fl_lid_values[neg_fl_indices]

# placeholders
weights_pos_normal = np.ones((len(pos_normal_lids),))
weights_pos_fl = np.ones((len(pos_fl_lids),))
weights_neg_normal = np.ones((len(neg_normal_lids),))
weights_neg_fl = np.ones((len(neg_fl_lids),))

# If there are labels flipped to positive
if pos_fl_lids.size > 1:
    pos_density_normal = get_kde(pos_normal_lids, bw=0.2)
    pos_density_fl = get_kde(pos_fl_lids, bw=0.1)

    lr_pos_normal, lr_pos_fl = weight_calculation(pos_normal_lids, pos_fl_lids, pos_density_normal,
                                                  pos_density_fl,
                                                  WEIGHT_LB)

    tmp_lid_values = np.concatenate((pos_normal_lids, pos_fl_lids), axis=0)
    tmp_lr = np.concatenate((lr_pos_normal, lr_pos_fl), axis=0)

    # fit a tanh function
    params, params_covariance = optimize.curve_fit(tanh_func, tmp_lid_values, tmp_lr)

    # obtain the weights from the fitted function
    weights_pos_normal = tanh_func(pos_normal_lids, params[0], params[1])
    weights_pos_fl = tanh_func(pos_fl_lids, params[0], params[1])

if neg_fl_lids.size > 1:
    neg_density_normal = get_kde(neg_normal_lids, bw=0.2)
    neg_density_fl = get_kde(neg_fl_lids, bw=0.1)

    lr_neg_normal, lr_neg_fl = weight_calculation(neg_normal_lids, neg_fl_lids, neg_density_normal,
                                                  neg_density_fl,
                                                  WEIGHT_LB)

    tmp_lid_values = np.concatenate((neg_normal_lids, neg_fl_lids), axis=0)
    tmp_lr = np.concatenate((lr_neg_normal, lr_neg_fl), axis=0)

    params, params_covariance = optimize.curve_fit(tanh_func, tmp_lid_values, tmp_lr)

    weights_neg_normal = tanh_func(neg_normal_lids, params[0], params[1])
    weights_neg_fl = tanh_func(neg_fl_lids, params[0], params[1])

weights_fl = np.zeros((len(fl_lid_values),))
weights_fl[pos_fl_indices] = weights_pos_fl
weights_fl[neg_fl_indices] = weights_neg_fl

weights_normal = np.zeros((len(normal_lid_values),))
weights_normal[pos_indices] = weights_pos_normal
weights_normal[neg_indices] = weights_neg_normal

weights = np.zeros((row_count,))
weights[fi] = weights_fl
weights[normal_idx] = weights_normal

model = svm.SVC(kernel='rbf', gamma=GAMMA, C=C)
model.fit(X, y.ravel(), sample_weight=weights)
y_hat = model.predict(xtt)

draw_contours(X, ytr, model, fi, 'LID-SVM - alfa attack', 'lid_svm')
