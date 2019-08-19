import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from numpy import inf
from scipy.spatial.distance import cdist
from scipy.stats import entropy
from sklearn.metrics import accuracy_score, recall_score, f1_score
from sklearn.metrics import precision_score
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.neighbors import KernelDensity
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm

sns.set(font_scale=1.9)
sns.set_style("whitegrid")


def make_meshgrid(x, y, h=.02):
    """Create a mesh of points to plot in

    Parameters
    ----------
    x: data to base x-axis meshgrid on
    y: data to base y-axis meshgrid on
    h: stepsize for meshgrid, optional

    Returns
    -------
    xx, yy : ndarray
    """
    x_min, x_max = x.min(), x.max()
    y_min, y_max = y.min(), y.max()
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    return xx, yy


def plot_contours(ax, clf, xx, yy, **params):
    """Plot the decision boundaries for a classifier.

    Parameters
    ----------
    ax: matplotlib axes object
    clf: a classifier
    xx: meshgrid ndarray
    yy: meshgrid ndarray
    params: dictionary of params to pass to contourf, optional
    """
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    out = ax.contourf(xx, yy, Z, **params)
    return out


def draw_contours(X, y, model, fi, title, file_name):
    pos_indices = np.where(y == 1)[0]
    neg_indices = np.where(y == -1)[0]

    pos_data_x0, pos_data_x1 = X[pos_indices, 0], X[pos_indices, 1]
    neg_data_x0, neg_data_x1 = X[neg_indices, 0], X[neg_indices, 1]

    X0, X1 = X[:, 0], X[:, 1]
    sv_ind = model.support_
    sv_X0, sv_X1 = X[sv_ind, 0], X[sv_ind, 1]

    flipped_X0, flipped_X1 = X[fi, 0], X[fi, 1]

    xx, yy = make_meshgrid(X0, X1, 0.01)
    ax = plt.gca()
    plot_contours(ax, model, xx, yy,
                  cmap=plt.cm.coolwarm, alpha=0.8)
    ax.scatter(pos_data_x0, pos_data_x1, c='red', s=40, edgecolors='k')
    ax.scatter(neg_data_x0, neg_data_x1, c='blue', s=40, edgecolors='k')

    ax.scatter(sv_X0, sv_X1, s=100, facecolors='none', zorder=10, edgecolor='k')
    ax.scatter(flipped_X0, flipped_X1, c='chartreuse', s=40, marker='+')
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    ax.set_xlabel('$x_0$')
    ax.set_ylabel('$x_1$')
    ax.set_xticks(())
    ax.set_yticks(())
    ax.set_title(title)
    plt.show()
    # plt.savefig('images/{}.pdf'.format(file_name), format='pdf', dpi=1000, bbox_inches='tight')
    plt.close()


def classification_perf(y, y_hat, scenario, index, column_names):
    accuracy = accuracy_score(y, y_hat)
    error_rate = 1 - accuracy
    precision = precision_score(y, y_hat, average=None)
    recall = recall_score(y, y_hat, average=None)
    fscore = f1_score(y, y_hat, average=None)

    return pd.DataFrame([[scenario, index, accuracy, error_rate, precision[0], precision[1], np.average(precision),
                          recall[0], recall[1], np.average(recall), fscore[0], fscore[1], np.average(fscore)]],
                        columns=column_names)


# lid of a batch of query points X
# X_opp = data, X_act = batch
def mle_batch(data, batch, k):
    data = np.asarray(data, dtype=np.float32)
    batch = np.asarray(batch, dtype=np.float32)

    k = min(k, len(data) - 1)
    f = lambda v: - k / np.sum(np.log(v / v[-1]))
    a = cdist(batch, data)
    # get the closest k neighbours
    a = np.apply_along_axis(np.sort, axis=1, arr=a)[:, 1:k + 1]
    a = np.apply_along_axis(f, axis=1, arr=a)
    return a


# lid of a batch of query points X
# X_opp = data, X_act = batch
def mle_batch_kernel(data, batch, gamma, k):
    data = np.asarray(data, dtype=np.float32)
    batch = np.asarray(batch, dtype=np.float32)

    k = min(k, len(data) - 1)
    f = lambda v: - k / np.sum(np.log(v / v[-1]))

    K = rbf_kernel(batch, Y=data, gamma=gamma)
    K = np.reciprocal(K) - 1
    # get the closest k neighbours
    a = np.apply_along_axis(np.sort, axis=1, arr=K)[:, 1:k + 1]
    a = np.apply_along_axis(f, axis=1, arr=a)

    # remove inf values
    y = np.isinf(a)
    a[y] = 1000
    return a


def get_lids_random_batch(X, gamma=None, lid_type='normal', k=10, batch_size=100):
    """
    Get the local intrinsic dimensionality of each Xi in X_adv
    estimated by k close neighbours in the random batch it lies in.
    :param model:
    :param X: normal images
    :param k: the number of nearest neighbours for LID estimation
    :param batch_size: default 100
    :return: lids: LID of normal images of shape (num_examples, lid_dim)
    """

    def estimate(i_batch):
        start = i_batch * batch_size
        end = np.minimum(len(X), (i_batch + 1) * batch_size)

        X_act = X[start:end]
        # Maximum likelihood estimation of local intrinsic dimensionality (LID)
        if lid_type == 'normal':
            lid_batch = mle_batch(X_act, X_act, k=k)
        elif lid_type == 'kernel':
            lid_batch = mle_batch_kernel(X_act, X_act, gamma, k=k)
        return lid_batch

    lids = []
    n_batches = int(np.ceil(X.shape[0] / float(batch_size)))
    for i_batch in tqdm(range(n_batches)):
        lid_batch = estimate(i_batch)
        lids.extend(lid_batch)
    lids = np.asarray(lids, dtype=np.float32)
    return lids


def get_cross_lids_random_batch(X, X_opp_class, gamma=None, lid_type='normal', k=10, batch_size=100):
    """
    Get the local intrinsic dimensionality of each Xi in X_adv
    estimated by k close neighbours in the random batch it lies in.
    :param model:
    :param X: normal images
    :param k: the number of nearest neighbours for LID estimation
    :param batch_size: default 100
    :return: lids: LID of normal images of shape (num_examples, lid_dim)
    """

    def estimate(i_batch):
        start = i_batch * batch_size
        end = np.minimum(len(X), (i_batch + 1) * batch_size)
        X_act = X[start:end]

        idx = np.random.randint(X_opp_class.shape[0], size=X_act.shape[0])
        X_opp = X_opp_class[idx, :]
        # Maximum likelihood estimation of local intrinsic dimensionality (LID)
        if lid_type == 'normal':
            lid_batch = mle_batch(X_opp, X_act, k=k)
        elif lid_type == 'kernel':
            lid_batch = mle_batch_kernel(X_opp, X_act, gamma, k=k)
        return lid_batch

    lids = []
    n_batches = int(np.ceil(X.shape[0] / float(batch_size)))
    for i_batch in tqdm(range(n_batches)):
        lid_batch = estimate(i_batch)
        lids.extend(lid_batch)
    lids = np.asarray(lids, dtype=np.float32)
    return lids


def weight_calculation(normal_lids, fl_lids, density_normal, density_fl, WEIGHT_LB):
    # density of normal lids from the normal distribution
    d_normal_normal = density_normal.score_samples(normal_lids.reshape(-1, 1))
    d_normal_normal = np.exp(d_normal_normal)
    # density of normal lids from the fl distribution
    d_normal_fl = density_fl.score_samples(normal_lids.reshape(-1, 1))
    d_normal_fl = np.exp(d_normal_fl)

    # to avoid div by 0
    zero_indices = np.where(d_normal_fl == 0)[0]
    d_normal_fl[zero_indices] = inf
    min_value = np.min(d_normal_fl)
    d_normal_fl[zero_indices] = min_value

    lr_normal = np.divide(d_normal_normal, d_normal_fl)
    # clip any lr above 25
    np.clip(lr_normal, 0, 25, out=lr_normal)
    lr_normal_shape = lr_normal.shape

    # density of fl lids from the normal distribution
    d_fl_normal = density_normal.score_samples(fl_lids.reshape(-1, 1))
    d_fl_normal = np.exp(d_fl_normal)
    # density of fl lids from the fl distribution
    d_fl_fl = density_fl.score_samples(fl_lids.reshape(-1, 1))
    d_fl_fl = np.exp(d_fl_fl)

    # to avoid div by 0
    zero_indices = np.where(d_fl_fl == 0)[0]
    d_fl_fl[zero_indices] = inf
    min_value = np.min(d_fl_fl)
    d_fl_fl[zero_indices] = min_value

    lr_fl = np.divide(d_fl_normal, d_fl_fl)
    np.clip(lr_fl, 0, 25, out=lr_fl)
    lr_fl_shape = lr_fl.shape

    tmp_weights = np.concatenate((lr_normal, lr_fl), axis=0).reshape(-1, 1)

    # scale between 0 and 1
    scaler = MinMaxScaler(feature_range=(WEIGHT_LB, 1))
    scaler.fit(tmp_weights)

    lr_normal = scaler.transform(lr_normal.reshape(-1, 1))
    lr_normal.shape = lr_normal_shape
    lr_fl = scaler.transform(lr_fl.reshape(-1, 1))
    lr_fl.shape = lr_fl_shape

    return lr_normal, lr_fl


def tanh_func(x, a, b):
    return 0.55 + -0.45 * np.tanh(a * x - b)


def get_kde(data, bw=0.2):
    kde = KernelDensity(bandwidth=bw,
                        kernel='gaussian')
    kde.fit(data.reshape(-1, 1))
    return kde


def calculate_kl_divergence(normal_lids, flipped_lids, gamma, flip_rate, dataset, attack, flip_rate_index, label,
                            lid_bins):
    eps = 0.0001
    leg = '-1'
    if label == 'pos':
        leg = '+1'

    hist_normal, bin_edges = np.histogram(normal_lids, bins=lid_bins, density=True)
    hist_fl, be = np.histogram(flipped_lids, bins=bin_edges, density=True)

    # hist_normal, bin_edges = np.histogram(flipped_lids, bins=lid_bins, density=True)
    # hist_fl, be = np.histogram(normal_lids, bins=bin_edges, density=True)
    q_0_indices = np.where(hist_fl == 0)[0]

    for indx in q_0_indices:
        if hist_normal[indx] != 0:
            hist_fl[indx] = eps

    kl = entropy(hist_normal, hist_fl)

    # plotting
    width = 0.7 * (bin_edges[1] - bin_edges[0])
    center = (bin_edges[:-1] + bin_edges[1:]) / 2
    plt.bar(center, hist_normal, align='center', width=width, label=leg, color='#3498db', alpha=0.5, )
    plt.bar(center, hist_fl, align='center', width=width, label='flipped to {}'.format(leg), color='#e74c3c',
            alpha=0.5, )

    ax = plt.gca()
    ax.set_xlim([0, 8])
    ax.set_ylim([0, 0.8])

    plt.grid()
    plt.legend(loc='upper right')
    plt.xlabel('LID')
    plt.ylabel('Probability Density')
    plt.title('PDF of LID values - flip rate {:.2f} | gamma - {:.2f}'.format(flip_rate, gamma))
    plt.savefig(
        'images/kl/{}_kl_{}_{}_{}_{:.2f}.png'.format(label, dataset, attack, flip_rate_index, gamma))
    plt.close()

    print(label)
    print(kl)
    return kl


# Python program to illustrate the intersection
# of two lists using set() method
def intersection(lst1, lst2):
    return list(set(lst1) & set(lst2))
