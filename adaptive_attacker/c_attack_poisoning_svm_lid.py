"""
.. module:: CAttackPoisoningSVMwithLID
   :synopsis: Poisoning attacks against Support Vector Machine

.. moduleauthor:: Battista Biggio <battista.biggio@unica.it>
.. moduleauthor:: Ambra Demontis <ambra.demontis@unica.it>

"""
import autograd.numpy as np
from autograd import grad as autograd_gradient
from scipy.spatial.distance import cdist
from secml.array import CArray
from sklearn.metrics.pairwise import rbf_kernel

from adaptive_attacker.c_attack_poisoning_lid import CAttackPoisoningLID


class CAttackPoisoningSVMwithLID(CAttackPoisoningLID):
    """Poisoning attacks against Support Vector Machines (SVMs).

    This is an implementation of the attack in https://arxiv.org/pdf/1206.6389:

     - B. Biggio, B. Nelson, and P. Laskov. Poisoning attacks against
       support vector machines. In J. Langford and J. Pineau, editors,
       29th Int'l Conf. on Machine Learning, pages 1807-1814. Omnipress, 2012.

    where the gradient is computed as described in Eq. (10) in
    https://www.usenix.org/conference/usenixsecurity19/presentation/demontis:

     - A. Demontis, M. Melis, M. Pintor, M. Jagielski, B. Biggio, A. Oprea,
       C. Nita-Rotaru, and F. Roli. Why do adversarial attacks transfer?
       Explaining transferability of evasion and poisoning attacks.
       In 28th USENIX Security Symposium. USENIX Association, 2019.

    For more details on poisoning attacks, see also:

     - https://arxiv.org/abs/1804.00308, IEEE Symp. SP 2018
     - https://arxiv.org/abs/1712.03141, Patt. Rec. 2018
     - https://arxiv.org/abs/1708.08689, AISec 2017
     - https://arxiv.org/abs/1804.07933, ICML 2015

    Parameters
    ----------
    classifier : CClassifierSVM
        Target SVM, trained in the dual (i.e., with kernel not set to None).
    training_data : CDataset
        Dataset on which the the classifier has been trained on.
    val : CDataset
        Validation set.
    distance : {'l1' or 'l2'}, optional
        Norm to use for computing the distance of the adversarial example
        from the original sample. Default 'l2'.
    dmax : scalar, optional
        Maximum value of the perturbation. Default 1.
    lb, ub : int or CArray, optional
        Lower/Upper bounds. If int, the same bound will be applied to all
        the features. If CArray, a different bound can be specified for each
        feature. Default `lb = 0`, `ub = 1`.
    y_target : int or None, optional
        If None an error-generic attack will be performed, else a
        error-specific attack to have the samples misclassified as
        belonging to the `y_target` class.
    solver_type : str or None, optional
        Identifier of the solver to be used. Default 'pgd-ls'.
    solver_params : dict or None, optional
        Parameters for the solver. Default None, meaning that default
        parameters will be used.
    init_type : {'random', 'loss_based'}, optional
        Strategy used to chose the initial random samples. Default 'random'.
    random_seed : int or None, optional
        If int, random_state is the seed used by the random number generator.
        If None, no fixed seed will be set.

    """
    __class_type = 'p-svm'

    def __init__(self, classifier,
                 training_data,
                 val,
                 distance='l1',
                 dmax=0,
                 lb=0,
                 ub=1,
                 y_target=None,
                 solver_type='pgd-ls',
                 solver_params=None,
                 init_type='random',
                 random_seed=None,
                 lid_cost_coefficient=50,
                 lid_k=10
                 ):

        CAttackPoisoningLID.__init__(self, classifier=classifier,
                                     training_data=training_data,
                                     val=val,
                                     distance=distance,
                                     dmax=dmax,
                                     lb=lb,
                                     ub=ub,
                                     y_target=y_target,
                                     solver_type=solver_type,
                                     solver_params=solver_params,
                                     init_type=init_type,
                                     random_seed=random_seed,
                                     lid_k=lid_k)

        self.lid_cost_coefficient = lid_cost_coefficient
        self.lid_of_x0 = None
        # check if SVM has been trained in the dual
        if self.classifier.kernel is None:
            raise ValueError(
                "Please retrain the SVM in the dual (kernel != None).")

        # indices of support vectors (at previous iteration)
        # used to check if warm_start can be used in the iterative solver
        self._sv_idx = None

    ###########################################################################
    #                           PRIVATE METHODS
    ###########################################################################

    def _init_solver(self):
        """Overrides _init_solver to additionally reset the SV indices."""
        super(CAttackPoisoningSVMwithLID, self)._init_solver()

        # reset stored indices of SVs
        self._sv_idx = None

    ###########################################################################
    #                  OBJECTIVE FUNCTION & GRAD COMPUTATION
    ###########################################################################

    def fast_lid_cost_calculation(self, xc, k=10):
        lid = self.mle_batch_euclidean(xc, k=k)
        # return lid
        if self.original_lid_values is None:
            if self.lid_of_x0 is None:
                # set the starting LID value
                self.lid_of_x0 = lid
            return self.lid_cost_coefficient * (lid - self.lid_of_x0) ** 2
        else:
            return self.lid_cost_coefficient * (lid - self.original_lid_values[self._idx]) ** 2

    # lid of a batch of query points X
    def mle_batch_euclidean(self, data, k):
        """
        Calculates LID values of data w.r.t batch
        Args:
            data: samples to calculate LIDs of
            batch: samples to calculate LIDs against
            k: the number of nearest neighbors to consider

        Returns: the calculated LID values

        """
        batch = self.training_data_ndarray
        f = lambda v: - k / np.sum(np.log((v / v[-1]) + 1e-9))
        gamma = self.classifier.kernel.gamma
        if gamma is None:
            gamma = 1.0 / self.training_data_ndarray.shape[1]
        K = rbf_kernel(data, Y=batch, gamma=gamma)
        K = np.reciprocal(K)
        # K = cdist(data, batch)
        # get the closest k neighbours
        if self.xc is not None and self.xc.shape[0] == 1:
            # only one attack sample
            sorted_distances = np.sort(K)[0, 1:1 + k]
        else:
            sorted_distances = np.sort(K)[0, 0:k]
        a = np.apply_along_axis(f, axis=0, arr=sorted_distances)
        return np.nan_to_num(a)


    def lid_cost(self, xc, closest_neighbours, k):
        gamma = self.classifier.kernel.gamma
        if gamma is None:
            gamma = 1.0 / self.training_data_ndarray.shape[1]

        r_max = np.sum((closest_neighbours[-1, :] - xc) ** 2, axis=1)
        r_max *= -gamma
        r_max = np.exp(r_max)  # exponentiate r_max
        r_max = np.reciprocal(r_max)

        sum = 0
        for i in range(closest_neighbours.shape[0]):
            r_i = np.sum((closest_neighbours[i, :] - xc) ** 2, axis=1)
            r_i *= -gamma
            r_i = np.exp(r_i)  # exponentiate r_i
            r_i = np.reciprocal(r_i)
            sum += np.log((r_i / r_max) + 1e-9)
        lid = -k / sum
        # r_max = np.sqrt(np.sum((closest_neighbours[-1, :] - xc) ** 2, axis=1))
        # sum = 0
        # for i in range(closest_neighbours.shape[0]):
        #     r_i = np.sqrt(np.sum((closest_neighbours[i, :] - xc) ** 2, axis=1))
        #     sum += np.log((r_i / r_max) + 1e-9)
        # lid = -k / sum
        # return lid
        lid_cost = self.lid_cost_coefficient * (lid - self.original_lid_values[self._idx]) ** 2
        return lid_cost


    def objective_function(self, xc, acc=False):
        """
        Parameters
        ----------
        xc: poisoning point

        Returns
        -------
        f_obj: values of objective function (average hinge loss) at x
        """

        # index of poisoning point within xc.
        # This will be replaced by the input parameter xc
        if self._idx is None:
            idx = 0
        else:
            idx = self._idx

        xc = CArray(xc).atleast_2d()

        n_samples = xc.shape[0]
        if n_samples > 1:
            raise TypeError("xc is not a single sample!")

        self._xc[idx, :] = xc
        clf, tr = self._update_poisoned_clf()

        # targeted attacks
        y_ts = self._y_target if self._y_target is not None else self.val.Y

        y_pred, score = clf.predict(self.val.X, return_decision_function=True)

        # TODO: binary loss check
        if self._attacker_loss.class_type != 'softmax':
            score = CArray(score[:, 1].ravel())

        if acc is True:
            error = CArray(y_ts != y_pred).ravel()  # compute test error
        else:
            error = self._attacker_loss.loss(y_ts, score)
        obj = error.mean()
        lid_cost = self.fast_lid_cost_calculation(xc.tondarray(), self.lid_k)
        obj = obj - lid_cost
        return obj

    def objective_function_gradient(self, xc, normalization=True):
        """
        Compute the loss derivative wrt the attack sample xc

        The derivative is decomposed as:

        dl / x = sum^n_c=1 ( dl / df_c * df_c / x )
        """

        xc = xc.atleast_2d()
        n_samples = xc.shape[0]

        if n_samples > 1:
            raise TypeError("x is not a single sample!")

        # index of poisoning point within xc.
        # This will be replaced by the input parameter xc
        if self._idx is None:
            idx = 0
        else:
            idx = self._idx

        self._xc[idx, :] = xc
        clf, tr = self._update_poisoned_clf()

        y_ts = self._y_target if self._y_target is not None else self.val.Y

        # computing gradient of loss(y, f(x)) w.r.t. f
        _, score = clf.predict(self.val.X, return_decision_function=True)

        grad = CArray.zeros((xc.size,))

        if clf.n_classes <= 2:
            loss_grad = self._attacker_loss.dloss(
                y_ts, CArray(score[:, 1]).ravel())
            grad = self._gradient_fk_xc(
                self._xc[idx, :], self._yc[idx], clf, loss_grad, tr)
        else:
            # compute the gradient as a sum of the gradient for each class
            for c in range(clf.n_classes):
                loss_grad = self._attacker_loss.dloss(y_ts, score, c=c)

                grad += self._gradient_fk_xc(self._xc[idx, :], self._yc[idx],
                                             clf, loss_grad, tr, c)

        # TODO: add lid loss gradient here
        a = cdist(xc.tondarray(), self.training_data_ndarray)
        sort_indices = np.apply_along_axis(np.argsort, axis=1, arr=a)[:, 0:self.lid_k]
        neighbors = self.training_data_ndarray[sort_indices, :].squeeze()
        # Create a function to compute the gradient
        grad_lid_cost = autograd_gradient(self.lid_cost, 0)
        lid_gradient = grad_lid_cost(xc.tondarray(), neighbors, self.lid_k)
        grad = grad - CArray(lid_gradient.reshape(grad.shape))
        if normalization:
            norm = grad.norm()
            return grad / norm if norm > 0 else grad
        else:
            return grad

    def _alpha_c(self, clf):
        """
        Returns alpha value of xc, assuming xc to be appended
        as the last point in tr
        """

        # index of poisoning point within xc.
        # This will be replaced by the input parameter xc
        if self._idx is None:
            idx = 0
        else:
            idx = self._idx

        # index of the current poisoning point in the set self._xc
        # as this set is appended to the training set, idx is shifted
        idx += self.training_data.num_samples

        # k is the index of sv_idx corresponding to the training idx of xc
        k = clf.sv_idx.find(clf.sv_idx == idx)
        if len(k) == 1:  # if not empty
            alpha_c = clf.alpha[k].todense().ravel()
            return alpha_c
        return 0

    def alpha_xc(self, xc):
        """
        Parameters
        ----------
        xc: poisoning point

        Returns
        -------
        f_obj: values of objective function (average hinge loss) at x
        """

        # index of poisoning point within xc.
        # This will be replaced by the input parameter xc
        if self._idx is None:
            idx = 0
        else:
            idx = self._idx

        xc = CArray(xc).atleast_2d()

        n_samples = xc.shape[0]
        if n_samples > 1:
            raise TypeError("xc is not a single sample!")

        self._xc[idx, :] = xc
        self._update_poisoned_clf()

        # PARAMETER CLF UNFILLED
        return self._alpha_c()

    ###########################################################################
    #                            GRAD COMPUTATION
    ###########################################################################

    def _Kd_xc(self, clf, alpha_c, xc, xk):
        """
        Derivative of the kernel w.r.t. a training sample xc

        Parameters
        ----------
        xk : CArray
            features of a validation set
        xc:  CArray
            features of the training point w.r.t. the derivative has to be
            computed
        alpha_c:  integer
            alpha value of the of the training point w.r.t. the derivative has
            to be
            computed
        """
        # handle normalizer, if present
        p = clf.kernel.preprocess
        # xc = xc if p is None else p.forward(xc, caching=False)
        xk = xk if p is None else p.forward(xk, caching=False)

        rv = clf.kernel.rv
        clf.kernel.rv = xk
        dKkc = alpha_c * clf.kernel.gradient(xc)
        clf.kernel.rv = rv
        return dKkc.T  # d * k

    def _gradient_fk_xc(self, xc, yc, clf, loss_grad, tr, k=None):
        """
        Derivative of the classifier's discriminant function f(xk)
        computed on a set of points xk w.r.t. a single poisoning point xc
        """

        xc0 = xc.deepcopy()
        d = xc.size
        grad = CArray.zeros(shape=(d,))  # gradient in input space
        alpha_c = self._alpha_c(clf)

        if abs(alpha_c) == 0:  # < svm.C:  # this include alpha_c == 0
            # self.logger.debug("Warning: xc is not an error vector.")
            return grad

        # take only validation points with non-null loss
        xk = self._val.X[abs(loss_grad) > 0, :].atleast_2d()
        grad_loss_fk = CArray(loss_grad[abs(loss_grad) > 0]).T

        # gt is the gradient in feature space
        # this gradient component is the only one if margin SV set is empty
        # gt is the derivative of the loss computed on a validation
        # set w.r.t. xc
        Kd_xc = self._Kd_xc(clf, alpha_c, xc, xk)
        assert (clf.kernel.rv.shape[0] == clf.alpha.shape[1])

        gt = Kd_xc.dot(grad_loss_fk).ravel()  # gradient of the loss w.r.t. xc

        xs, sv_idx = clf._sv_margin()  # these points are already normalized

        if xs is None:
            self.logger.debug("Warning: xs is empty "
                              "(all points are error vectors).")
            return gt if clf.kernel.preprocess is None else \
                clf.kernel.preprocess.gradient(xc0, w=gt)

        s = xs.shape[0]

        # derivative of the loss computed on a validation set w.r.t. the
        # classifier params
        fd_params = clf.grad_f_params(xk)
        grad_loss_params = fd_params.dot(grad_loss_fk)

        H = clf.hessian_tr_params()
        H += 1e-9 * CArray.eye(s + 1)

        # handle normalizer, if present
        # xc = xc if clf.preprocess is None else clf.kernel.transform(xc)
        G = CArray.zeros(shape=(gt.size, s + 1))
        rv = clf.kernel.rv
        clf.kernel.rv = xs
        G[:, :s] = clf.kernel.gradient(xc).T
        clf.kernel.rv = rv
        G *= alpha_c

        # warm start is disabled if the set of SVs changes!
        # if self._sv_idx is None or self._sv_idx.size != sv_idx.size or \
        #         (self._sv_idx != sv_idx).any():
        #     self._warm_start = None
        # self._sv_idx = sv_idx  # store SV indices for the next iteration
        #
        # # iterative solver
        # v = - self._compute_grad_solve_iterative(
        #     G, H, grad_loss_params, tol=1e-3)

        # solve with standard linear solver
        # v = - self._compute_grad_solve(G, H, grad_loss_params, sym_pos=False)

        # solve using inverse/pseudo-inverse of H
        # v = - self._compute_grad_inv(G, H, grad_loss_params)
        v = self._compute_grad_inv(G, H, grad_loss_params)

        gt += v

        # propagating gradient back to input space
        if clf.kernel.preprocess is not None:
            return clf.kernel.preprocess.gradient(xc0, w=gt)

        return gt
