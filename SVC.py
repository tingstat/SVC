from __future__ import print_function
from scipy.linalg import kron, inv, solveh_banded
from scipy.optimize import minimize
import numpy as np
import scipy as sp
from sklearn.utils.extmath import cartesian
from progress.bar import ShadyBar as basebar
import copy
from scipy.stats import chi2


class MyFancyBar(basebar):
    # suffix = " %(percent)d%%"
    suffix = "%(percent)d%% - Elapsed: %(elapsed_td)s - ETA: %(eta_td)s"


class svc:
    def __init__(self, data, use_banded_rbf):
        self.X = data['X']
        self.Y = data['Y']
        self.Z = data['Z']
        # self.S = data['S']
        self.n = self.X.shape[0]
        self.p = self.X.shape[1]
        self.m = self.Y.shape[1]
        self.N = self.n * self.m
        self.vY = np.reshape(self.Y, self.N)

        self.list_subgram = []
        for i in range(self.p):
            M1 = self.X[:, i].reshape((-1, 1)).dot(self.X[:, i].reshape((1, -1)))
            self.list_subgram.append(kron(M1, data['gram_beta'][i]))
        self.list_subgram.append(kron(data['gram_hz'], data['gram_hs']))
        self.use_banded_rbf = use_banded_rbf
        self.inv_reg_gram = None
        self.vC = None
        self.C = None
        self.seperate_smoothness = None
        self.total_smoothness = None
        self.var_noise = None
        self.sc_tst_stat = None
        self.kappa = None
        self.gamma = None
        self.gcv_score = None
        self.gcv_w_der = dict()
        self.gcv_t_der = dict()
        self.gcv_factor = None

    def copy(self):
        return copy.deepcopy(self)

    @staticmethod
    def band(X, bd_width):
        bd_X = np.zeros((bd_width, X.shape[0]))
        for i in range(bd_width):
            bd_X[i, i:] = np.diag(X, i)
        return bd_X

    def get_c(self):
        return self.vC, self.C

    def estimate(self, grams, diag_grams, components=['beta', 'h'], show=True):
        inv_reg_gram2 = self.inv_reg_gram.dot(self.inv_reg_gram)

        if 'beta' in components:
            beta = np.zeros((self.p, grams[self.p + 1].shape[1]))
            for i in range(self.p):
                beta[i, :] = self.seperate_smoothness[i] * self.X[:, i].dot(self.C).dot(grams[i])

        if 'h' in components:
            h = self.seperate_smoothness[self.p] * grams[self.p].T.dot(self.C).dot(grams[self.p + 1])


        if 'beta' in components and 'h' in components:
            return beta, h
        elif 'beta' in components:
            return beta
        else:
            return h

    def test_helper(self):
        inv_V = self.N * self.total_smoothness * self.inv_reg_gram / self.var_noise
        Imat = np.zeros((len(self.list_subgram) + 1, len(self.list_subgram) + 1))
        for i in range(len(self.list_subgram)):
            Imat[i, i] = 0.5 * sp.trace(inv_V.dot(self.list_subgram[i]).dot(inv_V).dot(self.list_subgram[i]))
            Imat[i, len(self.list_subgram)] = 0.5 * sp.trace(inv_V.dot(self.list_subgram[i]).dot(inv_V))
            Imat[len(self.list_subgram), i] = Imat[i, len(self.list_subgram)]
        for i in range(len(self.list_subgram) - 1):
            for j in range(i + 1, len(self.list_subgram)):
                Imat[i, j] = 0.5 * sp.trace(inv_V.dot(self.list_subgram[i]).dot(inv_V).dot(self.list_subgram[j]))
                Imat[j, i] = Imat[i, j]
        Imat[len(self.list_subgram), len(self.list_subgram)] = 0.5 * sp.trace(inv_V.dot(inv_V))
        return Imat, inv_V

    def test(self, component, smooth_seq, full_tun_param=None, show=True):
        sub_model = self.copy()
        if component < self.p:
            sub_model.X = sp.delete(self.X, [component], axis=1)
            sub_model.p -= 1
            sub_model.list_subgram.pop(component)
            # sub_model.grid_tuning(smooth_seq, sm_param_to_inv_gram, show=show)
            if full_tun_param is None:
                tps = sub_model.tuning(np.ones(len(sub_model.list_subgram)), maxiter=10)
            else:
                tps = np.delete(full_tun_param, component)
                sub_model.solve(tps, 1)
            print(tps)

            Imat, inv_V = sub_model.test_helper()

            I1 = sp.trace(inv_V.dot(self.list_subgram[component]).dot(inv_V).dot(self.list_subgram[component]))
            I2 = sp.zeros(self.p + 1)
            for i in range(0, self.p):
                I2[i] = 0.5 * sp.trace(
                    inv_V.dot(self.list_subgram[component]).dot(inv_V).dot(sub_model.list_subgram[i]))
            I2[self.p] = 0.5 * sp.trace(inv_V.dot(self.list_subgram[component]).dot(inv_V))
            iImat = inv(Imat)
            I = I1 - I2.dot(iImat).dot(I2)
            e = 0.5 * sp.trace(inv_V.dot(self.list_subgram[component]))
            kappa = 0.5 * I / e
            gamma = 2 * e ** 2 / I
            sc_tst_stat = 0.5 * self.vY.T.dot(inv_V).dot(self.list_subgram[component]).dot(inv_V).dot(self.vY)
        else:
            sub_model.list_subgram.pop()
            # sub_model.grid_tuning(smooth_seq, show=show)
            if full_tun_param is None:
                tps = sub_model.tuning(np.ones(len(sub_model.list_subgram)), maxiter=10)
            else:
                tps = np.delete(full_tun_param, component)
                sub_model.solve(tps, 1)
            print(tps)

            # print(sub_model.seperate_smoothness, sub_model.total_smoothness)
            Imat, inv_V = sub_model.test_helper()

            I1 = sp.trace(inv_V.dot(self.list_subgram[-1]).dot(inv_V).dot(self.list_subgram[-1]))
            I2 = sp.zeros(len(sub_model.list_subgram) + 1)
            for i in range(0, len(sub_model.list_subgram)):
                I2[i] = 0.5 * sp.trace(
                    inv_V.dot(self.list_subgram[-1]).dot(inv_V).dot(sub_model.list_subgram[i]))
            I2[len(sub_model.list_subgram)] = 0.5 * sp.trace(inv_V.dot(self.list_subgram[-1]).dot(inv_V))
            iImat = inv(Imat)
            I = I1 - I2.dot(iImat).dot(I2)
            e = 0.5 * sp.trace(inv_V.dot(self.list_subgram[-1]))
            kappa = 0.5 * I / e
            gamma = 2 * e ** 2 / I
            sc_tst_stat = 0.5 * self.vY.T.dot(inv_V).dot(self.list_subgram[-1]).dot(inv_V).dot(self.vY)

        # self.grid_tuning(smooth_seq=smooth_seq, show=show)
        # print(self.total_smoothness, self.seperate_smoothness)
        # Imat, inv_V = self.test_helper()

        pval = 1 - chi2.cdf(sc_tst_stat / kappa * 2, gamma * 2)
        return [sc_tst_stat, kappa, gamma, pval]
        # e = np.zeros(self.p + 1)
        # for i in range(self.p + 1):
        #     e[i] = 0.5 * sp.trace(inv_V.dot(self.list_subgram[i]))

        # self.kappa = 0.5 * diag_inv_Imat / e
        # self.gamma = 2 * e ** 2 / diag_inv_Imat

        # self.sc_tst_stat = np.zeros(self.p + 1)
        # for i in range(self.p + 1):
        #     self.sc_tst_stat[i] = 0.5 * self.vY.T.dot(inv_V).dot(self.list_subgram[i]).dot(inv_V).dot(self.vY)

    def solve(self, sep_smoothness, total_smoothness):
        self.seperate_smoothness = sep_smoothness
        self.total_smoothness = total_smoothness
        gram = np.zeros((self.N, self.N))
        for i in range(len(self.list_subgram)):
            gram += sep_smoothness[i] * self.list_subgram[i]
        reg_gram = gram + self.N * total_smoothness * np.eye(self.N, self.N)
        if self.use_banded_rbf:
            bd_gram = svc.band(reg_gram, int(np.ceil(self.N * 0.8)))
            b = np.zeros((self.N, self.N + 1))
            b[:, 0] = self.vY
            b[:, 1:] = np.eye(self.N, self.N)
            res = solveh_banded(bd_gram, b, lower=True)
            self.inv_reg_gram = res[:, 1:]
            self.vC = res[:, 0]
        else:
            self.inv_reg_gram = inv(reg_gram)
            self.vC = self.inv_reg_gram.dot(self.vY)
        self.C = np.reshape(self.vC, (self.n, self.m), 'C')
        self.var_noise = self.N * total_smoothness * self.vY.T.dot(self.inv_reg_gram).dot(self.inv_reg_gram).dot(
            self.vY) / np.trace(self.inv_reg_gram)
        if self.gcv_factor is not None:
            self.gcv_score = (10 ** self.gcv_factor) * self.var_noise / np.trace(self.inv_reg_gram)
        else:
            self.gcv_score = self.var_noise / np.trace(self.inv_reg_gram)

    def grid_tuning(self, smooth_seq, show=True):
        min_V = np.Inf
        params = cartesian(smooth_seq)
        opt_tp = None
        if show:
            bar = MyFancyBar(message='{:10s}'.format('Tuning'), max=params.shape[0])
        for i in range(params.shape[0]):
            tps = params[i, 0:]
            self.solve(tps[1:], tps[0])
            V = self.var_noise / (self.N * tps[0] * np.trace(self.inv_reg_gram))
            if V < min_V:
                min_V = V
                opt_tp = tps
                # if show:
                #     string = ['{:10.4e}'.format(i) for i in opt_tp]
                #     bar.message = 'Tuning: {}\n'.format(string)
            if show:
                bar.next()
        if show:
            bar.finish()
        self.solve(opt_tp[1:], opt_tp[0])
        return opt_tp

    def gcv_inflation(self, theta0):
        self.solve(theta0, 1)
        self.gcv_factor = 0
        V = self.gcv_score
        while np.log10(V) < 0:
            V *= 10
            self.gcv_factor += 1
        print('GCV inflation factor: {}'.format(self.gcv_factor))

    def gcv(self, eta):
        if self.seperate_smoothness is None:
            # print('gcv1')
            self.solve(np.exp(eta), 1)
        elif list(np.exp(eta)) != list(self.seperate_smoothness):
            # print('gcv2')
            self.solve(np.exp(eta), 1)
        # else:
        #     print('gcv3')
        V = (10 ** self.gcv_factor) * self.var_noise / np.trace(self.inv_reg_gram)
        self.gcv_score = V
        print('GCV = {}'.format(V))
        return V

    def gcv_der(self, eta):
        if self.seperate_smoothness is None:
            # print('der1')
            self.solve(np.exp(eta), 1)
        elif list(np.exp(eta)) != list(self.seperate_smoothness):
            # print('der2')
            self.solve(np.exp(eta), 1)
        # else:
        #     print('der3')
        t = np.trace(self.inv_reg_gram)
        w = self.var_noise * t / self.N
        tuple_eta = tuple(eta)
        if tuple_eta not in self.gcv_w_der:
            self.gcv_w_der[tuple_eta] = np.zeros(len(self.list_subgram))
            self.gcv_t_der[tuple_eta] = np.zeros(len(self.list_subgram))
            for i in range(len(self.gcv_w_der[tuple_eta])):
                self.gcv_w_der[tuple_eta][i] = -2 * np.exp(eta[i]) * self.vY.T.dot(self.inv_reg_gram).dot(
                    self.inv_reg_gram).dot(self.list_subgram[i]).dot(self.inv_reg_gram).dot(self.vY)
                self.gcv_t_der[tuple_eta][i] = -np.exp(eta[i]) * np.trace(
                    self.inv_reg_gram.dot(self.list_subgram[i]).dot(self.inv_reg_gram))
        V_der = (10 ** self.gcv_factor) * self.N * (
            self.gcv_w_der[tuple_eta] / t ** 2 - 2 * w * self.gcv_t_der[tuple_eta] / t ** 3)
        print('    Jacobian = ', V_der)
        return V_der

    def gcv_hess(self, eta):
        if self.seperate_smoothness is None:
            self.solve(np.exp(eta), 1)
        elif list(np.exp(eta)) != list(self.seperate_smoothness):
            self.solve(np.exp(eta), 1)
        D = self.inv_reg_gram
        t = np.trace(self.inv_reg_gram)
        w = self.var_noise * t / self.N
        yd = self.vC
        tuple_eta = tuple(eta)
        if tuple_eta not in self.gcv_w_der:
            self.gcv_w_der[tuple_eta] = np.zeros(len(self.list_subgram))
            self.gcv_t_der[tuple_eta] = np.zeros(len(self.list_subgram))
            for i in range(len(self.gcv_w_der[tuple_eta])):
                self.gcv_w_der[tuple_eta][i] = -2 * np.exp(eta[i]) * yd.T.dot(D).dot(self.list_subgram[i]).dot(yd)
                self.gcv_t_der[tuple_eta][i] = -np.exp(eta[i]) * np.trace(D.dot(self.list_subgram[i]).dot(D))
        w_der = self.gcv_w_der[tuple_eta]
        t_der = self.gcv_t_der[tuple_eta]
        H = np.zeros((len(self.list_subgram), len(self.list_subgram)))
        w_hess = np.zeros_like(H)
        t_hess = np.zeros_like(H)
        for i in range(len(self.list_subgram)):
            tmp = 2 * D.dot(self.list_subgram[i]).dot(D).dot(self.list_subgram[i])
            w_hess[i, i] = 2 * np.exp(eta[i] ** 2) * yd.T.dot(
                tmp + self.list_subgram[i].dot(D).dot(D).dot(self.list_subgram[i])).dot(yd) + w_der[i]
            t_hess[i, i] = 2 * np.trace(D.dot(self.list_subgram[i]).dot(D).dot(self.list_subgram[i]).dot(D)) + t_der[i]
            H[i, i] = (10 ** self.gcv_factor) * self.N * (
                w_hess[i, i] / t ** 2 - 4 * w_der[i] * t_der[i] / t ** 3 - 2 * w * t_hess[i, i] / t ** 3 + 6 * w *
                t_der[i] ** 2 / t ** 4)
            for j in range(i + 1, len(self.list_subgram)):
                w_hess[i, j] = 2 * np.exp(eta[i]) * np.exp(eta[j]) * yd.T.dot(
                    D.dot(self.list_subgram[i]).dot(D).dot(self.list_subgram[j]) + self.list_subgram[i].dot(D).dot(
                        D).dot(self.list_subgram[j]) + self.list_subgram[i].dot(D).dot(self.list_subgram[j]).dot(
                        D)).dot(yd)
                w_hess[j, i] = w_hess[i, j]
                t_hess[i, i] = 2 * np.trace(D.dot(self.list_subgram[i]).dot(D).dot(self.list_subgram[j]).dot(D))
                t_hess[j, i] = t_hess[i, j]
                H[i, j] = (10 ** self.gcv_factor) * self.N * (
                    w_hess[i, j] / t ** 2 - 2 * w_der[i] * t_der[j] / t ** 3 -
                    2 * w_der[j] * t_der[i] / t ** 3 - 2 * w * t_hess[i, j] / t ** 3 +
                    6 * w * t_der[i] * t_der[j] / t ** 4)
                H[j, i] = H[i, j]
        print('GCV Hessian:')
        print(H)
        return H

    def tuning(self, theta0, method='Newton-CG', maxiter=None):
        self.gcv_inflation(theta0)
        if method == 'BFGS':
            res = minimize(self.gcv, np.log(theta0), method='BFGS', jac=self.gcv_der,
                           options={'disp': True, 'maxiter': maxiter, 'gtol': 1e-05})
        elif method == 'Newton-CG':
            res = minimize(self.gcv, np.log(theta0), method='Newton-CG', jac=self.gcv_der, hess=self.gcv_hess,
                           options={'disp': True, 'maxiter': maxiter})
        self.solve(np.exp(res.x), 1)
        return np.exp(res.x)
