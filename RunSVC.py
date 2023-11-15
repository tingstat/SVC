# from __future__ import print_function
import scipy as sp
import numpy as np
from sklearn.utils.extmath import cartesian
from os.path import join, exists
from os import makedirs
from sklearn.metrics.pairwise import rbf_kernel, polynomial_kernel
from SVC import svc, MyFancyBar
import sys
import argparse
from numba import jit
from scipy.stats import chi2


def sim1d_beta(S):
    p = 2
    m = S.size

    Beta = np.zeros((p, m))
    # gs = rbf_kernel(S, gamma=1 / 1)
    # Beta[0, :] = np.ones((1, m)).dot(gs) / m
    # gs = rbf_kernel(S, gamma=1 / 3)
    # Beta[1, :] = np.ones((1, m)).dot(gs) / m
    Beta[0, :] = (10 * S ** 3 - 15 * S ** 2 + 5 * S + np.sin(2 * np.pi * S)).ravel()
    Beta[1, :] = (10 * S ** 6 - 30 * S ** 5 + 25 * S ** 4 - 5 * S ** 2 + 5.0 / 21 + np.sin(6 * np.pi * S)).ravel()
    return Beta


def sim1d_h(Z, S):
    n = Z.shape[0]
    m = S.size
    grids = cartesian([np.arange(n), np.arange(m)])
    Z1 = Z[grids[:, 0], 0]
    Z2 = Z[grids[:, 0], 1]
    S1 = S[grids[:, 1], 0]
    # gz = rbf_kernel(Z, gamma=1 / 0.1)
    # gs = rbf_kernel(S, gamma=1 / 0.03)
    # vH = np.kron(gz, gs).dot(np.ones((n * m, 1))) / 10
    vH = 2 * np.cos(2 * np.pi * (Z1 - Z2)) + S1 * np.sin(2 * np.pi * (Z1 + Z2))
    # vH = (Z1 ** 2 - Z1 + 1.0 / 6.0) * (Z2 ** 2 - Z2 + 1.0 / 6.0) * (S1 - 0.5) + \
    #      10 * (Z2 - 0.5) * (Z1 - 0.5) * np.cos(2 * np.pi * S1)
    # vH = (Z1 ** 2 - Z1 + 1.0 / 6.0) * (Z2 ** 2 - Z2 + 1.0 / 6.0) * (S1 - 0.5) + \
    #      10 * (Z2 - 0.5) * (Z1 - 0.5) * (S1 ** 2 - S1 + 1.0 / 6.0)
    H = np.reshape(vH, (n, m), 'C')
    return H


def gen1d_estimation(set, seed):
    simu_data = {'X': None, 'Y': None, 'Z': None, 'S': None, 'H:': None, 'Beta': None}

    n = set['n']
    p = set['p']
    q = set['q']
    m = set['m']

    # S
    S = np.reshape(np.linspace(0, 1, m), (-1, 1))
    simu_data['S'] = S

    # X
    # mean_x = [0 for _ in range(p - 1)]
    # cov_x = np.eye(p - 1, p - 1)
    # X = np.ones((n, p))
    # np.random.seed(seed=seed)
    # X[:, 1:] = np.random.multivariate_normal(mean_x, cov_x, n)
    mean_x = [0 for _ in range(p)]
    cov_x = np.eye(p, p)
    np.random.seed(seed=seed)
    X = np.random.multivariate_normal(mean_x, cov_x, n)
    simu_data['X'] = X

    # Z
    # mean_z = [0 for _ in range(q)]
    # cov_z = np.eye(q, q) * 0.3
    # np.random.seed(seed=seed)
    # Z = np.random.multivariate_normal(mean_z, cov_z, n)
    Z = np.random.uniform(0, 1, n * q).reshape((n, q))
    simu_data['Z'] = Z

    # H
    simu_data['H'] = sim1d_h(Z, S)

    # Beta
    simu_data['Beta'] = sim1d_beta(S)

    # noise
    mean_noise = [0 for _ in range(m)]
    cov_noise = np.eye(m, m) * set['sigmasq']
    np.random.seed(seed=seed)
    noise = np.random.multivariate_normal(mean_noise, cov_noise, n)

    # Y
    Y = X.dot(simu_data['Beta']) + simu_data['H'] + noise
    simu_data['Y'] = Y

    return simu_data


def gen1d_test_beta(set, seed):
    simu_data = {'X': None, 'Y': None, 'Z': None, 'S': None, 'H:': None, 'Beta': None}

    n = set['n']
    p = set['p']
    q = set['q']
    m = set['m']
    alpha = set['alpha']

    # S
    S = np.reshape(np.linspace(0, 1, m), (-1, 1))
    simu_data['S'] = S

    # X
    mean_x = [0 for _ in range(p)]
    cov_x = np.eye(p, p)
    np.random.seed(seed=seed)
    X = np.random.multivariate_normal(mean_x, cov_x, n)
    simu_data['X'] = X

    # Z
    Z = np.random.uniform(0, 1, n * q).reshape((n, q))
    simu_data['Z'] = Z

    # H
    simu_data['H'] = sim1d_h(Z, S)

    # Beta
    simu_data['Beta'] = sim1d_beta(S)
    simu_data['Beta'][0, :] *= alpha

    # noise
    mean_noise = [0 for _ in range(m)]
    cov_noise = np.eye(m, m) * set['sigmasq']
    np.random.seed(seed=seed)
    noise = np.random.multivariate_normal(mean_noise, cov_noise, n)

    # Y
    Y = X.dot(simu_data['Beta']) + simu_data['H'] + noise
    simu_data['Y'] = Y

    return simu_data


def gen1d_test_h(set, seed):
    simu_data = {'X': None, 'Y': None, 'Z': None, 'S': None, 'H:': None, 'Beta': None}

    n = set['n']
    p = set['p']
    q = set['q']
    m = set['m']
    alpha = set['alpha']

    # S
    S = np.reshape(np.linspace(0, 1, m), (-1, 1))
    simu_data['S'] = S

    # X
    mean_x = [0 for _ in range(p)]
    cov_x = np.eye(p, p)
    np.random.seed(seed=seed)
    X = np.random.multivariate_normal(mean_x, cov_x, n)
    simu_data['X'] = X

    # Z
    Z = np.random.uniform(0, 1, n * q).reshape((n, q))
    simu_data['Z'] = Z

    # H
    simu_data['H'] = sim1d_h(Z, S) * alpha

    # Beta
    simu_data['Beta'] = sim1d_beta(S)

    # noise
    mean_noise = [0 for _ in range(m)]
    cov_noise = np.eye(m, m) * set['sigmasq']
    np.random.seed(seed=seed)
    noise = np.random.multivariate_normal(mean_noise, cov_noise, n)

    # Y
    Y = X.dot(simu_data['Beta']) + simu_data['H'] + noise
    simu_data['Y'] = Y

    return simu_data


def sim1d_test_beta_one_run(work_dir, set, r, show=True):
    # makedirs(join(work_dir, str(r)), exist_ok=True)

    simu_data = gen1d_test_beta(set, seed=r)

    simu_data['gram_beta'] = []
    simu_data['gram_beta'].append(rbf_kernel(simu_data['S'], gamma=1 / set['kernel_param']))  # beta1
    simu_data['gram_beta'].append(rbf_kernel(simu_data['S'], gamma=1 / 0.06))  # beta2
    simu_data['gram_hz'] = rbf_kernel(simu_data['Z'], gamma=1)  # h(z, )
    simu_data['gram_hs'] = rbf_kernel(simu_data['S'], gamma=1 / 0.03)  # h( ,s)

    model = svc(simu_data, use_banded_rbf=False)

    smooth_seqs = [np.logspace(-4, 0, 5),  # total smoothness
                   np.logspace(-3, 2, 6),
                   np.logspace(-3, 2, 6)]

    stat, kappa, gamma, pvalue= model.test(component=0, smooth_seq=smooth_seqs, show=show)
    pvalue = 1 - chi2.cdf(stat / kappa * 2, gamma * 2)
    test_res = np.array([stat, kappa, gamma, pvalue])
    return test_res


def sim1d_test_h_one_run(work_dir, set, r, show=True):
    simu_data = gen1d_test_h(set, seed=r)

    simu_data['gram_beta'] = []
    simu_data['gram_beta'].append(rbf_kernel(simu_data['S'], gamma=1 / 0.03))  # beta1
    simu_data['gram_beta'].append(rbf_kernel(simu_data['S'], gamma=1 / 0.06))  # beta2
    simu_data['gram_hz'] = rbf_kernel(simu_data['Z'], gamma=1 / set['kernel_param_z'])  # h(z, )
    simu_data['gram_hs'] = rbf_kernel(simu_data['S'], gamma=1 / set['kernel_param_s'])  # h( ,s)

    model = svc(simu_data, use_banded_rbf=False)

    smooth_seqs = [np.logspace(-4, 0, 5),  # total smoothness
                   np.logspace(-2, 3, 3),
                   np.logspace(-2, 3, 3)]

    # sm_param_to_inv_gram = dict()
    # for sep_sm1 in np.logspace(-3, 2, 6):
    #     for sep_sm2 in np.logspace(-3, 2, 6):
    #         for tt_sm in np.logspace(-3, 2, 6):
    #             tpl = (sep_sm1, sep_sm2, tt_sm)
    #             print(tpl)
    #             M1 = simu_data['X'][:, 0].reshape((-1, 1)).dot(simu_data['X'][:, 0].reshape((1, -1)))
    #             M2 = simu_data['X'][:, 1].reshape((-1, 1)).dot(simu_data['X'][:, 1].reshape((1, -1)))
    #             sm_param_to_inv_gram[tpl] = sep_sm1 * sp.kron(M1, simu_data['gram_beta'][0]) + \
    #                                         sep_sm2 * sp.kron(M2, simu_data['gram_beta'][1])

    stat, kappa, gamma,pvalue = model.test(component=2, smooth_seq=smooth_seqs, show=show)
    pvalue =  1 - chi2.cdf(stat/ kappa * 2, gamma * 2)
    test_res = np.array([stat, kappa, gamma, pvalue])
    return test_res


@jit
def sim1d_test(work_dir, set, show=True):
    left = 1
    right = set['nb_run'] + 1
    test_res = np.zeros((right - left, 4))
    power = 0
    bar = MyFancyBar(message="alpha={}, kpz={}, kps={}, power={:0.4f}".format(set['alpha'],
                                                                              set['kernel_param_z'],
                                                                              set['kernel_param_s'],
                                                                              power),
                     max=right - left)

    for r in range(left, right):
        bar.next()
        # if (r - left) % 500 == 0:
        #     print("{} / {}: alpha={}, kp={}".format(r - left, right - left, set['alpha'], set['kernel_param']))
        test_res[r - 1, :] = sim1d_test_h_one_run(work_dir, set, r, show)
        if test_res[r - 1, -1] < 0.05:
            power += 1 / (right - left)
            bar.message = "alpha={}, kpz={}, kps={}, power={:0.4f}".format(set['alpha'],
                                                                           set['kernel_param_z'],
                                                                           set['kernel_param_s'],
                                                                           power)
    bar.finish()
    # np.savetxt(join(work_dir,
    #                 'result_n{}_m{}_p{}_q{}_alpha{:0.1f}_sigmasq{:0.1f}_kp{}.txt'.format(set['n'], set['m'], set['p'],
    #                                                                                      set['q'], set['alpha'],
    #                                                                                      set['sigmasq'],
    #                                                                                      set['kernel_param'])),
    #            test_res)
    np.savetxt(join(work_dir,
                    'result_n{}_m{}_p{}_q{}_alpha{:0.1f}_sigmasq{:0.1f}_kpz{}_kps{}.txt'.format(set['n'], set['m'],
                                                                                                set['p'],
                                                                                                set['q'], set['alpha'],
                                                                                                set['sigmasq'],
                                                                                                set['kernel_param_z'],
                                                                                                set['kernel_param_s'])),
               test_res)


def sim1d_estimation_one_run(work_dir, set, r, show=True):
    work_dir = join(work_dir, 'Sim1d_estn_n{}_m{}_p{}_q{}_sigmasq{}'.format(set['n'], set['m'], set['p'], set['q'],
                                                                            set['sigmasq']))
    makedirs(join(work_dir, str(r)), exist_ok=True)

    # True H function (dense)
    nb = 100
    dense_z = np.linspace(0, 1, nb)
    dense_s = np.linspace(0, 1, nb).reshape((-1, 1))

    true_H = sim1d_h(cartesian([dense_z, dense_z]), dense_s)
    true_beta = sim1d_beta(dense_s)
    if r == 1:
        np.savetxt(join(work_dir, 'true_H.txt'), true_H)
        np.savetxt(join(work_dir, 'true_beta.txt'), true_beta)

    simu_data = gen1d_estimation(set, seed=r)

    gammas = [1 / 0.03, 1 / 0.03, 1, 1 / 0.03]
    simu_data['gram_beta'] = [rbf_kernel(simu_data['S'], gamma=gammas[0]),
                              rbf_kernel(simu_data['S'], gamma=gammas[1])]
    simu_data['gram_hz'] = rbf_kernel(simu_data['Z'], gamma=gammas[2])  # h(z, )
    simu_data['gram_hs'] = rbf_kernel(simu_data['S'], gamma=gammas[3])  # h( ,s)

    np.savetxt(join(work_dir, str(r), 'Z.txt'), simu_data['Z'], fmt='%1.8e')
    np.savetxt(join(work_dir, str(r), 'X.txt'), simu_data['X'], fmt='%1.8e')
    np.savetxt(join(work_dir, str(r), 'Y.txt'), simu_data['Y'], fmt='%1.8e')
    np.savetxt(join(work_dir, str(r), 'beta0.txt'), simu_data['Beta'], fmt='%1.8e')
    np.savetxt(join(work_dir, str(r), 'h0.txt'), simu_data['H'], fmt='%1.8e')

    model = svc(simu_data, use_banded_rbf=False)
    sm_seq = np.logspace(-5, 0, 5)
    smooth_seqs = [np.logspace(-9, 0, 5), sm_seq, sm_seq, sm_seq]
    opt_tp = model.tuning(np.ones(model.p + 1))
    print(opt_tp)

    grams = simu_data['gram_beta'].copy()
    grams.append(simu_data['gram_hz'])
    grams.append(simu_data['gram_hs'])
    diag_grams = [1 for _ in range(set['p'] + 1)]
    beta,  h= model.estimate(grams, diag_grams, show=show)


    np.savetxt(join(work_dir, str(r), 'beta.txt'), beta, fmt='%1.8e')

    np.savetxt(join(work_dir, str(r), 'h.txt'), h, fmt='%1.8e')


    grams = [rbf_kernel(simu_data['S'], dense_s, gamma=gammas[0]),
             rbf_kernel(simu_data['S'], dense_s, gamma=gammas[1]),
             rbf_kernel(simu_data['Z'], cartesian([dense_z, dense_z]), gamma=gammas[2]),
             rbf_kernel(simu_data['S'], dense_s, gamma=gammas[3])]

    beta, h = model.estimate(grams, diag_grams, show=show)


    np.savetxt(join(work_dir, str(r), 'beta[ds].txt'), beta, fmt='%1.8e')

    np.savetxt(join(work_dir, str(r), 'h[ds].txt'), h, fmt='%1.8e')



def sim1d_estimation(work_dir, set, show=True):
    # makedirs(work_dir, exist_ok=True)

    for r in range(1, set['nb_run'] + 1):
        print("{} / {}".format(r, set['nb_run']))
        sim1d_estimation_one_run(work_dir, set, r, show)



def main(args):
    parser = argparse.ArgumentParser(description="Run SVC analysis")
    parser.add_argument('--dir', dest="dir", type=str,
                        default='output/simulation/test_beta')
    parser.add_argument('--run', dest="run", type=int, default=2000)
    parser.add_argument('--n', dest="n", type=int, default=50)
    parser.add_argument('--m', dest="m", type=int, default=20)
    parser.add_argument('--sigmasq', dest='sigmasq', type=float, default=0.5)
    parser.add_argument('--R', dest="nb_run", type=int, default=2000)
    parser.add_argument('--sim', dest="sim", type=str, default='test')
    parser.add_argument('--mode', dest="mode", type=str, default='separate')
    parser.add_argument('--alpha', dest='alpha', type=float, default=0.0)
    parser.add_argument('--rbf_bw', dest='rbf_bw', type=float, default=1)
    parser.add_argument('--tract', dest='tract', type=str)
    parser.add_argument('--gene', dest='gene', type=str)

    opts = parser.parse_args(args[1:])

    set = {'n': opts.n, 'm': opts.m, 'p': 2, 'q': 2, 'alpha': opts.alpha,
           'sigmasq': opts.sigmasq, 'kernel_param': opts.rbf_bw,
           'nb_run': opts.nb_run}

    if opts.sim == 'estimation' and opts.mode == 'whole':
        sim1d_estimation(opts.dir, set, True)
    elif opts.sim == 'estimation' and opts.mode == 'separate':
        for r in range(opts.run, opts.run + 1):
            sim1d_estimation_one_run(opts.dir, set, r)
    elif opts.sim == 'test' and opts.mode == 'whole':
        makedirs(opts.dir, exist_ok=True)
        for alpha in [0, 0.1, 0.2, 0.3, 0.4, 0.6, 0.8, 1]:
            for kpz in [0.0002, 0.0005, 0.002, 0.005, 0.02, 0.05, 0.2, 0.5]:
                for kps in [0.1, 1, 10]:
                    set['kernel_param_z'] = kpz
                    set['kernel_param_s'] = kps
                    set['alpha'] = alpha
                    sim1d_test(opts.dir, set, False)
    elif opts.sim == 'test' and opts.mode == 'separate':
        for r in range(opts.run, opts.run + 1):
            sim1d_test_beta_one_run(opts.dir, set, r)
    else:
        raise Exception(
            '--sim or --mode is not correct.\n--sim should be estimation or test\n--mode should be whole, separate, or adni')


if __name__ == '__main__':
    main(sys.argv)
