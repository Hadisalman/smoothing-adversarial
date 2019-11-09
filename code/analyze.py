import math
import os 

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from typing import *
from IPython import embed
sns.set()

class Accuracy(object):
    def at_radii(self, radii: np.ndarray):
        raise NotImplementedError()


class ApproximateAccuracy(Accuracy):
    def __init__(self, data_file_path: str):
        self.data_file_path = data_file_path

    def at_radii(self, radii: np.ndarray) -> np.ndarray:
        df = pd.read_csv(self.data_file_path, delimiter="\t")
        return np.array([self.at_radius(df, radius) for radius in radii])

    def at_radius(self, df: pd.DataFrame, radius: float):
        return (df["correct"] & (df["radius"] >= radius)).mean()

    def get_abstention_rate(self) -> np.ndarray:
        df = pd.read_csv(self.data_file_path, delimiter="\t")
        return 1.*(df["predict"]==-1).sum()/len(df["predict"])*100


class HighProbAccuracy(Accuracy):
    def __init__(self, data_file_path: str, alpha: float, rho: float):
        self.data_file_path = data_file_path
        self.alpha = alpha
        self.rho = rho

    def at_radii(self, radii: np.ndarray) -> np.ndarray:
        df = pd.read_csv(self.data_file_path, delimiter="\t")
        return np.array([self.at_radius(df, radius) for radius in radii])

    def at_radius(self, df: pd.DataFrame, radius: float):
        mean = (df["correct"] & (df["radius"] >= radius)).mean()
        num_examples = len(df)
        return (mean - self.alpha - math.sqrt(self.alpha * (1 - self.alpha) * math.log(1 / self.rho) / num_examples)
                - math.log(1 / self.rho) / (3 * num_examples))


class EmpiricalAccuracy(Accuracy):
    def __init__(self, data_dir: str):
        self.data_dir = data_dir

    def at_radii(self, radii: np.ndarray, attack: str, method: str) -> np.ndarray:
        accuracies = []
        for radius in radii:
            file_path = os.path.join(self.data_dir, '{}_{:.3f}/{}/predictions'.format(attack, radius, method))
            df = pd.read_csv(file_path, delimiter="\t")
            accuracies.append(self.at_radius(df, radius))
        return np.array(accuracies)

    def at_radius(self, df: pd.DataFrame, radius: float):
        return df["correct"].mean()

class Line(object):
    def __init__(self, quantity: Accuracy, legend: str = None, plot_fmt: str = "", scale_x: float = 1, alpha: float = 1):
        self.quantity = quantity
        self.legend = legend
        self.plot_fmt = plot_fmt
        self.scale_x = scale_x
        self.alpha = alpha


def plot_certified_accuracy(outfile: str, title: str, max_radius: float,
                            lines: List[Line], radius_step: float = 0.01, upper_bounds=False) -> None:
    radii = np.arange(0, max_radius + radius_step, radius_step)
    plt.figure()
    for line in lines:
        plt.plot(radii * line.scale_x, line.quantity.at_radii(radii), line.plot_fmt, alpha=line.alpha,)

    if upper_bounds:
        epsilons = [i for i in range(64,680,64)]
        for line in lines:
            accuracies = []
            for eps in epsilons:
                model = line.quantity.data_file_path.split('data/certify/')[1].split('/test')[0]
                data_file_path = 'philly_exp/predict/predict_all_20steps_PGD_DDN/my_models/{}/PGD_{}'.format(model, eps)
                df = pd.read_csv(data_file_path, delimiter="\t")
                accuracies.append(df['correct'].mean())
            plt.plot([0]+[eps/255.0 for eps in epsilons], [line.quantity.at_radii([0])] + accuracies, line.plot_fmt, dashes=[6, 2], alpha=0.2*line.alpha)

    plt.ylim((0, 1))
    plt.xlim((0, max_radius))
    plt.tick_params(labelsize=14)
    plt.xlabel("radius", fontsize=16)
    plt.ylabel("certified accuracy", fontsize=16)
    plt.legend([method.legend for method in lines if method.legend is not None], loc='upper right', fontsize=16)
    plt.tight_layout()
    plt.savefig(outfile + ".pdf")
    plt.title(title, fontsize=20)
    plt.tight_layout()
    plt.savefig(outfile + ".png", dpi=300)
    plt.close()


def plot_certified_accuracy_per_sigma_against_original_one_sample(outfile: str, title: str, max_radius: float,
                            methods: List[Line], methods_original: List[Line], radius_step: float = 0.01, upper_bounds=False) -> None:

    radii = np.arange(0, max_radius + radius_step, radius_step)
    plt.figure()
    color = ['b', 'orange', 'g', 'r']
    for it, line in enumerate(methods):
        plt.plot(radii * line.scale_x, line.quantity.at_radii(radii), color[it], alpha=line.alpha, label='Ours|'+line.legend)

    for it, line in enumerate(methods_original):
        plt.plot(radii * line.scale_x, line.quantity.at_radii(radii), color[it], dashes=[2, 2], alpha=line.alpha, label='Cohen et al.|'+line.legend)

    plt.ylim((0, 1))
    plt.xlim((0, max_radius))
    plt.tick_params(labelsize=14)
    plt.xlabel("$\ell_2$ radius", fontsize=16)
    plt.ylabel("Certified Accuracy", fontsize=16)
    plt.legend(loc='upper right', fontsize=16)
    plt.tight_layout()
    plt.savefig(outfile + ".pdf")
    plt.title(title, fontsize=20)
    plt.tight_layout()
    plt.savefig(outfile + ".png", dpi=300)
    plt.close()


def plot_certified_accuracy_per_sigma_against_original(outfile: str, title: str, max_radius: float,
                            methods: List[Line], methods_original: List[Line], radius_step: float = 0.01, upper_bounds=False) -> None:
    color = ['b', 'orange', 'g', 'r']
    if 'imagenet' in outfile:
        sigmas = [0.25, 0.5, 1.00]
    elif 'cifar' in outfile:
        sigmas = [0.12, 0.25, 0.5, 1.00]
    for it, sigma in enumerate(sigmas):
        methods_eps = [method for method in methods if '{:.2f}'.format(sigma) in method.quantity.data_file_path]
        accuracies_cert_ours, radii = _get_accuracies_at_radii(methods_eps, 0, max_radius, radius_step)
        plt.plot(radii, accuracies_cert_ours.max(0), color[it], label='Ours|$\sigma = {:.2f}$'.format(sigma))

    for it, line in enumerate(methods_original):
        plt.plot(radii * line.scale_x, line.quantity.at_radii(radii), color[it], dashes=[2, 2], alpha=line.alpha, label='Cohen et al.|'+line.legend)

    plt.ylim((0, 1))
    plt.xlim((0, max_radius))
    plt.tick_params(labelsize=14)
    plt.xlabel("$\ell_2$ radius", fontsize=16)
    plt.ylabel("Certified Accuracy", fontsize=16)
    plt.legend(loc='upper right', fontsize=16)
    plt.tight_layout()
    plt.savefig(outfile + ".pdf")
    plt.title(title, fontsize=20)
    plt.tight_layout()
    plt.savefig(outfile + ".png", dpi=300)
    plt.close()


def smallplot_certified_accuracy(outfile: str, title: str, max_radius: float,
                                 methods: List[Line], radius_step: float = 0.01, xticks=0.5) -> None:
    radii = np.arange(0, max_radius + radius_step, radius_step)
    plt.figure()
    for method in methods:
        plt.plot(radii, method.quantity.at_radii(radii), method.plot_fmt)

    plt.ylim((0, 1))
    plt.xlim((0, max_radius))
    plt.xlabel("radius", fontsize=22)
    plt.ylabel("certified accuracy", fontsize=22)
    plt.tick_params(labelsize=20)
    plt.gca().xaxis.set_major_locator(plt.MultipleLocator(xticks))
    plt.legend([method.legend for method in methods], loc='upper right', fontsize=20)
    plt.tight_layout()
    plt.savefig(outfile + ".pdf")
    plt.close()


def plot_certified_accuracy_upper_envelopes(
                                            outfile: str, 
                                            title: str,
                                            max_radius: float,
                                            methods_certified_ours: List[Line] = None, 
                                            methods_certified_cohen: List[Line]= None,
                                            methods_empirical_ours: str = None,
                                            methods_empirical_cohen: str = None,
                                            radius_step: float = 0.01) -> None:
    plt.figure()
    if methods_certified_ours is not None:
        accuracies_cert_ours, radii = _get_accuracies_at_radii(methods_certified_ours, 0, max_radius, radius_step)
        plt.plot(radii, accuracies_cert_ours.max(0), 'b', label='Ours certified')

    if methods_certified_cohen is not None:
        accuracies_cert_cohen, radii = _get_accuracies_at_radii(methods_certified_cohen, 0, max_radius, radius_step)
        plt.plot(radii, accuracies_cert_cohen.max(0), 'r', label='Cohen et al. certified')

    if 'imagenet' in outfile:
        m = 'num_16'
    elif 'cifar' in outfile:
        m = 'num_128'

    empirical_radii = np.arange(0, max_radius + 0.125, 0.125)
    if methods_empirical_ours is not None:
        emp_acc = EmpiricalAccuracy(methods_empirical_ours)
        accuracies_empirical_ours = emp_acc.at_radii(empirical_radii, attack='PGD', method=m)
        accuracies_empirical_ours[0] = accuracies_cert_ours.max(0)[0]
        plt.plot(empirical_radii, accuracies_empirical_ours, 'b', dashes=[6, 2], label='Ours empirical')

    if methods_empirical_cohen is not None:
        emp_acc = EmpiricalAccuracy(methods_empirical_cohen)
        accuracies_empirical_cohen = emp_acc.at_radii(empirical_radii, attack='PGD', method=m)
        accuracies_empirical_cohen[0] = accuracies_cert_cohen.max(0)[0]
        plt.plot(empirical_radii, accuracies_empirical_cohen, 'r',dashes=[6, 2], label='Cohen et. al empirical')


    plt.ylim((0, 1))
    plt.xlim((0, max_radius))
    plt.tick_params(labelsize=14)
    plt.xlabel("$\ell_2$ radius", fontsize=16)
    plt.ylabel("Accuracy", fontsize=16)
    plt.legend(loc='upper right', fontsize=16)
    plt.tight_layout()
    plt.savefig(outfile + ".pdf")
    plt.title(title, fontsize=20)
    plt.tight_layout()
    plt.savefig(outfile + ".png", dpi=300)
    plt.close()


def plot_certified_accuracy_upper_envelopes_all_methods(
                                            outfile: str, 
                                            title: str,
                                            max_radius: float,
                                            methods_certified_ours: List[Line] = None, 
                                            methods_certified_ours_pretrain: List[Line] = None, 
                                            methods_certified_ours_semisuper: List[Line] = None, 
                                            methods_certified_ours_pretrain_semisuper: List[Line] = None, 
                                            methods_certified_cohen: List[Line]= None,
                                            radius_step: float = 0.01) -> None:
    plt.figure()
    if methods_certified_ours is not None:
        accuracies_cert_ours, radii = _get_accuracies_at_radii(methods_certified_ours, 0, max_radius, radius_step)
        plt.plot(radii, accuracies_cert_ours.max(0), 'b', label='Ours ')

    if methods_certified_ours_pretrain is not None:
        accuracies_cert_ours, radii = _get_accuracies_at_radii(methods_certified_ours_pretrain, 0, max_radius, radius_step)
        plt.plot(radii, accuracies_cert_ours.max(0), 'k', label=' + pretraining ')

    if methods_certified_ours_semisuper is not None:
        accuracies_cert_ours, radii = _get_accuracies_at_radii(methods_certified_ours_semisuper, 0, max_radius, radius_step)
        plt.plot(radii, accuracies_cert_ours.max(0), 'g', label=' + semisupervision ')

    if methods_certified_ours_pretrain_semisuper is not None:
        accuracies_cert_ours, radii = _get_accuracies_at_radii(methods_certified_ours_pretrain_semisuper, 0, max_radius, radius_step)
        plt.plot(radii, accuracies_cert_ours.max(0), 'orange', label=' + Both')

    if methods_certified_cohen is not None:
        accuracies_cert_cohen, radii = _get_accuracies_at_radii(methods_certified_cohen, 0, max_radius, radius_step)
        plt.plot(radii, accuracies_cert_cohen.max(0), 'r', label='Cohen et al. ')

    if 'imagenet' in outfile:
        m = 'num_16'
    elif 'cifar' in outfile:
        m = 'num_128'


    plt.ylim((0, 1))
    plt.xlim((0, max_radius))
    plt.tick_params(labelsize=14)
    plt.xlabel("$\ell_2$ radius", fontsize=16)
    plt.ylabel("Certified Accuracy", fontsize=16)
    plt.legend(loc='upper right', fontsize=16)
    plt.tight_layout()
    plt.savefig(outfile + ".pdf")
    plt.title(title, fontsize=20)
    plt.tight_layout()
    plt.savefig(outfile + ".png", dpi=300)
    plt.close()


def plot_certified_accuracy_upper_envelopes_vary_eps(
                                            outfile: str, 
                                            title: str,
                                            max_radius: float,
                                            methods_certified_ours: List[Line] = None, 
                                            methods_certified_cohen: List[Line]= None,
                                            methods_empirical_ours: str = None,
                                            methods_empirical_cohen: str = None,
                                            radius_step: float = 0.01) -> None:
    plt.figure()
    epsilons = [64, 127, 255, 512]
    for eps in epsilons:
        methods_eps = [method for method in methods_certified_ours if '{}'.format(eps) in method.quantity.data_file_path 
                            and 'PGD' in method.quantity.data_file_path]
        accuracies_cert_ours, radii = _get_accuracies_at_radii(methods_eps, 0, max_radius, radius_step)
        plt.plot(radii, accuracies_cert_ours.max(0), label='$\epsilon = {}$'.format(round(eps/256,2)))

    if methods_certified_cohen is not None:
        accuracies_cert_cohen, radii = _get_accuracies_at_radii(methods_certified_cohen, 0, max_radius, radius_step)
        plt.plot(radii, accuracies_cert_cohen.max(0), dashes=[6, 2], label='Cohen et al.')


    plt.ylim((0, 1))
    plt.xlim((0, max_radius))
    plt.tick_params(labelsize=14)
    plt.xlabel("$\ell_2$ radius", fontsize=16)
    plt.ylabel("Certified Accuracy", fontsize=16)
    plt.legend(loc='upper right', fontsize=16)
    plt.tight_layout()
    plt.savefig(outfile + ".pdf")
    plt.title(title, fontsize=20)
    plt.tight_layout()
    plt.savefig(outfile + ".png", dpi=300)
    plt.close()


def plot_certified_accuracy_upper_envelopes_vary_m(
                                            outfile: str, 
                                            title: str,
                                            max_radius: float,
                                            methods_certified_cohen: List[Line], 
                                            methods_certified_N1: List[Line], 
                                            methods_certified_N2: List[Line], 
                                            methods_certified_N4: List[Line],
                                            methods_certified_N8: List[Line],
                                            radius_step: float = 0.01) -> None:

    plt.figure()
    accuracies_cert_ours_cohen, radii = _get_accuracies_at_radii(methods_certified_cohen, 0, max_radius, radius_step)
    plt.plot(radii, accuracies_cert_ours_cohen.max(0), dashes=[6, 2], label='Cohen et al.')

    accuracies_cert_ours_N1, radii = _get_accuracies_at_radii(methods_certified_N1, 0, max_radius, radius_step)
    plt.plot(radii, accuracies_cert_ours_N1.max(0), label='$m_{train} = 1$')

    accuracies_cert_ours_N2, radii = _get_accuracies_at_radii(methods_certified_N2, 0, max_radius, radius_step)
    plt.plot(radii, accuracies_cert_ours_N2.max(0), label='$m_{train} = 2$')

    accuracies_cert_ours_N4, radii = _get_accuracies_at_radii(methods_certified_N4, 0, max_radius, radius_step)
    plt.plot(radii, accuracies_cert_ours_N4.max(0), label='$m_{train} = 4$')

    accuracies_cert_ours_N8, radii = _get_accuracies_at_radii(methods_certified_N8, 0, max_radius, radius_step)
    plt.plot(radii, accuracies_cert_ours_N8.max(0), label='$m_{train} = 8$')


    plt.ylim((0, 1))
    plt.xlim((0, max_radius))
    plt.tick_params(labelsize=14)
    plt.xlabel("$\ell_2$ radius", fontsize=16)
    plt.ylabel("Certified Accuracy", fontsize=16)
    plt.legend(loc='upper right', fontsize=16)
    plt.tight_layout()
    plt.savefig(outfile + ".pdf")
    plt.title(title, fontsize=20)
    plt.tight_layout()
    plt.savefig(outfile + ".png", dpi=300)
    plt.close()


def plot_certified_accuracy_upper_envelopes_base_vs_ours_vs_cohen(
                                            outfile: str, 
                                            title: str,
                                            max_radius: float,
                                            methods_certified_cohen: List[Line], 
                                            methods_certified_ours: List[Line], 
                                            methods_certified_base: List[Line],
                                            methods_certified_base_with_noise: List[Line],
                                            radius_step: float = 0.01) -> None:

    plt.figure()
    accuracies_cert_cohen, radii = _get_accuracies_at_radii(methods_certified_cohen, 0, max_radius, radius_step)
    plt.plot(radii, accuracies_cert_cohen.max(0), 'g', dashes=[6, 2], label='Cohen et al.')

    accuracies_cert_ours, radii = _get_accuracies_at_radii(methods_certified_ours, 0, max_radius, radius_step)
    plt.plot(radii, accuracies_cert_ours.max(0), 'b', label='Ours')

    accuracies_cert_base, radii = _get_accuracies_at_radii(methods_certified_base, 0, max_radius, radius_step)
    plt.plot(radii, accuracies_cert_base.max(0), 'r', dashes=[2, 2], label='Vanilla PGD')

    accuracies_cert_base, radii = _get_accuracies_at_radii(methods_certified_base_with_noise, 0, max_radius, radius_step)
    plt.plot(radii, accuracies_cert_base.max(0), 'k', dashes=[1, 1], label='Vanilla PGD+noise')


    plt.ylim((0, 1))
    plt.xlim((0, max_radius))
    plt.tick_params(labelsize=14)
    plt.xlabel("$\ell_2$ radius", fontsize=16)
    plt.ylabel("Certified Accuracy", fontsize=16)
    plt.legend(loc='upper right', fontsize=16)
    plt.tight_layout()
    plt.savefig(outfile + ".pdf")
    plt.title(title, fontsize=20)
    plt.tight_layout()
    plt.savefig(outfile + ".png", dpi=300)
    plt.close()


def plot_empirical_accuracy_upper_envelopes_vary_num_samples_during_attack(
                                            outfile: str, 
                                            title: str,
                                            max_radius: float,
                                            methods_certified: List[Line]= None,
                                            methods_empirical: str = None,
                                            radius_step: float = 0.01) -> None:
    fig = plt.figure()
    ax = fig.add_subplot(111)
    if methods_certified is not None:
        accuracies_cert, radii = _get_accuracies_at_radii(methods_certified, 0, max_radius, radius_step)
        plt.plot(radii, accuracies_cert.max(0), label='Cohen et al. certified')

    empirical_radii = np.arange(0, max_radius + 0.125, 0.125)
    if methods_empirical is not None:
        emp_acc = EmpiricalAccuracy(methods_empirical)
        accuracies_empirical_cohen = emp_acc.at_radii(empirical_radii, attack='PGD', method='base')
        plt.plot(empirical_radii, accuracies_empirical_cohen,'xk', linestyle='dashed', 
                                        linewidth=1, markersize=3, label='Empirical vanilla PGD ')

        accuracies_empirical_cohen = emp_acc.at_radii(empirical_radii, attack='PGD', method='num_1')
        plt.plot(empirical_radii, accuracies_empirical_cohen, dashes=[4, 2], linewidth=1, label='$Empirical, m_{test} = 1$')

        accuracies_empirical_cohen = emp_acc.at_radii(empirical_radii, attack='PGD', method='num_4')
        plt.plot(empirical_radii, accuracies_empirical_cohen, dashes=[4, 2], linewidth=1, label='$Empirical, m_{test} = 4$')

        accuracies_empirical_cohen = emp_acc.at_radii(empirical_radii, attack='PGD', method='num_8')
        plt.plot(empirical_radii, accuracies_empirical_cohen, dashes=[4, 2], linewidth=1, label='$Empirical, m_{test} = 8$')

        accuracies_empirical_cohen = emp_acc.at_radii(empirical_radii, attack='PGD', method='num_16')
        plt.plot(empirical_radii, accuracies_empirical_cohen, dashes=[4, 2], linewidth=1, label='$Empirical, m_{test} = 16$')

        accuracies_empirical_cohen = emp_acc.at_radii(empirical_radii, attack='PGD', method='num_64')
        plt.plot(empirical_radii, accuracies_empirical_cohen, dashes=[4, 2], linewidth=1, label='$Empirical, m_{test} = 64$')

        accuracies_empirical_cohen = emp_acc.at_radii(empirical_radii, attack='PGD', method='num_128')
        plt.plot(empirical_radii, accuracies_empirical_cohen, dashes=[4, 2], linewidth=1, label='$Empirical, m_{test} = 128$')


    ax.set_ylim((0, 1))
    ax.set_xlim((0, max_radius))
    ax.tick_params(labelsize=12)
    ax.set_xlabel("$\ell_2$ radius", fontsize=12)
    ax.set_ylabel("Accuracy", fontsize=12)
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.70, box.height])
    ax.legend(loc='center right', fontsize=8, bbox_to_anchor=(1.5, 0.5))
    # ax.tight_layout()
    ax.set_aspect(aspect=1)
    plt.savefig(outfile + ".pdf")
    ax.set_title(title, fontsize=20)
    # plt.tight_layout()
    plt.savefig(outfile + ".png", dpi=300)
    plt.close()


def plot_empirical_accuracy_vary_N(
                                            outfile: str, 
                                            title: str,
                                            max_radius: float,
                                            methods_certified: list,
                                            methods_empirical: list,
                                            radius_step: float = 0.125) -> None:

    plt.figure()

    # if methods_certified is not None:
    #     accuracies_cert_cohen, radii = _get_accuracies_at_radii(methods_certified, 0, max_radius, 0.01)
    #     plt.plot(radii, accuracies_cert_cohen.max(0), label='Cohen et al. certified')

    empirical_radii = np.arange(0, max_radius + radius_step, radius_step)
    for method in methods_empirical:
        N = method.split('N')[-1]
        emp_acc = EmpiricalAccuracy(method)
        accuracies_empirical = emp_acc.at_radii(empirical_radii, attack='PGD', method='num_128')
        plt.plot(empirical_radii, accuracies_empirical, dashes=[6, 2], label='n = {}'.format(N))

    plt.ylim((0, 1))
    plt.xlim((0, max_radius))
    plt.tick_params(labelsize=14)
    plt.xlabel("$\ell_2$ radius", fontsize=16)
    plt.ylabel("Accuracy", fontsize=16)
    plt.legend(loc='upper right', fontsize=16)
    plt.tight_layout()
    plt.savefig(outfile + ".pdf")
    plt.title(title, fontsize=20)
    plt.tight_layout()
    plt.savefig(outfile + ".png", dpi=300)
    plt.close()


def plot_empirical_accuracy_nograd_trick(
                                            outfile: str, 
                                            title: str,
                                            max_radius: float,
                                            methods_certified: list,
                                            methods_empirical: str,
                                            methods_empirical_nograd_trick: str,
                                            radius_step: float = 0.125) -> None:

    fig = plt.figure()
    ax = fig.add_subplot(111)

    if methods_certified is not None:
        accuracies_cert_cohen, radii = _get_accuracies_at_radii(methods_certified, 0, max_radius, 0.01)
        plt.plot(radii, accuracies_cert_cohen.max(0), label='Cohen et al. certified')

    empirical_radii = np.arange(0, max_radius + radius_step, radius_step)
    if methods_empirical is not None:
        emp_acc = EmpiricalAccuracy(methods_empirical)
        accuracies_empirical = emp_acc.at_radii(empirical_radii, attack='PGD', method='num_64')
        plt.plot(empirical_radii, accuracies_empirical, dashes=[2, 3], label='Eq.(6), $m_{test} = 128$')
    
    if methods_empirical_nograd_trick is not None:
        emp_acc = EmpiricalAccuracy(methods_empirical_nograd_trick)
        accuracies_empirical = emp_acc.at_radii(empirical_radii, attack='PGD', method='num_4')
        plt.plot(empirical_radii, accuracies_empirical, dashes=[6, 2], label='Eq.(7), $m_{test} = 4$')

        accuracies_empirical = emp_acc.at_radii(empirical_radii, attack='PGD', method='num_8')
        plt.plot(empirical_radii, accuracies_empirical, dashes=[6, 2], label='Eq.(7), $m_{test} = 8$')

        accuracies_empirical = emp_acc.at_radii(empirical_radii, attack='PGD', method='num_16')
        plt.plot(empirical_radii, accuracies_empirical, dashes=[6, 2], label='Eq.(7), $m_{test} = 16$')

        accuracies_empirical = emp_acc.at_radii(empirical_radii, attack='PGD', method='num_64')
        plt.plot(empirical_radii, accuracies_empirical, dashes=[6, 2], label='Eq.(7), $m_{test} = 64$')

        accuracies_empirical = emp_acc.at_radii(empirical_radii, attack='PGD', method='num_128')
        plt.plot(empirical_radii, accuracies_empirical, dashes=[6, 2], label='Eq.(7), $m_{test} = 128$')

        accuracies_empirical = emp_acc.at_radii(empirical_radii, attack='PGD', method='num_256')
        plt.plot(empirical_radii, accuracies_empirical, dashes=[6, 2], label='Eq.(7), $m_{test} = 256$')

        accuracies_empirical = emp_acc.at_radii(empirical_radii, attack='PGD', method='num_512')
        plt.plot(empirical_radii, accuracies_empirical, dashes=[6, 2], label='Eq.(7), $m_{test} = 512$')

    ax.set_ylim((0, 1))
    ax.set_xlim((0, max_radius))
    ax.tick_params(labelsize=12)
    ax.set_xlabel("$\ell_2$ radius", fontsize=12)
    ax.set_ylabel("Accuracy", fontsize=12)
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.70, box.height])
    ax.legend(loc='center right', fontsize=8, bbox_to_anchor=(1.5, 0.5))
    # ax.tight_layout()
    ax.set_aspect(aspect=1)
    plt.savefig(outfile + ".pdf")
    ax.set_title(title, fontsize=20)
    # plt.tight_layout()
    plt.savefig(outfile + ".png", dpi=300)
    plt.close()



def latex_table_certified_accuracy(outfile: str, radius_start: float, radius_stop: float, radius_step: float,
                                   methods: List[Line]):
    accuracies, radii = _get_accuracies_at_radii(methods, radius_start, radius_stop, radius_step)

    f = open(outfile, 'w')

    f.write("$\ell_2$ Radius")
    for radius in radii:
        f.write("& ${:.3}$".format(radius))
    f.write("\\\\\n")

    f.write("\midrule\n")

    for i, method in enumerate(methods):
        f.write(method.legend)
        for j, radius in enumerate(radii):
            if i == accuracies[:, j].argmax():
                txt = r" & \textbf{" + "{:.2f}".format(accuracies[i, j]) + "}"
            else:
                txt = " & {:.2f}".format(accuracies[i, j])
            f.write(txt)
        f.write("\\\\\n")
    f.close()

def latex_table_certified_accuracy_upper_envelope(outfile: str, radius_start: float, radius_stop: float, radius_step: float,
                                   methods: List[Line], clean_accuracy=True):
    accuracies, radii = _get_accuracies_at_radii(methods, radius_start, radius_stop, radius_step)
    clean_accuracies, _ = _get_accuracies_at_radii(methods, 0, 0, 0.25)
    assert clean_accuracies.shape[1] == 1

    f = open(outfile, 'w')

    f.write("$\ell_2$ Radius")
    for radius in radii:
        f.write("& ${:.3}$".format(radius))
    f.write("\\\\\n")

    f.write("\midrule\n")

    for j, radius in enumerate(radii):
        argmaxs = np.argwhere(accuracies[:,j] == accuracies[:, j].max())
        argmaxs = argmaxs.flatten()
        i = argmaxs[clean_accuracies[argmaxs, 0].argmax()]
        # i = i.flatten()[0]
        if clean_accuracy:
            txt = " & $^{("+"{:.2f})".format(clean_accuracies[i, 0]) + "}" + "${:.2f}".format(accuracies[i, j])
        else:
            txt = " & {:.2f}".format(accuracies[i, j])
        f.write(txt)
    f.write("\\\\\n")
    f.close()


def latex_table_abstention_rate(outfile: str, radius_start: float, radius_stop: float, radius_step: float,
                                   methods: List[Line], clean_accuracy=True):

    accuracies, radii = _get_accuracies_at_radii(methods, radius_start, radius_stop, radius_step)
    clean_accuracies, _ = _get_accuracies_at_radii(methods, 0, 0, 0.25)
    assert clean_accuracies.shape[1] == 1

    abstention_rates = 0*accuracies
    for i, method in enumerate(methods):
        abstention_rates[i,:] = method.quantity.get_abstention_rate()

    f = open(outfile, 'w')

    f.write("$\ell_2$ Radius")
    for radius in radii:
        f.write("& ${:.3}$".format(radius))
    f.write("\\\\\n")

    f.write("\midrule\n")

    for j, radius in enumerate(radii):
        argmaxs = np.argwhere(accuracies[:,j] == accuracies[:, j].max())
        argmaxs = argmaxs.flatten()
        i = argmaxs[clean_accuracies[argmaxs, 0].argmax()]
        # i = i.flatten()[0]
        if clean_accuracy:
            txt = " & $^{("+"{:.2f})".format(clean_accuracies[i, 0]) + "}" + "${:.2f}".format(accuracies[i, j])
        else:
            txt = " & {:.2f}".format(accuracies[i, j])
        f.write(txt)

    f.write("\midrule\n")
    for j, radius in enumerate(radii):
        argmaxs = np.argwhere(accuracies[:,j] == accuracies[:, j].max())
        argmaxs = argmaxs.flatten()
        i = argmaxs[clean_accuracies[argmaxs, 0].argmax()]
        # i = i.flatten()[0]
        txt = " & {:.1f}".format(abstention_rates[i, j])
        f.write(txt)

    f.write("\\\\\n")
    f.close()


def markdown_table_certified_accuracy(outfile: str, radius_start: float, radius_stop: float, radius_step: float,
                                      methods: List[Line]):
    accuracies, radii = _get_accuracies_at_radii(methods, radius_start, radius_stop, radius_step)

    f = open(outfile, 'w')
    f.write("|  | ")
    for radius in radii:
        f.write("r = {:.3} |".format(radius))
    f.write("\n")

    f.write("| --- | ")
    for i in range(len(radii)):
        f.write(" --- |")
    f.write("\n")

    for i, method in enumerate(methods):
        f.write("<b> {} </b>| ".format(method.legend))
        for j, radius in enumerate(radii):
            if i == accuracies[:, j].argmax():
                txt = "{:.2f}<b>*</b> |".format(accuracies[i, j])
            else:
                txt = "{:.2f} |".format(accuracies[i, j])
            f.write(txt)
        f.write("\n")
    f.close()


def radii_to_best_models(outfile: str, methods: List[Line], max_radius: float, radius_step: float = 0.01):
    accuracies, radii = _get_accuracies_at_radii(methods, 0, max_radius, radius_step)

    f = open(outfile, 'w')
    f.write('radius\tbest_cert_accuracy\tbest_model\n')
    for j, radius in enumerate(radii):
        i = accuracies[:, j].argmax()
        model = methods[i].quantity.data_file_path.split('certify/')[1].split('/test')[0]
        f.write('{:.3f}\t{}\t{}'.format(radius, accuracies[i, j], model))
        f.write('\n')
    f.close()


def _get_accuracies_at_radii(methods: List[Line], radius_start: float, radius_stop: float, radius_step: float):
    radii = np.arange(radius_start, radius_stop + radius_step, radius_step)
    accuracies = np.zeros((len(methods), len(radii)))
    for i, method in enumerate(methods):
        accuracies[i, :] = method.quantity.at_radii(radii)
    return accuracies, radii


def print_certified_accuracy(radii: float, lines: List[Line]) -> None:
    for line in lines:
        print(line.quantity.data_file_path)
        print()
        accuracies = line.quantity.at_radii(radii)
        for radius, accuracy in zip(radii, accuracies):
            print("certified accuracy at radius {} is {}%\n".format(radius, accuracy))

#####################################################################################################
## PGD and DDN no noise no noise during attack, just noise data augmentation

PGD_with_noise_during_attack = [
Line(ApproximateAccuracy("data/certify/cifar10/PGD_2steps_baseAttackExp/with_noise/eps_64/cifar10/resnet110/noise_0.12/test/sigma_0.12"), "$\sigma = 0.12$ | $\epsilon = 0.25$"),
Line(ApproximateAccuracy("data/certify/cifar10/PGD_2steps_baseAttackExp/with_noise/eps_127/cifar10/resnet110/noise_0.12/test/sigma_0.12"), "$\sigma = 0.12$ | $\epsilon = 0.50$"),
Line(ApproximateAccuracy("data/certify/cifar10/PGD_2steps_baseAttackExp/with_noise/eps_255/cifar10/resnet110/noise_0.12/test/sigma_0.12"), "$\sigma = 0.12$ | $\epsilon = 1.00$"),
Line(ApproximateAccuracy("data/certify/cifar10/PGD_2steps_baseAttackExp/with_noise/eps_512/cifar10/resnet110/noise_0.12/test/sigma_0.12"), "$\sigma = 0.12$ | $\epsilon = 2.00$"),

Line(ApproximateAccuracy("data/certify/cifar10/PGD_2steps_baseAttackExp/with_noise/eps_64/cifar10/resnet110/noise_0.25/test/sigma_0.25"), "$\sigma = 0.25$ | $\epsilon = 0.25$"),
Line(ApproximateAccuracy("data/certify/cifar10/PGD_2steps_baseAttackExp/with_noise/eps_127/cifar10/resnet110/noise_0.25/test/sigma_0.25"), "$\sigma = 0.25$ | $\epsilon = 0.50$"),
Line(ApproximateAccuracy("data/certify/cifar10/PGD_2steps_baseAttackExp/with_noise/eps_255/cifar10/resnet110/noise_0.25/test/sigma_0.25"), "$\sigma = 0.25$ | $\epsilon = 1.00$"),
Line(ApproximateAccuracy("data/certify/cifar10/PGD_2steps_baseAttackExp/with_noise/eps_512/cifar10/resnet110/noise_0.25/test/sigma_0.25"), "$\sigma = 0.25$ | $\epsilon = 2.00$"),

Line(ApproximateAccuracy("data/certify/cifar10/PGD_2steps_baseAttackExp/with_noise/eps_64/cifar10/resnet110/noise_0.50/test/sigma_0.50"), "$\sigma = 0.50$ | $\epsilon = 0.25$"),
Line(ApproximateAccuracy("data/certify/cifar10/PGD_2steps_baseAttackExp/with_noise/eps_127/cifar10/resnet110/noise_0.50/test/sigma_0.50"), "$\sigma = 0.50$ | $\epsilon = 0.50$"),
Line(ApproximateAccuracy("data/certify/cifar10/PGD_2steps_baseAttackExp/with_noise/eps_255/cifar10/resnet110/noise_0.50/test/sigma_0.50"), "$\sigma = 0.50$ | $\epsilon = 1.00$"),
Line(ApproximateAccuracy("data/certify/cifar10/PGD_2steps_baseAttackExp/with_noise/eps_512/cifar10/resnet110/noise_0.50/test/sigma_0.50"), "$\sigma = 0.50$ | $\epsilon = 2.00$"),

Line(ApproximateAccuracy("data/certify/cifar10/PGD_2steps_baseAttackExp/with_noise/eps_64/cifar10/resnet110/noise_1.00/test/sigma_1.00"), "$\sigma = 1.00$ | $\epsilon = 0.25$"),
Line(ApproximateAccuracy("data/certify/cifar10/PGD_2steps_baseAttackExp/with_noise/eps_127/cifar10/resnet110/noise_1.00/test/sigma_1.00"), "$\sigma = 1.00$ | $\epsilon = 0.50$"),
Line(ApproximateAccuracy("data/certify/cifar10/PGD_2steps_baseAttackExp/with_noise/eps_255/cifar10/resnet110/noise_1.00/test/sigma_1.00"), "$\sigma = 1.00$ | $\epsilon = 1.00$"),
Line(ApproximateAccuracy("data/certify/cifar10/PGD_2steps_baseAttackExp/with_noise/eps_512/cifar10/resnet110/noise_1.00/test/sigma_1.00"), "$\sigma = 1.00$ | $\epsilon = 2.00$"),
]

PGD_no_noise_during_attack = [
Line(ApproximateAccuracy("data/certify/cifar10/PGD_2steps_baseAttackExp/without_noise/eps_64/cifar10/resnet110/noise_0.12/test/sigma_0.12"), "$\sigma = 0.12$ | $\epsilon = 0.25$"),
Line(ApproximateAccuracy("data/certify/cifar10/PGD_2steps_baseAttackExp/without_noise/eps_127/cifar10/resnet110/noise_0.12/test/sigma_0.12"), "$\sigma = 0.12$ | $\epsilon = 0.50$"),
Line(ApproximateAccuracy("data/certify/cifar10/PGD_2steps_baseAttackExp/without_noise/eps_255/cifar10/resnet110/noise_0.12/test/sigma_0.12"), "$\sigma = 0.12$ | $\epsilon = 1.00$"),
Line(ApproximateAccuracy("data/certify/cifar10/PGD_2steps_baseAttackExp/without_noise/eps_512/cifar10/resnet110/noise_0.12/test/sigma_0.12"), "$\sigma = 0.12$ | $\epsilon = 2.00$"),

Line(ApproximateAccuracy("data/certify/cifar10/PGD_2steps_baseAttackExp/without_noise/eps_64/cifar10/resnet110/noise_0.25/test/sigma_0.25"), "$\sigma = 0.25$ | $\epsilon = 0.25$"),
Line(ApproximateAccuracy("data/certify/cifar10/PGD_2steps_baseAttackExp/without_noise/eps_127/cifar10/resnet110/noise_0.25/test/sigma_0.25"), "$\sigma = 0.25$ | $\epsilon = 0.50$"),
Line(ApproximateAccuracy("data/certify/cifar10/PGD_2steps_baseAttackExp/without_noise/eps_255/cifar10/resnet110/noise_0.25/test/sigma_0.25"), "$\sigma = 0.25$ | $\epsilon = 1.00$"),
Line(ApproximateAccuracy("data/certify/cifar10/PGD_2steps_baseAttackExp/without_noise/eps_512/cifar10/resnet110/noise_0.25/test/sigma_0.25"), "$\sigma = 0.25$ | $\epsilon = 2.00$"),

Line(ApproximateAccuracy("data/certify/cifar10/PGD_2steps_baseAttackExp/without_noise/eps_64/cifar10/resnet110/noise_0.50/test/sigma_0.50"), "$\sigma = 0.50$ | $\epsilon = 0.25$"),
Line(ApproximateAccuracy("data/certify/cifar10/PGD_2steps_baseAttackExp/without_noise/eps_127/cifar10/resnet110/noise_0.50/test/sigma_0.50"), "$\sigma = 0.50$ | $\epsilon = 0.50$"),
Line(ApproximateAccuracy("data/certify/cifar10/PGD_2steps_baseAttackExp/without_noise/eps_255/cifar10/resnet110/noise_0.50/test/sigma_0.50"), "$\sigma = 0.50$ | $\epsilon = 1.00$"),
Line(ApproximateAccuracy("data/certify/cifar10/PGD_2steps_baseAttackExp/without_noise/eps_512/cifar10/resnet110/noise_0.50/test/sigma_0.50"), "$\sigma = 0.50$ | $\epsilon = 2.00$"),

Line(ApproximateAccuracy("data/certify/cifar10/PGD_2steps_baseAttackExp/without_noise/eps_64/cifar10/resnet110/noise_1.00/test/sigma_1.00"), "$\sigma = 1.00$ | $\epsilon = 0.25$"),
Line(ApproximateAccuracy("data/certify/cifar10/PGD_2steps_baseAttackExp/without_noise/eps_127/cifar10/resnet110/noise_1.00/test/sigma_1.00"), "$\sigma = 1.00$ | $\epsilon = 0.50$"),
Line(ApproximateAccuracy("data/certify/cifar10/PGD_2steps_baseAttackExp/without_noise/eps_255/cifar10/resnet110/noise_1.00/test/sigma_1.00"), "$\sigma = 1.00$ | $\epsilon = 1.00$"),
Line(ApproximateAccuracy("data/certify/cifar10/PGD_2steps_baseAttackExp/without_noise/eps_512/cifar10/resnet110/noise_1.00/test/sigma_1.00"), "$\sigma = 1.00$ | $\epsilon = 2.00$"),
]


PGD_no_noise_during_attack_and_training = [
Line(ApproximateAccuracy("data/certify/cifar10/PGD_2steps_baseAttackExp/without_noise/eps_64/cifar10/resnet110/noise_0.00/test/sigma_0.12"), "$\sigma = 0.12$ | $\epsilon = 0.25$"),
Line(ApproximateAccuracy("data/certify/cifar10/PGD_2steps_baseAttackExp/without_noise/eps_127/cifar10/resnet110/noise_0.00/test/sigma_0.12"), "$\sigma = 0.12$ | $\epsilon = 0.50$"),
Line(ApproximateAccuracy("data/certify/cifar10/PGD_2steps_baseAttackExp/without_noise/eps_255/cifar10/resnet110/noise_0.00/test/sigma_0.12"), "$\sigma = 0.12$ | $\epsilon = 1.00$"),
Line(ApproximateAccuracy("data/certify/cifar10/PGD_2steps_baseAttackExp/without_noise/eps_512/cifar10/resnet110/noise_0.00/test/sigma_0.12"), "$\sigma = 0.12$ | $\epsilon = 2.00$"),

Line(ApproximateAccuracy("data/certify/cifar10/PGD_2steps_baseAttackExp/without_noise/eps_64/cifar10/resnet110/noise_0.00/test/sigma_0.25"), "$\sigma = 0.25$ | $\epsilon = 0.25$"),
Line(ApproximateAccuracy("data/certify/cifar10/PGD_2steps_baseAttackExp/without_noise/eps_127/cifar10/resnet110/noise_0.00/test/sigma_0.25"), "$\sigma = 0.25$ | $\epsilon = 0.50$"),
Line(ApproximateAccuracy("data/certify/cifar10/PGD_2steps_baseAttackExp/without_noise/eps_255/cifar10/resnet110/noise_0.00/test/sigma_0.25"), "$\sigma = 0.25$ | $\epsilon = 1.00$"),
Line(ApproximateAccuracy("data/certify/cifar10/PGD_2steps_baseAttackExp/without_noise/eps_512/cifar10/resnet110/noise_0.00/test/sigma_0.25"), "$\sigma = 0.25$ | $\epsilon = 2.00$"),

Line(ApproximateAccuracy("data/certify/cifar10/PGD_2steps_baseAttackExp/without_noise/eps_64/cifar10/resnet110/noise_0.00/test/sigma_0.50"), "$\sigma = 0.50$ | $\epsilon = 0.25$"),
Line(ApproximateAccuracy("data/certify/cifar10/PGD_2steps_baseAttackExp/without_noise/eps_127/cifar10/resnet110/noise_0.00/test/sigma_0.50"), "$\sigma = 0.50$ | $\epsilon = 0.50$"),
Line(ApproximateAccuracy("data/certify/cifar10/PGD_2steps_baseAttackExp/without_noise/eps_255/cifar10/resnet110/noise_0.00/test/sigma_0.50"), "$\sigma = 0.50$ | $\epsilon = 1.00$"),
Line(ApproximateAccuracy("data/certify/cifar10/PGD_2steps_baseAttackExp/without_noise/eps_512/cifar10/resnet110/noise_0.00/test/sigma_0.50"), "$\sigma = 0.50$ | $\epsilon = 2.00$"),

Line(ApproximateAccuracy("data/certify/cifar10/PGD_2steps_baseAttackExp/without_noise/eps_64/cifar10/resnet110/noise_0.00/test/sigma_1.00"), "$\sigma = 1.00$ | $\epsilon = 0.25$"),
Line(ApproximateAccuracy("data/certify/cifar10/PGD_2steps_baseAttackExp/without_noise/eps_127/cifar10/resnet110/noise_0.00/test/sigma_1.00"), "$\sigma = 1.00$ | $\epsilon = 0.50$"),
Line(ApproximateAccuracy("data/certify/cifar10/PGD_2steps_baseAttackExp/without_noise/eps_255/cifar10/resnet110/noise_0.00/test/sigma_1.00"), "$\sigma = 1.00$ | $\epsilon = 1.00$"),
Line(ApproximateAccuracy("data/certify/cifar10/PGD_2steps_baseAttackExp/without_noise/eps_512/cifar10/resnet110/noise_0.00/test/sigma_1.00"), "$\sigma = 1.00$ | $\epsilon = 2.00$"),
]

#####################################################################################################
## DDN/PGD varying number of attack steps experiments


DDN_2steps = [
Line(ApproximateAccuracy("data/certify/cifar10/DDN_2steps/eps_64/cifar10/resnet110/noise_0.12/test/sigma_0.12"), "$\sigma = 0.12$ | $\epsilon = 0.25$"),
Line(ApproximateAccuracy("data/certify/cifar10/DDN_2steps/eps_127/cifar10/resnet110/noise_0.12/test/sigma_0.12"), "$\sigma = 0.12$ | $\epsilon = 0.50$"),
Line(ApproximateAccuracy("data/certify/cifar10/DDN_2steps/eps_255/cifar10/resnet110/noise_0.12/test/sigma_0.12"), "$\sigma = 0.12$ | $\epsilon = 1.00$"),
Line(ApproximateAccuracy("data/certify/cifar10/DDN_2steps/eps_512/cifar10/resnet110/noise_0.12/test/sigma_0.12"), "$\sigma = 0.12$ | $\epsilon = 2.00$"),            

Line(ApproximateAccuracy("data/certify/cifar10/DDN_2steps/eps_64/cifar10/resnet110/noise_0.25/test/sigma_0.25"), "$\sigma = 0.25$ | $\epsilon = 0.25$"),
Line(ApproximateAccuracy("data/certify/cifar10/DDN_2steps/eps_127/cifar10/resnet110/noise_0.25/test/sigma_0.25"), "$\sigma = 0.25$ | $\epsilon = 0.50$"),
Line(ApproximateAccuracy("data/certify/cifar10/DDN_2steps/eps_255/cifar10/resnet110/noise_0.25/test/sigma_0.25"), "$\sigma = 0.25$ | $\epsilon = 1.00$"),
Line(ApproximateAccuracy("data/certify/cifar10/DDN_2steps/eps_512/cifar10/resnet110/noise_0.25/test/sigma_0.25"), "$\sigma = 0.25$ | $\epsilon = 2.00$"),

Line(ApproximateAccuracy("data/certify/cifar10/DDN_2steps/eps_64/cifar10/resnet110/noise_0.50/test/sigma_0.50"), "$\sigma = 0.50$ | $\epsilon = 0.25$"),
Line(ApproximateAccuracy("data/certify/cifar10/DDN_2steps/eps_127/cifar10/resnet110/noise_0.50/test/sigma_0.50"), "$\sigma = 0.50$ | $\epsilon = 0.50$"),
Line(ApproximateAccuracy("data/certify/cifar10/DDN_2steps/eps_255/cifar10/resnet110/noise_0.50/test/sigma_0.50"), "$\sigma = 0.50$ | $\epsilon = 1.00$"),
Line(ApproximateAccuracy("data/certify/cifar10/DDN_2steps/eps_512/cifar10/resnet110/noise_0.50/test/sigma_0.50"), "$\sigma = 0.50$ | $\epsilon = 2.00$"),

Line(ApproximateAccuracy("data/certify/cifar10/DDN_2steps/eps_64/cifar10/resnet110/noise_1.00/test/sigma_1.00"), "$\sigma = 1.00$ | $\epsilon = 0.25$"),            
Line(ApproximateAccuracy("data/certify/cifar10/DDN_2steps/eps_127/cifar10/resnet110/noise_1.00/test/sigma_1.00"), "$\sigma = 1.00$ | $\epsilon = 0.50$"),
Line(ApproximateAccuracy("data/certify/cifar10/DDN_2steps/eps_255/cifar10/resnet110/noise_1.00/test/sigma_1.00"), "$\sigma = 1.00$ | $\epsilon = 1.00$"),
Line(ApproximateAccuracy("data/certify/cifar10/DDN_2steps/eps_512/cifar10/resnet110/noise_1.00/test/sigma_1.00"), "$\sigma = 1.00$ | $\epsilon = 2.00$"),
]

PGD_2steps = [
Line(ApproximateAccuracy("data/certify/cifar10/PGD_2steps/eps_64/cifar10/resnet110/noise_0.12/test/sigma_0.12"), "$\sigma = 0.12$ | $\epsilon = 0.25$"),
Line(ApproximateAccuracy("data/certify/cifar10/PGD_2steps/eps_127/cifar10/resnet110/noise_0.12/test/sigma_0.12"), "$\sigma = 0.12$ | $\epsilon = 0.50$"),
Line(ApproximateAccuracy("data/certify/cifar10/PGD_2steps/eps_255/cifar10/resnet110/noise_0.12/test/sigma_0.12"), "$\sigma = 0.12$ | $\epsilon = 1.00$"),
Line(ApproximateAccuracy("data/certify/cifar10/PGD_2steps/eps_512/cifar10/resnet110/noise_0.12/test/sigma_0.12"), "$\sigma = 0.12$ | $\epsilon = 2.00$"),            

Line(ApproximateAccuracy("data/certify/cifar10/PGD_2steps/eps_64/cifar10/resnet110/noise_0.25/test/sigma_0.25"), "$\sigma = 0.25$ | $\epsilon = 0.25$"),
Line(ApproximateAccuracy("data/certify/cifar10/PGD_2steps/eps_127/cifar10/resnet110/noise_0.25/test/sigma_0.25"), "$\sigma = 0.25$ | $\epsilon = 0.50$"),
Line(ApproximateAccuracy("data/certify/cifar10/PGD_2steps/eps_255/cifar10/resnet110/noise_0.25/test/sigma_0.25"), "$\sigma = 0.25$ | $\epsilon = 1.00$"),
Line(ApproximateAccuracy("data/certify/cifar10/PGD_2steps/eps_512/cifar10/resnet110/noise_0.25/test/sigma_0.25"), "$\sigma = 0.25$ | $\epsilon = 2.00$"),

Line(ApproximateAccuracy("data/certify/cifar10/PGD_2steps/eps_64/cifar10/resnet110/noise_0.50/test/sigma_0.50"), "$\sigma = 0.50$ | $\epsilon = 0.25$"),
Line(ApproximateAccuracy("data/certify/cifar10/PGD_2steps/eps_127/cifar10/resnet110/noise_0.50/test/sigma_0.50"), "$\sigma = 0.50$ | $\epsilon = 0.50$"),
Line(ApproximateAccuracy("data/certify/cifar10/PGD_2steps/eps_255/cifar10/resnet110/noise_0.50/test/sigma_0.50"), "$\sigma = 0.50$ | $\epsilon = 1.00$"),
Line(ApproximateAccuracy("data/certify/cifar10/PGD_2steps/eps_512/cifar10/resnet110/noise_0.50/test/sigma_0.50"), "$\sigma = 0.50$ | $\epsilon = 2.00$"),

Line(ApproximateAccuracy("data/certify/cifar10/PGD_2steps/eps_64/cifar10/resnet110/noise_1.00/test/sigma_1.00"), "$\sigma = 1.00$ | $\epsilon = 0.25$"),            
Line(ApproximateAccuracy("data/certify/cifar10/PGD_2steps/eps_127/cifar10/resnet110/noise_1.00/test/sigma_1.00"), "$\sigma = 1.00$ | $\epsilon = 0.50$"),
Line(ApproximateAccuracy("data/certify/cifar10/PGD_2steps/eps_255/cifar10/resnet110/noise_1.00/test/sigma_1.00"), "$\sigma = 1.00$ | $\epsilon = 1.00$"),
Line(ApproximateAccuracy("data/certify/cifar10/PGD_2steps/eps_512/cifar10/resnet110/noise_1.00/test/sigma_1.00"), "$\sigma = 1.00$ | $\epsilon = 2.00$"),
]



DDN_4steps = [
Line(ApproximateAccuracy("data/certify/cifar10/DDN_4steps/eps_64/cifar10/resnet110/noise_0.12/test/sigma_0.12"), "$\sigma = 0.12$ | $\epsilon = 0.25$"),
Line(ApproximateAccuracy("data/certify/cifar10/DDN_4steps/eps_127/cifar10/resnet110/noise_0.12/test/sigma_0.12"), "$\sigma = 0.12$ | $\epsilon = 0.50$"),
Line(ApproximateAccuracy("data/certify/cifar10/DDN_4steps/eps_255/cifar10/resnet110/noise_0.12/test/sigma_0.12"), "$\sigma = 0.12$ | $\epsilon = 1.00$"),
Line(ApproximateAccuracy("data/certify/cifar10/DDN_4steps/eps_512/cifar10/resnet110/noise_0.12/test/sigma_0.12"), "$\sigma = 0.12$ | $\epsilon = 2.00$"),            

Line(ApproximateAccuracy("data/certify/cifar10/DDN_4steps/eps_64/cifar10/resnet110/noise_0.25/test/sigma_0.25"), "$\sigma = 0.25$ | $\epsilon = 0.25$"),
Line(ApproximateAccuracy("data/certify/cifar10/DDN_4steps/eps_127/cifar10/resnet110/noise_0.25/test/sigma_0.25"), "$\sigma = 0.25$ | $\epsilon = 0.50$"),
Line(ApproximateAccuracy("data/certify/cifar10/DDN_4steps/eps_255/cifar10/resnet110/noise_0.25/test/sigma_0.25"), "$\sigma = 0.25$ | $\epsilon = 1.00$"),
Line(ApproximateAccuracy("data/certify/cifar10/DDN_4steps/eps_512/cifar10/resnet110/noise_0.25/test/sigma_0.25"), "$\sigma = 0.25$ | $\epsilon = 2.00$"),

Line(ApproximateAccuracy("data/certify/cifar10/DDN_4steps/eps_64/cifar10/resnet110/noise_0.50/test/sigma_0.50"), "$\sigma = 0.50$ | $\epsilon = 0.25$"),
Line(ApproximateAccuracy("data/certify/cifar10/DDN_4steps/eps_127/cifar10/resnet110/noise_0.50/test/sigma_0.50"), "$\sigma = 0.50$ | $\epsilon = 0.50$"),
Line(ApproximateAccuracy("data/certify/cifar10/DDN_4steps/eps_255/cifar10/resnet110/noise_0.50/test/sigma_0.50"), "$\sigma = 0.50$ | $\epsilon = 1.00$"),
Line(ApproximateAccuracy("data/certify/cifar10/DDN_4steps/eps_512/cifar10/resnet110/noise_0.50/test/sigma_0.50"), "$\sigma = 0.50$ | $\epsilon = 2.00$"),

Line(ApproximateAccuracy("data/certify/cifar10/DDN_4steps/eps_64/cifar10/resnet110/noise_1.00/test/sigma_1.00"), "$\sigma = 1.00$ | $\epsilon = 0.25$"),            
Line(ApproximateAccuracy("data/certify/cifar10/DDN_4steps/eps_127/cifar10/resnet110/noise_1.00/test/sigma_1.00"), "$\sigma = 1.00$ | $\epsilon = 0.50$"),
Line(ApproximateAccuracy("data/certify/cifar10/DDN_4steps/eps_255/cifar10/resnet110/noise_1.00/test/sigma_1.00"), "$\sigma = 1.00$ | $\epsilon = 1.00$"),
Line(ApproximateAccuracy("data/certify/cifar10/DDN_4steps/eps_512/cifar10/resnet110/noise_1.00/test/sigma_1.00"), "$\sigma = 1.00$ | $\epsilon = 2.00$"),
]

PGD_4steps = [
Line(ApproximateAccuracy("data/certify/cifar10/PGD_4steps/eps_64/cifar10/resnet110/noise_0.12/test/sigma_0.12"), "$\sigma = 0.12$ | $\epsilon = 0.25$"),
Line(ApproximateAccuracy("data/certify/cifar10/PGD_4steps/eps_127/cifar10/resnet110/noise_0.12/test/sigma_0.12"), "$\sigma = 0.12$ | $\epsilon = 0.50$"),
Line(ApproximateAccuracy("data/certify/cifar10/PGD_4steps/eps_255/cifar10/resnet110/noise_0.12/test/sigma_0.12"), "$\sigma = 0.12$ | $\epsilon = 1.00$"),
Line(ApproximateAccuracy("data/certify/cifar10/PGD_4steps/eps_512/cifar10/resnet110/noise_0.12/test/sigma_0.12"), "$\sigma = 0.12$ | $\epsilon = 2.00$"),            

Line(ApproximateAccuracy("data/certify/cifar10/PGD_4steps/eps_64/cifar10/resnet110/noise_0.25/test/sigma_0.25"), "$\sigma = 0.25$ | $\epsilon = 0.25$"),
Line(ApproximateAccuracy("data/certify/cifar10/PGD_4steps/eps_127/cifar10/resnet110/noise_0.25/test/sigma_0.25"), "$\sigma = 0.25$ | $\epsilon = 0.50$"),
Line(ApproximateAccuracy("data/certify/cifar10/PGD_4steps/eps_255/cifar10/resnet110/noise_0.25/test/sigma_0.25"), "$\sigma = 0.25$ | $\epsilon = 1.00$"),
Line(ApproximateAccuracy("data/certify/cifar10/PGD_4steps/eps_512/cifar10/resnet110/noise_0.25/test/sigma_0.25"), "$\sigma = 0.25$ | $\epsilon = 2.00$"),

Line(ApproximateAccuracy("data/certify/cifar10/PGD_4steps/eps_64/cifar10/resnet110/noise_0.50/test/sigma_0.50"), "$\sigma = 0.50$ | $\epsilon = 0.25$"),
Line(ApproximateAccuracy("data/certify/cifar10/PGD_4steps/eps_127/cifar10/resnet110/noise_0.50/test/sigma_0.50"), "$\sigma = 0.50$ | $\epsilon = 0.50$"),
Line(ApproximateAccuracy("data/certify/cifar10/PGD_4steps/eps_255/cifar10/resnet110/noise_0.50/test/sigma_0.50"), "$\sigma = 0.50$ | $\epsilon = 1.00$"),
Line(ApproximateAccuracy("data/certify/cifar10/PGD_4steps/eps_512/cifar10/resnet110/noise_0.50/test/sigma_0.50"), "$\sigma = 0.50$ | $\epsilon = 2.00$"),

Line(ApproximateAccuracy("data/certify/cifar10/PGD_4steps/eps_64/cifar10/resnet110/noise_1.00/test/sigma_1.00"), "$\sigma = 1.00$ | $\epsilon = 0.25$"),            
Line(ApproximateAccuracy("data/certify/cifar10/PGD_4steps/eps_127/cifar10/resnet110/noise_1.00/test/sigma_1.00"), "$\sigma = 1.00$ | $\epsilon = 0.50$"),
Line(ApproximateAccuracy("data/certify/cifar10/PGD_4steps/eps_255/cifar10/resnet110/noise_1.00/test/sigma_1.00"), "$\sigma = 1.00$ | $\epsilon = 1.00$"),
Line(ApproximateAccuracy("data/certify/cifar10/PGD_4steps/eps_512/cifar10/resnet110/noise_1.00/test/sigma_1.00"), "$\sigma = 1.00$ | $\epsilon = 2.00$"),
]




DDN_6steps = [
Line(ApproximateAccuracy("data/certify/cifar10/DDN_6steps/eps_64/cifar10/resnet110/noise_0.12/test/sigma_0.12"), "$\sigma = 0.12$ | $\epsilon = 0.25$"),
Line(ApproximateAccuracy("data/certify/cifar10/DDN_6steps/eps_127/cifar10/resnet110/noise_0.12/test/sigma_0.12"), "$\sigma = 0.12$ | $\epsilon = 0.50$"),
Line(ApproximateAccuracy("data/certify/cifar10/DDN_6steps/eps_255/cifar10/resnet110/noise_0.12/test/sigma_0.12"), "$\sigma = 0.12$ | $\epsilon = 1.00$"),
Line(ApproximateAccuracy("data/certify/cifar10/DDN_6steps/eps_512/cifar10/resnet110/noise_0.12/test/sigma_0.12"), "$\sigma = 0.12$ | $\epsilon = 2.00$"),            

Line(ApproximateAccuracy("data/certify/cifar10/DDN_6steps/eps_64/cifar10/resnet110/noise_0.25/test/sigma_0.25"), "$\sigma = 0.25$ | $\epsilon = 0.25$"),
Line(ApproximateAccuracy("data/certify/cifar10/DDN_6steps/eps_127/cifar10/resnet110/noise_0.25/test/sigma_0.25"), "$\sigma = 0.25$ | $\epsilon = 0.50$"),
Line(ApproximateAccuracy("data/certify/cifar10/DDN_6steps/eps_255/cifar10/resnet110/noise_0.25/test/sigma_0.25"), "$\sigma = 0.25$ | $\epsilon = 1.00$"),
Line(ApproximateAccuracy("data/certify/cifar10/DDN_6steps/eps_512/cifar10/resnet110/noise_0.25/test/sigma_0.25"), "$\sigma = 0.25$ | $\epsilon = 2.00$"),

Line(ApproximateAccuracy("data/certify/cifar10/DDN_6steps/eps_64/cifar10/resnet110/noise_0.50/test/sigma_0.50"), "$\sigma = 0.50$ | $\epsilon = 0.25$"),
Line(ApproximateAccuracy("data/certify/cifar10/DDN_6steps/eps_127/cifar10/resnet110/noise_0.50/test/sigma_0.50"), "$\sigma = 0.50$ | $\epsilon = 0.50$"),
Line(ApproximateAccuracy("data/certify/cifar10/DDN_6steps/eps_255/cifar10/resnet110/noise_0.50/test/sigma_0.50"), "$\sigma = 0.50$ | $\epsilon = 1.00$"),
Line(ApproximateAccuracy("data/certify/cifar10/DDN_6steps/eps_512/cifar10/resnet110/noise_0.50/test/sigma_0.50"), "$\sigma = 0.50$ | $\epsilon = 2.00$"),

Line(ApproximateAccuracy("data/certify/cifar10/DDN_6steps/eps_64/cifar10/resnet110/noise_1.00/test/sigma_1.00"), "$\sigma = 1.00$ | $\epsilon = 0.25$"),            
Line(ApproximateAccuracy("data/certify/cifar10/DDN_6steps/eps_127/cifar10/resnet110/noise_1.00/test/sigma_1.00"), "$\sigma = 1.00$ | $\epsilon = 0.50$"),
Line(ApproximateAccuracy("data/certify/cifar10/DDN_6steps/eps_255/cifar10/resnet110/noise_1.00/test/sigma_1.00"), "$\sigma = 1.00$ | $\epsilon = 1.00$"),
Line(ApproximateAccuracy("data/certify/cifar10/DDN_6steps/eps_512/cifar10/resnet110/noise_1.00/test/sigma_1.00"), "$\sigma = 1.00$ | $\epsilon = 2.00$"),
]

PGD_6steps = [
Line(ApproximateAccuracy("data/certify/cifar10/PGD_6steps/eps_64/cifar10/resnet110/noise_0.12/test/sigma_0.12"), "$\sigma = 0.12$ | $\epsilon = 0.25$"),
Line(ApproximateAccuracy("data/certify/cifar10/PGD_6steps/eps_127/cifar10/resnet110/noise_0.12/test/sigma_0.12"), "$\sigma = 0.12$ | $\epsilon = 0.50$"),
Line(ApproximateAccuracy("data/certify/cifar10/PGD_6steps/eps_255/cifar10/resnet110/noise_0.12/test/sigma_0.12"), "$\sigma = 0.12$ | $\epsilon = 1.00$"),
Line(ApproximateAccuracy("data/certify/cifar10/PGD_6steps/eps_512/cifar10/resnet110/noise_0.12/test/sigma_0.12"), "$\sigma = 0.12$ | $\epsilon = 2.00$"),            

Line(ApproximateAccuracy("data/certify/cifar10/PGD_6steps/eps_64/cifar10/resnet110/noise_0.25/test/sigma_0.25"), "$\sigma = 0.25$ | $\epsilon = 0.25$"),
Line(ApproximateAccuracy("data/certify/cifar10/PGD_6steps/eps_127/cifar10/resnet110/noise_0.25/test/sigma_0.25"), "$\sigma = 0.25$ | $\epsilon = 0.50$"),
Line(ApproximateAccuracy("data/certify/cifar10/PGD_6steps/eps_255/cifar10/resnet110/noise_0.25/test/sigma_0.25"), "$\sigma = 0.25$ | $\epsilon = 1.00$"),
Line(ApproximateAccuracy("data/certify/cifar10/PGD_6steps/eps_512/cifar10/resnet110/noise_0.25/test/sigma_0.25"), "$\sigma = 0.25$ | $\epsilon = 2.00$"),

Line(ApproximateAccuracy("data/certify/cifar10/PGD_6steps/eps_64/cifar10/resnet110/noise_0.50/test/sigma_0.50"), "$\sigma = 0.50$ | $\epsilon = 0.25$"),
Line(ApproximateAccuracy("data/certify/cifar10/PGD_6steps/eps_127/cifar10/resnet110/noise_0.50/test/sigma_0.50"), "$\sigma = 0.50$ | $\epsilon = 0.50$"),
Line(ApproximateAccuracy("data/certify/cifar10/PGD_6steps/eps_255/cifar10/resnet110/noise_0.50/test/sigma_0.50"), "$\sigma = 0.50$ | $\epsilon = 1.00$"),
Line(ApproximateAccuracy("data/certify/cifar10/PGD_6steps/eps_512/cifar10/resnet110/noise_0.50/test/sigma_0.50"), "$\sigma = 0.50$ | $\epsilon = 2.00$"),

Line(ApproximateAccuracy("data/certify/cifar10/PGD_6steps/eps_64/cifar10/resnet110/noise_1.00/test/sigma_1.00"), "$\sigma = 1.00$ | $\epsilon = 0.25$"),            
Line(ApproximateAccuracy("data/certify/cifar10/PGD_6steps/eps_127/cifar10/resnet110/noise_1.00/test/sigma_1.00"), "$\sigma = 1.00$ | $\epsilon = 0.50$"),
Line(ApproximateAccuracy("data/certify/cifar10/PGD_6steps/eps_255/cifar10/resnet110/noise_1.00/test/sigma_1.00"), "$\sigma = 1.00$ | $\epsilon = 1.00$"),
Line(ApproximateAccuracy("data/certify/cifar10/PGD_6steps/eps_512/cifar10/resnet110/noise_1.00/test/sigma_1.00"), "$\sigma = 1.00$ | $\epsilon = 2.00$"),
]

DDN_8steps = [
Line(ApproximateAccuracy("data/certify/cifar10/DDN_8steps/eps_64/cifar10/resnet110/noise_0.12/test/sigma_0.12"), "$\sigma = 0.12$ | $\epsilon = 0.25$"),
Line(ApproximateAccuracy("data/certify/cifar10/DDN_8steps/eps_127/cifar10/resnet110/noise_0.12/test/sigma_0.12"), "$\sigma = 0.12$ | $\epsilon = 0.50$"),
Line(ApproximateAccuracy("data/certify/cifar10/DDN_8steps/eps_255/cifar10/resnet110/noise_0.12/test/sigma_0.12"), "$\sigma = 0.12$ | $\epsilon = 1.00$"),
Line(ApproximateAccuracy("data/certify/cifar10/DDN_8steps/eps_512/cifar10/resnet110/noise_0.12/test/sigma_0.12"), "$\sigma = 0.12$ | $\epsilon = 2.00$"),            

Line(ApproximateAccuracy("data/certify/cifar10/DDN_8steps/eps_64/cifar10/resnet110/noise_0.25/test/sigma_0.25"), "$\sigma = 0.25$ | $\epsilon = 0.25$"),
Line(ApproximateAccuracy("data/certify/cifar10/DDN_8steps/eps_127/cifar10/resnet110/noise_0.25/test/sigma_0.25"), "$\sigma = 0.25$ | $\epsilon = 0.50$"),
Line(ApproximateAccuracy("data/certify/cifar10/DDN_8steps/eps_255/cifar10/resnet110/noise_0.25/test/sigma_0.25"), "$\sigma = 0.25$ | $\epsilon = 1.00$"),
Line(ApproximateAccuracy("data/certify/cifar10/DDN_8steps/eps_512/cifar10/resnet110/noise_0.25/test/sigma_0.25"), "$\sigma = 0.25$ | $\epsilon = 2.00$"),

Line(ApproximateAccuracy("data/certify/cifar10/DDN_8steps/eps_64/cifar10/resnet110/noise_0.50/test/sigma_0.50"), "$\sigma = 0.50$ | $\epsilon = 0.25$"),
Line(ApproximateAccuracy("data/certify/cifar10/DDN_8steps/eps_127/cifar10/resnet110/noise_0.50/test/sigma_0.50"), "$\sigma = 0.50$ | $\epsilon = 0.50$"),
Line(ApproximateAccuracy("data/certify/cifar10/DDN_8steps/eps_255/cifar10/resnet110/noise_0.50/test/sigma_0.50"), "$\sigma = 0.50$ | $\epsilon = 1.00$"),
Line(ApproximateAccuracy("data/certify/cifar10/DDN_8steps/eps_512/cifar10/resnet110/noise_0.50/test/sigma_0.50"), "$\sigma = 0.50$ | $\epsilon = 2.00$"),

Line(ApproximateAccuracy("data/certify/cifar10/DDN_8steps/eps_64/cifar10/resnet110/noise_1.00/test/sigma_1.00"), "$\sigma = 1.00$ | $\epsilon = 0.25$"),            
Line(ApproximateAccuracy("data/certify/cifar10/DDN_8steps/eps_127/cifar10/resnet110/noise_1.00/test/sigma_1.00"), "$\sigma = 1.00$ | $\epsilon = 0.50$"),
Line(ApproximateAccuracy("data/certify/cifar10/DDN_8steps/eps_255/cifar10/resnet110/noise_1.00/test/sigma_1.00"), "$\sigma = 1.00$ | $\epsilon = 1.00$"),
Line(ApproximateAccuracy("data/certify/cifar10/DDN_8steps/eps_512/cifar10/resnet110/noise_1.00/test/sigma_1.00"), "$\sigma = 1.00$ | $\epsilon = 2.00$"),
]

PGD_8steps = [
Line(ApproximateAccuracy("data/certify/cifar10/PGD_8steps/eps_64/cifar10/resnet110/noise_0.12/test/sigma_0.12"), "$\sigma = 0.12$ | $\epsilon = 0.25$"),
Line(ApproximateAccuracy("data/certify/cifar10/PGD_8steps/eps_127/cifar10/resnet110/noise_0.12/test/sigma_0.12"), "$\sigma = 0.12$ | $\epsilon = 0.50$"),
Line(ApproximateAccuracy("data/certify/cifar10/PGD_8steps/eps_255/cifar10/resnet110/noise_0.12/test/sigma_0.12"), "$\sigma = 0.12$ | $\epsilon = 1.00$"),
Line(ApproximateAccuracy("data/certify/cifar10/PGD_8steps/eps_512/cifar10/resnet110/noise_0.12/test/sigma_0.12"), "$\sigma = 0.12$ | $\epsilon = 2.00$"),            

Line(ApproximateAccuracy("data/certify/cifar10/PGD_8steps/eps_64/cifar10/resnet110/noise_0.25/test/sigma_0.25"), "$\sigma = 0.25$ | $\epsilon = 0.25$"),
Line(ApproximateAccuracy("data/certify/cifar10/PGD_8steps/eps_127/cifar10/resnet110/noise_0.25/test/sigma_0.25"), "$\sigma = 0.25$ | $\epsilon = 0.50$"),
Line(ApproximateAccuracy("data/certify/cifar10/PGD_8steps/eps_255/cifar10/resnet110/noise_0.25/test/sigma_0.25"), "$\sigma = 0.25$ | $\epsilon = 1.00$"),
Line(ApproximateAccuracy("data/certify/cifar10/PGD_8steps/eps_512/cifar10/resnet110/noise_0.25/test/sigma_0.25"), "$\sigma = 0.25$ | $\epsilon = 2.00$"),

Line(ApproximateAccuracy("data/certify/cifar10/PGD_8steps/eps_64/cifar10/resnet110/noise_0.50/test/sigma_0.50"), "$\sigma = 0.50$ | $\epsilon = 0.25$"),
Line(ApproximateAccuracy("data/certify/cifar10/PGD_8steps/eps_127/cifar10/resnet110/noise_0.50/test/sigma_0.50"), "$\sigma = 0.50$ | $\epsilon = 0.50$"),
Line(ApproximateAccuracy("data/certify/cifar10/PGD_8steps/eps_255/cifar10/resnet110/noise_0.50/test/sigma_0.50"), "$\sigma = 0.50$ | $\epsilon = 1.00$"),
Line(ApproximateAccuracy("data/certify/cifar10/PGD_8steps/eps_512/cifar10/resnet110/noise_0.50/test/sigma_0.50"), "$\sigma = 0.50$ | $\epsilon = 2.00$"),

Line(ApproximateAccuracy("data/certify/cifar10/PGD_8steps/eps_64/cifar10/resnet110/noise_1.00/test/sigma_1.00"), "$\sigma = 1.00$ | $\epsilon = 0.25$"),            
Line(ApproximateAccuracy("data/certify/cifar10/PGD_8steps/eps_127/cifar10/resnet110/noise_1.00/test/sigma_1.00"), "$\sigma = 1.00$ | $\epsilon = 0.50$"),
Line(ApproximateAccuracy("data/certify/cifar10/PGD_8steps/eps_255/cifar10/resnet110/noise_1.00/test/sigma_1.00"), "$\sigma = 1.00$ | $\epsilon = 1.00$"),
Line(ApproximateAccuracy("data/certify/cifar10/PGD_8steps/eps_512/cifar10/resnet110/noise_1.00/test/sigma_1.00"), "$\sigma = 1.00$ | $\epsilon = 2.00$"),
]




DDN_10steps = [
Line(ApproximateAccuracy("data/certify/cifar10/DDN_10steps/eps_64/cifar10/resnet110/noise_0.12/test/sigma_0.12"), "$\sigma = 0.12$ | $\epsilon = 0.25$"),
Line(ApproximateAccuracy("data/certify/cifar10/DDN_10steps/eps_127/cifar10/resnet110/noise_0.12/test/sigma_0.12"), "$\sigma = 0.12$ | $\epsilon = 0.50$"),
Line(ApproximateAccuracy("data/certify/cifar10/DDN_10steps/eps_255/cifar10/resnet110/noise_0.12/test/sigma_0.12"), "$\sigma = 0.12$ | $\epsilon = 1.00$"),
Line(ApproximateAccuracy("data/certify/cifar10/DDN_10steps/eps_512/cifar10/resnet110/noise_0.12/test/sigma_0.12"), "$\sigma = 0.12$ | $\epsilon = 2.00$"),            

Line(ApproximateAccuracy("data/certify/cifar10/DDN_10steps/eps_64/cifar10/resnet110/noise_0.25/test/sigma_0.25"), "$\sigma = 0.25$ | $\epsilon = 0.25$"),
Line(ApproximateAccuracy("data/certify/cifar10/DDN_10steps/eps_127/cifar10/resnet110/noise_0.25/test/sigma_0.25"), "$\sigma = 0.25$ | $\epsilon = 0.50$"),
Line(ApproximateAccuracy("data/certify/cifar10/DDN_10steps/eps_255/cifar10/resnet110/noise_0.25/test/sigma_0.25"), "$\sigma = 0.25$ | $\epsilon = 1.00$"),
Line(ApproximateAccuracy("data/certify/cifar10/DDN_10steps/eps_512/cifar10/resnet110/noise_0.25/test/sigma_0.25"), "$\sigma = 0.25$ | $\epsilon = 2.00$"),

Line(ApproximateAccuracy("data/certify/cifar10/DDN_10steps/eps_64/cifar10/resnet110/noise_0.50/test/sigma_0.50"), "$\sigma = 0.50$ | $\epsilon = 0.25$"),
Line(ApproximateAccuracy("data/certify/cifar10/DDN_10steps/eps_127/cifar10/resnet110/noise_0.50/test/sigma_0.50"), "$\sigma = 0.50$ | $\epsilon = 0.50$"),
Line(ApproximateAccuracy("data/certify/cifar10/DDN_10steps/eps_255/cifar10/resnet110/noise_0.50/test/sigma_0.50"), "$\sigma = 0.50$ | $\epsilon = 1.00$"),
Line(ApproximateAccuracy("data/certify/cifar10/DDN_10steps/eps_512/cifar10/resnet110/noise_0.50/test/sigma_0.50"), "$\sigma = 0.50$ | $\epsilon = 2.00$"),

Line(ApproximateAccuracy("data/certify/cifar10/DDN_10steps/eps_64/cifar10/resnet110/noise_1.00/test/sigma_1.00"), "$\sigma = 1.00$ | $\epsilon = 0.25$"),            
Line(ApproximateAccuracy("data/certify/cifar10/DDN_10steps/eps_127/cifar10/resnet110/noise_1.00/test/sigma_1.00"), "$\sigma = 1.00$ | $\epsilon = 0.50$"),
Line(ApproximateAccuracy("data/certify/cifar10/DDN_10steps/eps_255/cifar10/resnet110/noise_1.00/test/sigma_1.00"), "$\sigma = 1.00$ | $\epsilon = 1.00$"),
Line(ApproximateAccuracy("data/certify/cifar10/DDN_10steps/eps_512/cifar10/resnet110/noise_1.00/test/sigma_1.00"), "$\sigma = 1.00$ | $\epsilon = 2.00$"),
]


PGD_10steps = [
Line(ApproximateAccuracy("data/certify/cifar10/PGD_10steps/eps_64/cifar10/resnet110/noise_0.12/test/sigma_0.12"), "$\sigma = 0.12$ | $\epsilon = 0.25$"),
Line(ApproximateAccuracy("data/certify/cifar10/PGD_10steps/eps_127/cifar10/resnet110/noise_0.12/test/sigma_0.12"), "$\sigma = 0.12$ | $\epsilon = 0.50$"),
Line(ApproximateAccuracy("data/certify/cifar10/PGD_10steps/eps_255/cifar10/resnet110/noise_0.12/test/sigma_0.12"), "$\sigma = 0.12$ | $\epsilon = 1.00$"),
Line(ApproximateAccuracy("data/certify/cifar10/PGD_10steps/eps_512/cifar10/resnet110/noise_0.12/test/sigma_0.12"), "$\sigma = 0.12$ | $\epsilon = 2.00$"),            

Line(ApproximateAccuracy("data/certify/cifar10/PGD_10steps/eps_64/cifar10/resnet110/noise_0.25/test/sigma_0.25"), "$\sigma = 0.25$ | $\epsilon = 0.25$"),
Line(ApproximateAccuracy("data/certify/cifar10/PGD_10steps/eps_127/cifar10/resnet110/noise_0.25/test/sigma_0.25"), "$\sigma = 0.25$ | $\epsilon = 0.50$"),
Line(ApproximateAccuracy("data/certify/cifar10/PGD_10steps/eps_255/cifar10/resnet110/noise_0.25/test/sigma_0.25"), "$\sigma = 0.25$ | $\epsilon = 1.00$"),
Line(ApproximateAccuracy("data/certify/cifar10/PGD_10steps/eps_512/cifar10/resnet110/noise_0.25/test/sigma_0.25"), "$\sigma = 0.25$ | $\epsilon = 2.00$"),

Line(ApproximateAccuracy("data/certify/cifar10/PGD_10steps/eps_64/cifar10/resnet110/noise_0.50/test/sigma_0.50"), "$\sigma = 0.50$ | $\epsilon = 0.25$"),
Line(ApproximateAccuracy("data/certify/cifar10/PGD_10steps/eps_127/cifar10/resnet110/noise_0.50/test/sigma_0.50"), "$\sigma = 0.50$ | $\epsilon = 0.50$"),
Line(ApproximateAccuracy("data/certify/cifar10/PGD_10steps/eps_255/cifar10/resnet110/noise_0.50/test/sigma_0.50"), "$\sigma = 0.50$ | $\epsilon = 1.00$"),
Line(ApproximateAccuracy("data/certify/cifar10/PGD_10steps/eps_512/cifar10/resnet110/noise_0.50/test/sigma_0.50"), "$\sigma = 0.50$ | $\epsilon = 2.00$"),

Line(ApproximateAccuracy("data/certify/cifar10/PGD_10steps/eps_64/cifar10/resnet110/noise_1.00/test/sigma_1.00"), "$\sigma = 1.00$ | $\epsilon = 0.25$"),            
Line(ApproximateAccuracy("data/certify/cifar10/PGD_10steps/eps_127/cifar10/resnet110/noise_1.00/test/sigma_1.00"), "$\sigma = 1.00$ | $\epsilon = 0.50$"),
Line(ApproximateAccuracy("data/certify/cifar10/PGD_10steps/eps_255/cifar10/resnet110/noise_1.00/test/sigma_1.00"), "$\sigma = 1.00$ | $\epsilon = 1.00$"),
Line(ApproximateAccuracy("data/certify/cifar10/PGD_10steps/eps_512/cifar10/resnet110/noise_1.00/test/sigma_1.00"), "$\sigma = 1.00$ | $\epsilon = 2.00$"),
]


#####################################################################################################
## DDN varying number of noise samples experiments

DDN_4_steps_2_samples = [
Line(ApproximateAccuracy("data/certify/cifar10/DDN_4steps_multiNoiseSamples/2-multitrain/eps_64/cifar10/resnet110/noise_0.12/test/sigma_0.12"), "$\sigma = 0.12$ | $\epsilon = 0.25$"),
Line(ApproximateAccuracy("data/certify/cifar10/DDN_4steps_multiNoiseSamples/2-multitrain/eps_127/cifar10/resnet110/noise_0.12/test/sigma_0.12"), "$\sigma = 0.12$ | $\epsilon = 0.50$"),
Line(ApproximateAccuracy("data/certify/cifar10/DDN_4steps_multiNoiseSamples/2-multitrain/eps_255/cifar10/resnet110/noise_0.12/test/sigma_0.12"), "$\sigma = 0.12$ | $\epsilon = 1.00$"),
Line(ApproximateAccuracy("data/certify/cifar10/DDN_4steps_multiNoiseSamples/2-multitrain/eps_512/cifar10/resnet110/noise_0.12/test/sigma_0.12"), "$\sigma = 0.12$ | $\epsilon = 2.00$"),            

Line(ApproximateAccuracy("data/certify/cifar10/DDN_4steps_multiNoiseSamples/2-multitrain/eps_64/cifar10/resnet110/noise_0.25/test/sigma_0.25"), "$\sigma = 0.25$ | $\epsilon = 0.25$"),
Line(ApproximateAccuracy("data/certify/cifar10/DDN_4steps_multiNoiseSamples/2-multitrain/eps_127/cifar10/resnet110/noise_0.25/test/sigma_0.25"), "$\sigma = 0.25$ | $\epsilon = 0.50$"),
Line(ApproximateAccuracy("data/certify/cifar10/DDN_4steps_multiNoiseSamples/2-multitrain/eps_255/cifar10/resnet110/noise_0.25/test/sigma_0.25"), "$\sigma = 0.25$ | $\epsilon = 1.00$"),
Line(ApproximateAccuracy("data/certify/cifar10/DDN_4steps_multiNoiseSamples/2-multitrain/eps_512/cifar10/resnet110/noise_0.25/test/sigma_0.25"), "$\sigma = 0.25$ | $\epsilon = 2.00$"),

Line(ApproximateAccuracy("data/certify/cifar10/DDN_4steps_multiNoiseSamples/2-multitrain/eps_64/cifar10/resnet110/noise_0.50/test/sigma_0.50"), "$\sigma = 0.50$ | $\epsilon = 0.25$"),
Line(ApproximateAccuracy("data/certify/cifar10/DDN_4steps_multiNoiseSamples/2-multitrain/eps_127/cifar10/resnet110/noise_0.50/test/sigma_0.50"), "$\sigma = 0.50$ | $\epsilon = 0.50$"),
Line(ApproximateAccuracy("data/certify/cifar10/DDN_4steps_multiNoiseSamples/2-multitrain/eps_255/cifar10/resnet110/noise_0.50/test/sigma_0.50"), "$\sigma = 0.50$ | $\epsilon = 1.00$"),
Line(ApproximateAccuracy("data/certify/cifar10/DDN_4steps_multiNoiseSamples/2-multitrain/eps_512/cifar10/resnet110/noise_0.50/test/sigma_0.50"), "$\sigma = 0.50$ | $\epsilon = 2.00$"),

Line(ApproximateAccuracy("data/certify/cifar10/DDN_4steps_multiNoiseSamples/2-multitrain/eps_64/cifar10/resnet110/noise_1.00/test/sigma_1.00"), "$\sigma = 1.00$ | $\epsilon = 0.25$"),            
Line(ApproximateAccuracy("data/certify/cifar10/DDN_4steps_multiNoiseSamples/2-multitrain/eps_127/cifar10/resnet110/noise_1.00/test/sigma_1.00"), "$\sigma = 1.00$ | $\epsilon = 0.50$"),
Line(ApproximateAccuracy("data/certify/cifar10/DDN_4steps_multiNoiseSamples/2-multitrain/eps_255/cifar10/resnet110/noise_1.00/test/sigma_1.00"), "$\sigma = 1.00$ | $\epsilon = 1.00$"),
Line(ApproximateAccuracy("data/certify/cifar10/DDN_4steps_multiNoiseSamples/2-multitrain/eps_512/cifar10/resnet110/noise_1.00/test/sigma_1.00"), "$\sigma = 1.00$ | $\epsilon = 2.00$"),
]

DDN_4_steps_4_samples = [
Line(ApproximateAccuracy("data/certify/cifar10/DDN_4steps_multiNoiseSamples/4-multitrain/eps_64/cifar10/resnet110/noise_0.12/test/sigma_0.12"), "$\sigma = 0.12$ | $\epsilon = 0.25$"),
Line(ApproximateAccuracy("data/certify/cifar10/DDN_4steps_multiNoiseSamples/4-multitrain/eps_127/cifar10/resnet110/noise_0.12/test/sigma_0.12"), "$\sigma = 0.12$ | $\epsilon = 0.50$"),
Line(ApproximateAccuracy("data/certify/cifar10/DDN_4steps_multiNoiseSamples/4-multitrain/eps_255/cifar10/resnet110/noise_0.12/test/sigma_0.12"), "$\sigma = 0.12$ | $\epsilon = 1.00$"),
Line(ApproximateAccuracy("data/certify/cifar10/DDN_4steps_multiNoiseSamples/4-multitrain/eps_512/cifar10/resnet110/noise_0.12/test/sigma_0.12"), "$\sigma = 0.12$ | $\epsilon = 2.00$"),            

Line(ApproximateAccuracy("data/certify/cifar10/DDN_4steps_multiNoiseSamples/4-multitrain/eps_64/cifar10/resnet110/noise_0.25/test/sigma_0.25"), "$\sigma = 0.25$ | $\epsilon = 0.25$"),
Line(ApproximateAccuracy("data/certify/cifar10/DDN_4steps_multiNoiseSamples/4-multitrain/eps_127/cifar10/resnet110/noise_0.25/test/sigma_0.25"), "$\sigma = 0.25$ | $\epsilon = 0.50$"),
Line(ApproximateAccuracy("data/certify/cifar10/DDN_4steps_multiNoiseSamples/4-multitrain/eps_255/cifar10/resnet110/noise_0.25/test/sigma_0.25"), "$\sigma = 0.25$ | $\epsilon = 1.00$"),
Line(ApproximateAccuracy("data/certify/cifar10/DDN_4steps_multiNoiseSamples/4-multitrain/eps_512/cifar10/resnet110/noise_0.25/test/sigma_0.25"), "$\sigma = 0.25$ | $\epsilon = 2.00$"),

Line(ApproximateAccuracy("data/certify/cifar10/DDN_4steps_multiNoiseSamples/4-multitrain/eps_64/cifar10/resnet110/noise_0.50/test/sigma_0.50"), "$\sigma = 0.50$ | $\epsilon = 0.25$"),
Line(ApproximateAccuracy("data/certify/cifar10/DDN_4steps_multiNoiseSamples/4-multitrain/eps_127/cifar10/resnet110/noise_0.50/test/sigma_0.50"), "$\sigma = 0.50$ | $\epsilon = 0.50$"),
Line(ApproximateAccuracy("data/certify/cifar10/DDN_4steps_multiNoiseSamples/4-multitrain/eps_255/cifar10/resnet110/noise_0.50/test/sigma_0.50"), "$\sigma = 0.50$ | $\epsilon = 1.00$"),
Line(ApproximateAccuracy("data/certify/cifar10/DDN_4steps_multiNoiseSamples/4-multitrain/eps_512/cifar10/resnet110/noise_0.50/test/sigma_0.50"), "$\sigma = 0.50$ | $\epsilon = 2.00$"),

Line(ApproximateAccuracy("data/certify/cifar10/DDN_4steps_multiNoiseSamples/4-multitrain/eps_64/cifar10/resnet110/noise_1.00/test/sigma_1.00"), "$\sigma = 1.00$ | $\epsilon = 0.25$"),            
Line(ApproximateAccuracy("data/certify/cifar10/DDN_4steps_multiNoiseSamples/4-multitrain/eps_127/cifar10/resnet110/noise_1.00/test/sigma_1.00"), "$\sigma = 1.00$ | $\epsilon = 0.50$"),
Line(ApproximateAccuracy("data/certify/cifar10/DDN_4steps_multiNoiseSamples/4-multitrain/eps_255/cifar10/resnet110/noise_1.00/test/sigma_1.00"), "$\sigma = 1.00$ | $\epsilon = 1.00$"),
Line(ApproximateAccuracy("data/certify/cifar10/DDN_4steps_multiNoiseSamples/4-multitrain/eps_512/cifar10/resnet110/noise_1.00/test/sigma_1.00"), "$\sigma = 1.00$ | $\epsilon = 2.00$"),
]



DDN_4_steps_8_samples = [
# Line(ApproximateAccuracy("data/certify/cifar10/DDN_4steps_multiNoiseSamples/8-multitrain/eps_64/cifar10/resnet110/noise_0.12/test/sigma_0.12"), "$\sigma = 0.12$ | $\epsilon = 0.25$"),
Line(ApproximateAccuracy("data/certify/cifar10/DDN_4steps_multiNoiseSamples/8-multitrain/eps_127/cifar10/resnet110/noise_0.12/test/sigma_0.12"), "$\sigma = 0.12$ | $\epsilon = 0.50$"),
Line(ApproximateAccuracy("data/certify/cifar10/DDN_4steps_multiNoiseSamples/8-multitrain/eps_255/cifar10/resnet110/noise_0.12/test/sigma_0.12"), "$\sigma = 0.12$ | $\epsilon = 1.00$"),
Line(ApproximateAccuracy("data/certify/cifar10/DDN_4steps_multiNoiseSamples/8-multitrain/eps_512/cifar10/resnet110/noise_0.12/test/sigma_0.12"), "$\sigma = 0.12$ | $\epsilon = 2.00$"),                       

Line(ApproximateAccuracy("data/certify/cifar10/DDN_4steps_multiNoiseSamples/8-multitrain/eps_64/cifar10/resnet110/noise_0.25/test/sigma_0.25"), "$\sigma = 0.25$ | $\epsilon = 0.25$"),
Line(ApproximateAccuracy("data/certify/cifar10/DDN_4steps_multiNoiseSamples/8-multitrain/eps_127/cifar10/resnet110/noise_0.25/test/sigma_0.25"), "$\sigma = 0.25$ | $\epsilon = 0.50$"),
Line(ApproximateAccuracy("data/certify/cifar10/DDN_4steps_multiNoiseSamples/8-multitrain/eps_255/cifar10/resnet110/noise_0.25/test/sigma_0.25"), "$\sigma = 0.25$ | $\epsilon = 1.00$"),
Line(ApproximateAccuracy("data/certify/cifar10/DDN_4steps_multiNoiseSamples/8-multitrain/eps_512/cifar10/resnet110/noise_0.25/test/sigma_0.25"), "$\sigma = 0.25$ | $\epsilon = 2.00$"),

Line(ApproximateAccuracy("data/certify/cifar10/DDN_4steps_multiNoiseSamples/8-multitrain/eps_64/cifar10/resnet110/noise_0.50/test/sigma_0.50"), "$\sigma = 0.50$ | $\epsilon = 0.25$"),
Line(ApproximateAccuracy("data/certify/cifar10/DDN_4steps_multiNoiseSamples/8-multitrain/eps_127/cifar10/resnet110/noise_0.50/test/sigma_0.50"), "$\sigma = 0.50$ | $\epsilon = 0.50$"),
Line(ApproximateAccuracy("data/certify/cifar10/DDN_4steps_multiNoiseSamples/8-multitrain/eps_255/cifar10/resnet110/noise_0.50/test/sigma_0.50"), "$\sigma = 0.50$ | $\epsilon = 1.00$"),
Line(ApproximateAccuracy("data/certify/cifar10/DDN_4steps_multiNoiseSamples/8-multitrain/eps_512/cifar10/resnet110/noise_0.50/test/sigma_0.50"), "$\sigma = 0.50$ | $\epsilon = 2.00$"),

Line(ApproximateAccuracy("data/certify/cifar10/DDN_4steps_multiNoiseSamples/8-multitrain/eps_64/cifar10/resnet110/noise_1.00/test/sigma_1.00"), "$\sigma = 1.00$ | $\epsilon = 0.25$"),            
Line(ApproximateAccuracy("data/certify/cifar10/DDN_4steps_multiNoiseSamples/8-multitrain/eps_127/cifar10/resnet110/noise_1.00/test/sigma_1.00"), "$\sigma = 1.00$ | $\epsilon = 0.50$"),
Line(ApproximateAccuracy("data/certify/cifar10/DDN_4steps_multiNoiseSamples/8-multitrain/eps_255/cifar10/resnet110/noise_1.00/test/sigma_1.00"), "$\sigma = 1.00$ | $\epsilon = 1.00$"),
Line(ApproximateAccuracy("data/certify/cifar10/DDN_4steps_multiNoiseSamples/8-multitrain/eps_512/cifar10/resnet110/noise_1.00/test/sigma_1.00"), "$\sigma = 1.00$ | $\epsilon = 2.00$"),
]

DDN_4_steps_16_samples = [
Line(ApproximateAccuracy("data/certify/cifar10/DDN_4steps_multiNoiseSamples/16-multitrain/eps_64/cifar10/resnet110/noise_0.12/test/sigma_0.12"), "$\sigma = 0.12$ | $\epsilon = 0.25$"),
Line(ApproximateAccuracy("data/certify/cifar10/DDN_4steps_multiNoiseSamples/16-multitrain/eps_127/cifar10/resnet110/noise_0.12/test/sigma_0.12"), "$\sigma = 0.12$ | $\epsilon = 0.50$"),
Line(ApproximateAccuracy("data/certify/cifar10/DDN_4steps_multiNoiseSamples/16-multitrain/eps_255/cifar10/resnet110/noise_0.12/test/sigma_0.12"), "$\sigma = 0.12$ | $\epsilon = 1.00$"),
Line(ApproximateAccuracy("data/certify/cifar10/DDN_4steps_multiNoiseSamples/16-multitrain/eps_512/cifar10/resnet110/noise_0.12/test/sigma_0.12"), "$\sigma = 0.12$ | $\epsilon = 2.00$"),                       

Line(ApproximateAccuracy("data/certify/cifar10/DDN_4steps_multiNoiseSamples/16-multitrain/eps_64/cifar10/resnet110/noise_0.25/test/sigma_0.25"), "$\sigma = 0.25$ | $\epsilon = 0.25$"),
Line(ApproximateAccuracy("data/certify/cifar10/DDN_4steps_multiNoiseSamples/16-multitrain/eps_127/cifar10/resnet110/noise_0.25/test/sigma_0.25"), "$\sigma = 0.25$ | $\epsilon = 0.50$"),
Line(ApproximateAccuracy("data/certify/cifar10/DDN_4steps_multiNoiseSamples/16-multitrain/eps_255/cifar10/resnet110/noise_0.25/test/sigma_0.25"), "$\sigma = 0.25$ | $\epsilon = 1.00$"),
Line(ApproximateAccuracy("data/certify/cifar10/DDN_4steps_multiNoiseSamples/16-multitrain/eps_512/cifar10/resnet110/noise_0.25/test/sigma_0.25"), "$\sigma = 0.25$ | $\epsilon = 2.00$"),

Line(ApproximateAccuracy("data/certify/cifar10/DDN_4steps_multiNoiseSamples/16-multitrain/eps_64/cifar10/resnet110/noise_0.50/test/sigma_0.50"), "$\sigma = 0.50$ | $\epsilon = 0.25$"),
Line(ApproximateAccuracy("data/certify/cifar10/DDN_4steps_multiNoiseSamples/16-multitrain/eps_127/cifar10/resnet110/noise_0.50/test/sigma_0.50"), "$\sigma = 0.50$ | $\epsilon = 0.50$"),
Line(ApproximateAccuracy("data/certify/cifar10/DDN_4steps_multiNoiseSamples/16-multitrain/eps_255/cifar10/resnet110/noise_0.50/test/sigma_0.50"), "$\sigma = 0.50$ | $\epsilon = 1.00$"),
Line(ApproximateAccuracy("data/certify/cifar10/DDN_4steps_multiNoiseSamples/16-multitrain/eps_512/cifar10/resnet110/noise_0.50/test/sigma_0.50"), "$\sigma = 0.50$ | $\epsilon = 2.00$"),

Line(ApproximateAccuracy("data/certify/cifar10/DDN_4steps_multiNoiseSamples/16-multitrain/eps_64/cifar10/resnet110/noise_1.00/test/sigma_1.00"), "$\sigma = 1.00$ | $\epsilon = 0.25$"),            
Line(ApproximateAccuracy("data/certify/cifar10/DDN_4steps_multiNoiseSamples/16-multitrain/eps_127/cifar10/resnet110/noise_1.00/test/sigma_1.00"), "$\sigma = 1.00$ | $\epsilon = 0.50$"),
Line(ApproximateAccuracy("data/certify/cifar10/DDN_4steps_multiNoiseSamples/16-multitrain/eps_255/cifar10/resnet110/noise_1.00/test/sigma_1.00"), "$\sigma = 1.00$ | $\epsilon = 1.00$"),
Line(ApproximateAccuracy("data/certify/cifar10/DDN_4steps_multiNoiseSamples/16-multitrain/eps_512/cifar10/resnet110/noise_1.00/test/sigma_1.00"), "$\sigma = 1.00$ | $\epsilon = 2.00$"),
]




DDN_10_steps_2_samples = [
Line(ApproximateAccuracy("data/certify/cifar10/DDN_10steps_multiNoiseSamples/2-multitrain/eps_64/cifar10/resnet110/noise_0.12/test/sigma_0.12"), "$\sigma = 0.12$ | $\epsilon = 0.25$"),
Line(ApproximateAccuracy("data/certify/cifar10/DDN_10steps_multiNoiseSamples/2-multitrain/eps_127/cifar10/resnet110/noise_0.12/test/sigma_0.12"), "$\sigma = 0.12$ | $\epsilon = 0.50$"),
Line(ApproximateAccuracy("data/certify/cifar10/DDN_10steps_multiNoiseSamples/2-multitrain/eps_255/cifar10/resnet110/noise_0.12/test/sigma_0.12"), "$\sigma = 0.12$ | $\epsilon = 1.00$"),
Line(ApproximateAccuracy("data/certify/cifar10/DDN_10steps_multiNoiseSamples/2-multitrain/eps_512/cifar10/resnet110/noise_0.12/test/sigma_0.12"), "$\sigma = 0.12$ | $\epsilon = 2.00$"),            

Line(ApproximateAccuracy("data/certify/cifar10/DDN_10steps_multiNoiseSamples/2-multitrain/eps_64/cifar10/resnet110/noise_0.25/test/sigma_0.25"), "$\sigma = 0.25$ | $\epsilon = 0.25$"),
Line(ApproximateAccuracy("data/certify/cifar10/DDN_10steps_multiNoiseSamples/2-multitrain/eps_127/cifar10/resnet110/noise_0.25/test/sigma_0.25"), "$\sigma = 0.25$ | $\epsilon = 0.50$"),
Line(ApproximateAccuracy("data/certify/cifar10/DDN_10steps_multiNoiseSamples/2-multitrain/eps_255/cifar10/resnet110/noise_0.25/test/sigma_0.25"), "$\sigma = 0.25$ | $\epsilon = 1.00$"),
Line(ApproximateAccuracy("data/certify/cifar10/DDN_10steps_multiNoiseSamples/2-multitrain/eps_512/cifar10/resnet110/noise_0.25/test/sigma_0.25"), "$\sigma = 0.25$ | $\epsilon = 2.00$"),

Line(ApproximateAccuracy("data/certify/cifar10/DDN_10steps_multiNoiseSamples/2-multitrain/eps_64/cifar10/resnet110/noise_0.50/test/sigma_0.50"), "$\sigma = 0.50$ | $\epsilon = 0.25$"),
Line(ApproximateAccuracy("data/certify/cifar10/DDN_10steps_multiNoiseSamples/2-multitrain/eps_127/cifar10/resnet110/noise_0.50/test/sigma_0.50"), "$\sigma = 0.50$ | $\epsilon = 0.50$"),
Line(ApproximateAccuracy("data/certify/cifar10/DDN_10steps_multiNoiseSamples/2-multitrain/eps_255/cifar10/resnet110/noise_0.50/test/sigma_0.50"), "$\sigma = 0.50$ | $\epsilon = 1.00$"),
Line(ApproximateAccuracy("data/certify/cifar10/DDN_10steps_multiNoiseSamples/2-multitrain/eps_512/cifar10/resnet110/noise_0.50/test/sigma_0.50"), "$\sigma = 0.50$ | $\epsilon = 2.00$"),

Line(ApproximateAccuracy("data/certify/cifar10/DDN_10steps_multiNoiseSamples/2-multitrain/eps_64/cifar10/resnet110/noise_1.00/test/sigma_1.00"), "$\sigma = 1.00$ | $\epsilon = 0.25$"),            
Line(ApproximateAccuracy("data/certify/cifar10/DDN_10steps_multiNoiseSamples/2-multitrain/eps_127/cifar10/resnet110/noise_1.00/test/sigma_1.00"), "$\sigma = 1.00$ | $\epsilon = 0.50$"),
Line(ApproximateAccuracy("data/certify/cifar10/DDN_10steps_multiNoiseSamples/2-multitrain/eps_255/cifar10/resnet110/noise_1.00/test/sigma_1.00"), "$\sigma = 1.00$ | $\epsilon = 1.00$"),
Line(ApproximateAccuracy("data/certify/cifar10/DDN_10steps_multiNoiseSamples/2-multitrain/eps_512/cifar10/resnet110/noise_1.00/test/sigma_1.00"), "$\sigma = 1.00$ | $\epsilon = 2.00$"),
]

DDN_10_steps_4_samples = [
Line(ApproximateAccuracy("data/certify/cifar10/DDN_10steps_multiNoiseSamples/4-multitrain/eps_64/cifar10/resnet110/noise_0.12/test/sigma_0.12"), "$\sigma = 0.12$ | $\epsilon = 0.25$"),
Line(ApproximateAccuracy("data/certify/cifar10/DDN_10steps_multiNoiseSamples/4-multitrain/eps_127/cifar10/resnet110/noise_0.12/test/sigma_0.12"), "$\sigma = 0.12$ | $\epsilon = 0.50$"),
Line(ApproximateAccuracy("data/certify/cifar10/DDN_10steps_multiNoiseSamples/4-multitrain/eps_255/cifar10/resnet110/noise_0.12/test/sigma_0.12"), "$\sigma = 0.12$ | $\epsilon = 1.00$"),
Line(ApproximateAccuracy("data/certify/cifar10/DDN_10steps_multiNoiseSamples/4-multitrain/eps_512/cifar10/resnet110/noise_0.12/test/sigma_0.12"), "$\sigma = 0.12$ | $\epsilon = 2.00$"),            

Line(ApproximateAccuracy("data/certify/cifar10/DDN_10steps_multiNoiseSamples/4-multitrain/eps_64/cifar10/resnet110/noise_0.25/test/sigma_0.25"), "$\sigma = 0.25$ | $\epsilon = 0.25$"),
Line(ApproximateAccuracy("data/certify/cifar10/DDN_10steps_multiNoiseSamples/4-multitrain/eps_127/cifar10/resnet110/noise_0.25/test/sigma_0.25"), "$\sigma = 0.25$ | $\epsilon = 0.50$"),
Line(ApproximateAccuracy("data/certify/cifar10/DDN_10steps_multiNoiseSamples/4-multitrain/eps_255/cifar10/resnet110/noise_0.25/test/sigma_0.25"), "$\sigma = 0.25$ | $\epsilon = 1.00$"),
Line(ApproximateAccuracy("data/certify/cifar10/DDN_10steps_multiNoiseSamples/4-multitrain/eps_512/cifar10/resnet110/noise_0.25/test/sigma_0.25"), "$\sigma = 0.25$ | $\epsilon = 2.00$"),

Line(ApproximateAccuracy("data/certify/cifar10/DDN_10steps_multiNoiseSamples/4-multitrain/eps_64/cifar10/resnet110/noise_0.50/test/sigma_0.50"), "$\sigma = 0.50$ | $\epsilon = 0.25$"),
Line(ApproximateAccuracy("data/certify/cifar10/DDN_10steps_multiNoiseSamples/4-multitrain/eps_127/cifar10/resnet110/noise_0.50/test/sigma_0.50"), "$\sigma = 0.50$ | $\epsilon = 0.50$"),
Line(ApproximateAccuracy("data/certify/cifar10/DDN_10steps_multiNoiseSamples/4-multitrain/eps_255/cifar10/resnet110/noise_0.50/test/sigma_0.50"), "$\sigma = 0.50$ | $\epsilon = 1.00$"),
Line(ApproximateAccuracy("data/certify/cifar10/DDN_10steps_multiNoiseSamples/4-multitrain/eps_512/cifar10/resnet110/noise_0.50/test/sigma_0.50"), "$\sigma = 0.50$ | $\epsilon = 2.00$"),

Line(ApproximateAccuracy("data/certify/cifar10/DDN_10steps_multiNoiseSamples/4-multitrain/eps_64/cifar10/resnet110/noise_1.00/test/sigma_1.00"), "$\sigma = 1.00$ | $\epsilon = 0.25$"),            
Line(ApproximateAccuracy("data/certify/cifar10/DDN_10steps_multiNoiseSamples/4-multitrain/eps_127/cifar10/resnet110/noise_1.00/test/sigma_1.00"), "$\sigma = 1.00$ | $\epsilon = 0.50$"),
Line(ApproximateAccuracy("data/certify/cifar10/DDN_10steps_multiNoiseSamples/4-multitrain/eps_255/cifar10/resnet110/noise_1.00/test/sigma_1.00"), "$\sigma = 1.00$ | $\epsilon = 1.00$"),
Line(ApproximateAccuracy("data/certify/cifar10/DDN_10steps_multiNoiseSamples/4-multitrain/eps_512/cifar10/resnet110/noise_1.00/test/sigma_1.00"), "$\sigma = 1.00$ | $\epsilon = 2.00$"),
]

DDN_10_steps_8_samples = [
Line(ApproximateAccuracy("data/certify/cifar10/DDN_10steps_multiNoiseSamples/8-multitrain/eps_64/cifar10/resnet110/noise_0.12/test/sigma_0.12"), "$\sigma = 0.12$ | $\epsilon = 0.25$"),
Line(ApproximateAccuracy("data/certify/cifar10/DDN_10steps_multiNoiseSamples/8-multitrain/eps_127/cifar10/resnet110/noise_0.12/test/sigma_0.12"), "$\sigma = 0.12$ | $\epsilon = 0.50$"),
Line(ApproximateAccuracy("data/certify/cifar10/DDN_10steps_multiNoiseSamples/8-multitrain/eps_255/cifar10/resnet110/noise_0.12/test/sigma_0.12"), "$\sigma = 0.12$ | $\epsilon = 1.00$"),
Line(ApproximateAccuracy("data/certify/cifar10/DDN_10steps_multiNoiseSamples/8-multitrain/eps_512/cifar10/resnet110/noise_0.12/test/sigma_0.12"), "$\sigma = 0.12$ | $\epsilon = 2.00$"),            

Line(ApproximateAccuracy("data/certify/cifar10/DDN_10steps_multiNoiseSamples/8-multitrain/eps_64/cifar10/resnet110/noise_0.25/test/sigma_0.25"), "$\sigma = 0.25$ | $\epsilon = 0.25$"),
Line(ApproximateAccuracy("data/certify/cifar10/DDN_10steps_multiNoiseSamples/8-multitrain/eps_127/cifar10/resnet110/noise_0.25/test/sigma_0.25"), "$\sigma = 0.25$ | $\epsilon = 0.50$"),
Line(ApproximateAccuracy("data/certify/cifar10/DDN_10steps_multiNoiseSamples/8-multitrain/eps_255/cifar10/resnet110/noise_0.25/test/sigma_0.25"), "$\sigma = 0.25$ | $\epsilon = 1.00$"),
Line(ApproximateAccuracy("data/certify/cifar10/DDN_10steps_multiNoiseSamples/8-multitrain/eps_512/cifar10/resnet110/noise_0.25/test/sigma_0.25"), "$\sigma = 0.25$ | $\epsilon = 2.00$"),

Line(ApproximateAccuracy("data/certify/cifar10/DDN_10steps_multiNoiseSamples/8-multitrain/eps_64/cifar10/resnet110/noise_0.50/test/sigma_0.50"), "$\sigma = 0.50$ | $\epsilon = 0.25$"),
Line(ApproximateAccuracy("data/certify/cifar10/DDN_10steps_multiNoiseSamples/8-multitrain/eps_127/cifar10/resnet110/noise_0.50/test/sigma_0.50"), "$\sigma = 0.50$ | $\epsilon = 0.50$"),
Line(ApproximateAccuracy("data/certify/cifar10/DDN_10steps_multiNoiseSamples/8-multitrain/eps_255/cifar10/resnet110/noise_0.50/test/sigma_0.50"), "$\sigma = 0.50$ | $\epsilon = 1.00$"),
Line(ApproximateAccuracy("data/certify/cifar10/DDN_10steps_multiNoiseSamples/8-multitrain/eps_512/cifar10/resnet110/noise_0.50/test/sigma_0.50"), "$\sigma = 0.50$ | $\epsilon = 2.00$"),

Line(ApproximateAccuracy("data/certify/cifar10/DDN_10steps_multiNoiseSamples/8-multitrain/eps_64/cifar10/resnet110/noise_1.00/test/sigma_1.00"), "$\sigma = 1.00$ | $\epsilon = 0.25$"),            
Line(ApproximateAccuracy("data/certify/cifar10/DDN_10steps_multiNoiseSamples/8-multitrain/eps_127/cifar10/resnet110/noise_1.00/test/sigma_1.00"), "$\sigma = 1.00$ | $\epsilon = 0.50$"),
Line(ApproximateAccuracy("data/certify/cifar10/DDN_10steps_multiNoiseSamples/8-multitrain/eps_255/cifar10/resnet110/noise_1.00/test/sigma_1.00"), "$\sigma = 1.00$ | $\epsilon = 1.00$"),
Line(ApproximateAccuracy("data/certify/cifar10/DDN_10steps_multiNoiseSamples/8-multitrain/eps_512/cifar10/resnet110/noise_1.00/test/sigma_1.00"), "$\sigma = 1.00$ | $\epsilon = 2.00$"),
]

DDN_10_steps_16_samples = [
Line(ApproximateAccuracy("data/certify/cifar10/DDN_10steps_multiNoiseSamples/16-multitrain/eps_64/cifar10/resnet110/noise_0.12/test/sigma_0.12"), "$\sigma = 0.12$ | $\epsilon = 0.25$"),
Line(ApproximateAccuracy("data/certify/cifar10/DDN_10steps_multiNoiseSamples/16-multitrain/eps_127/cifar10/resnet110/noise_0.12/test/sigma_0.12"), "$\sigma = 0.12$ | $\epsilon = 0.50$"),
Line(ApproximateAccuracy("data/certify/cifar10/DDN_10steps_multiNoiseSamples/16-multitrain/eps_255/cifar10/resnet110/noise_0.12/test/sigma_0.12"), "$\sigma = 0.12$ | $\epsilon = 1.00$"),
Line(ApproximateAccuracy("data/certify/cifar10/DDN_10steps_multiNoiseSamples/16-multitrain/eps_512/cifar10/resnet110/noise_0.12/test/sigma_0.12"), "$\sigma = 0.12$ | $\epsilon = 2.00$"),            

Line(ApproximateAccuracy("data/certify/cifar10/DDN_10steps_multiNoiseSamples/16-multitrain/eps_64/cifar10/resnet110/noise_0.25/test/sigma_0.25"), "$\sigma = 0.25$ | $\epsilon = 0.25$"),
Line(ApproximateAccuracy("data/certify/cifar10/DDN_10steps_multiNoiseSamples/16-multitrain/eps_127/cifar10/resnet110/noise_0.25/test/sigma_0.25"), "$\sigma = 0.25$ | $\epsilon = 0.50$"),
Line(ApproximateAccuracy("data/certify/cifar10/DDN_10steps_multiNoiseSamples/16-multitrain/eps_255/cifar10/resnet110/noise_0.25/test/sigma_0.25"), "$\sigma = 0.25$ | $\epsilon = 1.00$"),
Line(ApproximateAccuracy("data/certify/cifar10/DDN_10steps_multiNoiseSamples/16-multitrain/eps_512/cifar10/resnet110/noise_0.25/test/sigma_0.25"), "$\sigma = 0.25$ | $\epsilon = 2.00$"),

Line(ApproximateAccuracy("data/certify/cifar10/DDN_10steps_multiNoiseSamples/16-multitrain/eps_64/cifar10/resnet110/noise_0.50/test/sigma_0.50"), "$\sigma = 0.50$ | $\epsilon = 0.25$"),
Line(ApproximateAccuracy("data/certify/cifar10/DDN_10steps_multiNoiseSamples/16-multitrain/eps_127/cifar10/resnet110/noise_0.50/test/sigma_0.50"), "$\sigma = 0.50$ | $\epsilon = 0.50$"),
Line(ApproximateAccuracy("data/certify/cifar10/DDN_10steps_multiNoiseSamples/16-multitrain/eps_255/cifar10/resnet110/noise_0.50/test/sigma_0.50"), "$\sigma = 0.50$ | $\epsilon = 1.00$"),
Line(ApproximateAccuracy("data/certify/cifar10/DDN_10steps_multiNoiseSamples/16-multitrain/eps_512/cifar10/resnet110/noise_0.50/test/sigma_0.50"), "$\sigma = 0.50$ | $\epsilon = 2.00$"),

Line(ApproximateAccuracy("data/certify/cifar10/DDN_10steps_multiNoiseSamples/16-multitrain/eps_64/cifar10/resnet110/noise_1.00/test/sigma_1.00"), "$\sigma = 1.00$ | $\epsilon = 0.25$"),            
Line(ApproximateAccuracy("data/certify/cifar10/DDN_10steps_multiNoiseSamples/16-multitrain/eps_127/cifar10/resnet110/noise_1.00/test/sigma_1.00"), "$\sigma = 1.00$ | $\epsilon = 0.50$"),
Line(ApproximateAccuracy("data/certify/cifar10/DDN_10steps_multiNoiseSamples/16-multitrain/eps_255/cifar10/resnet110/noise_1.00/test/sigma_1.00"), "$\sigma = 1.00$ | $\epsilon = 1.00$"),
Line(ApproximateAccuracy("data/certify/cifar10/DDN_10steps_multiNoiseSamples/16-multitrain/eps_512/cifar10/resnet110/noise_1.00/test/sigma_1.00"), "$\sigma = 1.00$ | $\epsilon = 2.00$"),
]


#####################################################################################################
## PGD varying number of noise samples experiments

PGD_2_steps_2_samples = [
Line(ApproximateAccuracy("data/certify/cifar10/PGD_2steps_multiNoiseSamples/2-multitrain/eps_64/cifar10/resnet110/noise_0.12/test/sigma_0.12"), "$\sigma = 0.12$ | $\epsilon = 0.25$"),
Line(ApproximateAccuracy("data/certify/cifar10/PGD_2steps_multiNoiseSamples/2-multitrain/eps_127/cifar10/resnet110/noise_0.12/test/sigma_0.12"), "$\sigma = 0.12$ | $\epsilon = 0.50$"),
Line(ApproximateAccuracy("data/certify/cifar10/PGD_2steps_multiNoiseSamples/2-multitrain/eps_255/cifar10/resnet110/noise_0.12/test/sigma_0.12"), "$\sigma = 0.12$ | $\epsilon = 1.00$"),
Line(ApproximateAccuracy("data/certify/cifar10/PGD_2steps_multiNoiseSamples/2-multitrain/eps_512/cifar10/resnet110/noise_0.12/test/sigma_0.12"), "$\sigma = 0.12$ | $\epsilon = 2.00$"),            

Line(ApproximateAccuracy("data/certify/cifar10/PGD_2steps_multiNoiseSamples/2-multitrain/eps_64/cifar10/resnet110/noise_0.25/test/sigma_0.25"), "$\sigma = 0.25$ | $\epsilon = 0.25$"),
Line(ApproximateAccuracy("data/certify/cifar10/PGD_2steps_multiNoiseSamples/2-multitrain/eps_127/cifar10/resnet110/noise_0.25/test/sigma_0.25"), "$\sigma = 0.25$ | $\epsilon = 0.50$"),
Line(ApproximateAccuracy("data/certify/cifar10/PGD_2steps_multiNoiseSamples/2-multitrain/eps_255/cifar10/resnet110/noise_0.25/test/sigma_0.25"), "$\sigma = 0.25$ | $\epsilon = 1.00$"),
Line(ApproximateAccuracy("data/certify/cifar10/PGD_2steps_multiNoiseSamples/2-multitrain/eps_512/cifar10/resnet110/noise_0.25/test/sigma_0.25"), "$\sigma = 0.25$ | $\epsilon = 2.00$"),

Line(ApproximateAccuracy("data/certify/cifar10/PGD_2steps_multiNoiseSamples/2-multitrain/eps_64/cifar10/resnet110/noise_0.50/test/sigma_0.50"), "$\sigma = 0.50$ | $\epsilon = 0.25$"),
Line(ApproximateAccuracy("data/certify/cifar10/PGD_2steps_multiNoiseSamples/2-multitrain/eps_127/cifar10/resnet110/noise_0.50/test/sigma_0.50"), "$\sigma = 0.50$ | $\epsilon = 0.50$"),
Line(ApproximateAccuracy("data/certify/cifar10/PGD_2steps_multiNoiseSamples/2-multitrain/eps_255/cifar10/resnet110/noise_0.50/test/sigma_0.50"), "$\sigma = 0.50$ | $\epsilon = 1.00$"),
Line(ApproximateAccuracy("data/certify/cifar10/PGD_2steps_multiNoiseSamples/2-multitrain/eps_512/cifar10/resnet110/noise_0.50/test/sigma_0.50"), "$\sigma = 0.50$ | $\epsilon = 2.00$"),

Line(ApproximateAccuracy("data/certify/cifar10/PGD_2steps_multiNoiseSamples/2-multitrain/eps_64/cifar10/resnet110/noise_1.00/test/sigma_1.00"), "$\sigma = 1.00$ | $\epsilon = 0.25$"),            
Line(ApproximateAccuracy("data/certify/cifar10/PGD_2steps_multiNoiseSamples/2-multitrain/eps_127/cifar10/resnet110/noise_1.00/test/sigma_1.00"), "$\sigma = 1.00$ | $\epsilon = 0.50$"),
Line(ApproximateAccuracy("data/certify/cifar10/PGD_2steps_multiNoiseSamples/2-multitrain/eps_255/cifar10/resnet110/noise_1.00/test/sigma_1.00"), "$\sigma = 1.00$ | $\epsilon = 1.00$"),
Line(ApproximateAccuracy("data/certify/cifar10/PGD_2steps_multiNoiseSamples/2-multitrain/eps_512/cifar10/resnet110/noise_1.00/test/sigma_1.00"), "$\sigma = 1.00$ | $\epsilon = 2.00$"),
]

PGD_2_steps_4_samples = [
Line(ApproximateAccuracy("data/certify/cifar10/PGD_2steps_multiNoiseSamples/4-multitrain/eps_64/cifar10/resnet110/noise_0.12/test/sigma_0.12"), "$\sigma = 0.12$ | $\epsilon = 0.25$"),
Line(ApproximateAccuracy("data/certify/cifar10/PGD_2steps_multiNoiseSamples/4-multitrain/eps_127/cifar10/resnet110/noise_0.12/test/sigma_0.12"), "$\sigma = 0.12$ | $\epsilon = 0.50$"),
Line(ApproximateAccuracy("data/certify/cifar10/PGD_2steps_multiNoiseSamples/4-multitrain/eps_255/cifar10/resnet110/noise_0.12/test/sigma_0.12"), "$\sigma = 0.12$ | $\epsilon = 1.00$"),
Line(ApproximateAccuracy("data/certify/cifar10/PGD_2steps_multiNoiseSamples/4-multitrain/eps_512/cifar10/resnet110/noise_0.12/test/sigma_0.12"), "$\sigma = 0.12$ | $\epsilon = 2.00$"),            

Line(ApproximateAccuracy("data/certify/cifar10/PGD_2steps_multiNoiseSamples/4-multitrain/eps_64/cifar10/resnet110/noise_0.25/test/sigma_0.25"), "$\sigma = 0.25$ | $\epsilon = 0.25$"),
Line(ApproximateAccuracy("data/certify/cifar10/PGD_2steps_multiNoiseSamples/4-multitrain/eps_127/cifar10/resnet110/noise_0.25/test/sigma_0.25"), "$\sigma = 0.25$ | $\epsilon = 0.50$"),
Line(ApproximateAccuracy("data/certify/cifar10/PGD_2steps_multiNoiseSamples/4-multitrain/eps_255/cifar10/resnet110/noise_0.25/test/sigma_0.25"), "$\sigma = 0.25$ | $\epsilon = 1.00$"),
Line(ApproximateAccuracy("data/certify/cifar10/PGD_2steps_multiNoiseSamples/4-multitrain/eps_512/cifar10/resnet110/noise_0.25/test/sigma_0.25"), "$\sigma = 0.25$ | $\epsilon = 2.00$"),

Line(ApproximateAccuracy("data/certify/cifar10/PGD_2steps_multiNoiseSamples/4-multitrain/eps_64/cifar10/resnet110/noise_0.50/test/sigma_0.50"), "$\sigma = 0.50$ | $\epsilon = 0.25$"),
Line(ApproximateAccuracy("data/certify/cifar10/PGD_2steps_multiNoiseSamples/4-multitrain/eps_127/cifar10/resnet110/noise_0.50/test/sigma_0.50"), "$\sigma = 0.50$ | $\epsilon = 0.50$"),
Line(ApproximateAccuracy("data/certify/cifar10/PGD_2steps_multiNoiseSamples/4-multitrain/eps_255/cifar10/resnet110/noise_0.50/test/sigma_0.50"), "$\sigma = 0.50$ | $\epsilon = 1.00$"),
Line(ApproximateAccuracy("data/certify/cifar10/PGD_2steps_multiNoiseSamples/4-multitrain/eps_512/cifar10/resnet110/noise_0.50/test/sigma_0.50"), "$\sigma = 0.50$ | $\epsilon = 2.00$"),

Line(ApproximateAccuracy("data/certify/cifar10/PGD_2steps_multiNoiseSamples/4-multitrain/eps_64/cifar10/resnet110/noise_1.00/test/sigma_1.00"), "$\sigma = 1.00$ | $\epsilon = 0.25$"),            
Line(ApproximateAccuracy("data/certify/cifar10/PGD_2steps_multiNoiseSamples/4-multitrain/eps_127/cifar10/resnet110/noise_1.00/test/sigma_1.00"), "$\sigma = 1.00$ | $\epsilon = 0.50$"),
Line(ApproximateAccuracy("data/certify/cifar10/PGD_2steps_multiNoiseSamples/4-multitrain/eps_255/cifar10/resnet110/noise_1.00/test/sigma_1.00"), "$\sigma = 1.00$ | $\epsilon = 1.00$"),
Line(ApproximateAccuracy("data/certify/cifar10/PGD_2steps_multiNoiseSamples/4-multitrain/eps_512/cifar10/resnet110/noise_1.00/test/sigma_1.00"), "$\sigma = 1.00$ | $\epsilon = 2.00$"),
]

PGD_2_steps_8_samples = [
Line(ApproximateAccuracy("data/certify/cifar10/PGD_2steps_multiNoiseSamples/8-multitrain/eps_64/cifar10/resnet110/noise_0.12/test/sigma_0.12"), "$\sigma = 0.12$ | $\epsilon = 0.25$"),
Line(ApproximateAccuracy("data/certify/cifar10/PGD_2steps_multiNoiseSamples/8-multitrain/eps_127/cifar10/resnet110/noise_0.12/test/sigma_0.12"), "$\sigma = 0.12$ | $\epsilon = 0.50$"),
Line(ApproximateAccuracy("data/certify/cifar10/PGD_2steps_multiNoiseSamples/8-multitrain/eps_255/cifar10/resnet110/noise_0.12/test/sigma_0.12"), "$\sigma = 0.12$ | $\epsilon = 1.00$"),
Line(ApproximateAccuracy("data/certify/cifar10/PGD_2steps_multiNoiseSamples/8-multitrain/eps_512/cifar10/resnet110/noise_0.12/test/sigma_0.12"), "$\sigma = 0.12$ | $\epsilon = 2.00$"),                       

Line(ApproximateAccuracy("data/certify/cifar10/PGD_2steps_multiNoiseSamples/8-multitrain/eps_64/cifar10/resnet110/noise_0.25/test/sigma_0.25"), "$\sigma = 0.25$ | $\epsilon = 0.25$"),
Line(ApproximateAccuracy("data/certify/cifar10/PGD_2steps_multiNoiseSamples/8-multitrain/eps_127/cifar10/resnet110/noise_0.25/test/sigma_0.25"), "$\sigma = 0.25$ | $\epsilon = 0.50$"),
Line(ApproximateAccuracy("data/certify/cifar10/PGD_2steps_multiNoiseSamples/8-multitrain/eps_255/cifar10/resnet110/noise_0.25/test/sigma_0.25"), "$\sigma = 0.25$ | $\epsilon = 1.00$"),
Line(ApproximateAccuracy("data/certify/cifar10/PGD_2steps_multiNoiseSamples/8-multitrain/eps_512/cifar10/resnet110/noise_0.25/test/sigma_0.25"), "$\sigma = 0.25$ | $\epsilon = 2.00$"),

Line(ApproximateAccuracy("data/certify/cifar10/PGD_2steps_multiNoiseSamples/8-multitrain/eps_64/cifar10/resnet110/noise_0.50/test/sigma_0.50"), "$\sigma = 0.50$ | $\epsilon = 0.25$"),
Line(ApproximateAccuracy("data/certify/cifar10/PGD_2steps_multiNoiseSamples/8-multitrain/eps_127/cifar10/resnet110/noise_0.50/test/sigma_0.50"), "$\sigma = 0.50$ | $\epsilon = 0.50$"),
Line(ApproximateAccuracy("data/certify/cifar10/PGD_2steps_multiNoiseSamples/8-multitrain/eps_255/cifar10/resnet110/noise_0.50/test/sigma_0.50"), "$\sigma = 0.50$ | $\epsilon = 1.00$"),
Line(ApproximateAccuracy("data/certify/cifar10/PGD_2steps_multiNoiseSamples/8-multitrain/eps_512/cifar10/resnet110/noise_0.50/test/sigma_0.50"), "$\sigma = 0.50$ | $\epsilon = 2.00$"),

Line(ApproximateAccuracy("data/certify/cifar10/PGD_2steps_multiNoiseSamples/8-multitrain/eps_64/cifar10/resnet110/noise_1.00/test/sigma_1.00"), "$\sigma = 1.00$ | $\epsilon = 0.25$"),            
Line(ApproximateAccuracy("data/certify/cifar10/PGD_2steps_multiNoiseSamples/8-multitrain/eps_127/cifar10/resnet110/noise_1.00/test/sigma_1.00"), "$\sigma = 1.00$ | $\epsilon = 0.50$"),
Line(ApproximateAccuracy("data/certify/cifar10/PGD_2steps_multiNoiseSamples/8-multitrain/eps_255/cifar10/resnet110/noise_1.00/test/sigma_1.00"), "$\sigma = 1.00$ | $\epsilon = 1.00$"),
Line(ApproximateAccuracy("data/certify/cifar10/PGD_2steps_multiNoiseSamples/8-multitrain/eps_512/cifar10/resnet110/noise_1.00/test/sigma_1.00"), "$\sigma = 1.00$ | $\epsilon = 2.00$"),
]


PGD_4_steps_2_samples = [
Line(ApproximateAccuracy("data/certify/cifar10/PGD_4steps_multiNoiseSamples/2-multitrain/eps_64/cifar10/resnet110/noise_0.12/test/sigma_0.12"), "$\sigma = 0.12$ | $\epsilon = 0.25$"),
Line(ApproximateAccuracy("data/certify/cifar10/PGD_4steps_multiNoiseSamples/2-multitrain/eps_127/cifar10/resnet110/noise_0.12/test/sigma_0.12"), "$\sigma = 0.12$ | $\epsilon = 0.50$"),
Line(ApproximateAccuracy("data/certify/cifar10/PGD_4steps_multiNoiseSamples/2-multitrain/eps_255/cifar10/resnet110/noise_0.12/test/sigma_0.12"), "$\sigma = 0.12$ | $\epsilon = 1.00$"),
Line(ApproximateAccuracy("data/certify/cifar10/PGD_4steps_multiNoiseSamples/2-multitrain/eps_512/cifar10/resnet110/noise_0.12/test/sigma_0.12"), "$\sigma = 0.12$ | $\epsilon = 2.00$"),            

Line(ApproximateAccuracy("data/certify/cifar10/PGD_4steps_multiNoiseSamples/2-multitrain/eps_64/cifar10/resnet110/noise_0.25/test/sigma_0.25"), "$\sigma = 0.25$ | $\epsilon = 0.25$"),
Line(ApproximateAccuracy("data/certify/cifar10/PGD_4steps_multiNoiseSamples/2-multitrain/eps_127/cifar10/resnet110/noise_0.25/test/sigma_0.25"), "$\sigma = 0.25$ | $\epsilon = 0.50$"),
Line(ApproximateAccuracy("data/certify/cifar10/PGD_4steps_multiNoiseSamples/2-multitrain/eps_255/cifar10/resnet110/noise_0.25/test/sigma_0.25"), "$\sigma = 0.25$ | $\epsilon = 1.00$"),
Line(ApproximateAccuracy("data/certify/cifar10/PGD_4steps_multiNoiseSamples/2-multitrain/eps_512/cifar10/resnet110/noise_0.25/test/sigma_0.25"), "$\sigma = 0.25$ | $\epsilon = 2.00$"),

Line(ApproximateAccuracy("data/certify/cifar10/PGD_4steps_multiNoiseSamples/2-multitrain/eps_64/cifar10/resnet110/noise_0.50/test/sigma_0.50"), "$\sigma = 0.50$ | $\epsilon = 0.25$"),
Line(ApproximateAccuracy("data/certify/cifar10/PGD_4steps_multiNoiseSamples/2-multitrain/eps_127/cifar10/resnet110/noise_0.50/test/sigma_0.50"), "$\sigma = 0.50$ | $\epsilon = 0.50$"),
Line(ApproximateAccuracy("data/certify/cifar10/PGD_4steps_multiNoiseSamples/2-multitrain/eps_255/cifar10/resnet110/noise_0.50/test/sigma_0.50"), "$\sigma = 0.50$ | $\epsilon = 1.00$"),
Line(ApproximateAccuracy("data/certify/cifar10/PGD_4steps_multiNoiseSamples/2-multitrain/eps_512/cifar10/resnet110/noise_0.50/test/sigma_0.50"), "$\sigma = 0.50$ | $\epsilon = 2.00$"),

Line(ApproximateAccuracy("data/certify/cifar10/PGD_4steps_multiNoiseSamples/2-multitrain/eps_64/cifar10/resnet110/noise_1.00/test/sigma_1.00"), "$\sigma = 1.00$ | $\epsilon = 0.25$"),            
Line(ApproximateAccuracy("data/certify/cifar10/PGD_4steps_multiNoiseSamples/2-multitrain/eps_127/cifar10/resnet110/noise_1.00/test/sigma_1.00"), "$\sigma = 1.00$ | $\epsilon = 0.50$"),
Line(ApproximateAccuracy("data/certify/cifar10/PGD_4steps_multiNoiseSamples/2-multitrain/eps_255/cifar10/resnet110/noise_1.00/test/sigma_1.00"), "$\sigma = 1.00$ | $\epsilon = 1.00$"),
Line(ApproximateAccuracy("data/certify/cifar10/PGD_4steps_multiNoiseSamples/2-multitrain/eps_512/cifar10/resnet110/noise_1.00/test/sigma_1.00"), "$\sigma = 1.00$ | $\epsilon = 2.00$"),
]

PGD_4_steps_4_samples = [
Line(ApproximateAccuracy("data/certify/cifar10/PGD_4steps_multiNoiseSamples/4-multitrain/eps_64/cifar10/resnet110/noise_0.12/test/sigma_0.12"), "$\sigma = 0.12$ | $\epsilon = 0.25$"),
Line(ApproximateAccuracy("data/certify/cifar10/PGD_4steps_multiNoiseSamples/4-multitrain/eps_127/cifar10/resnet110/noise_0.12/test/sigma_0.12"), "$\sigma = 0.12$ | $\epsilon = 0.50$"),
Line(ApproximateAccuracy("data/certify/cifar10/PGD_4steps_multiNoiseSamples/4-multitrain/eps_255/cifar10/resnet110/noise_0.12/test/sigma_0.12"), "$\sigma = 0.12$ | $\epsilon = 1.00$"),
Line(ApproximateAccuracy("data/certify/cifar10/PGD_4steps_multiNoiseSamples/4-multitrain/eps_512/cifar10/resnet110/noise_0.12/test/sigma_0.12"), "$\sigma = 0.12$ | $\epsilon = 2.00$"),            

Line(ApproximateAccuracy("data/certify/cifar10/PGD_4steps_multiNoiseSamples/4-multitrain/eps_64/cifar10/resnet110/noise_0.25/test/sigma_0.25"), "$\sigma = 0.25$ | $\epsilon = 0.25$"),
Line(ApproximateAccuracy("data/certify/cifar10/PGD_4steps_multiNoiseSamples/4-multitrain/eps_127/cifar10/resnet110/noise_0.25/test/sigma_0.25"), "$\sigma = 0.25$ | $\epsilon = 0.50$"),
Line(ApproximateAccuracy("data/certify/cifar10/PGD_4steps_multiNoiseSamples/4-multitrain/eps_255/cifar10/resnet110/noise_0.25/test/sigma_0.25"), "$\sigma = 0.25$ | $\epsilon = 1.00$"),
Line(ApproximateAccuracy("data/certify/cifar10/PGD_4steps_multiNoiseSamples/4-multitrain/eps_512/cifar10/resnet110/noise_0.25/test/sigma_0.25"), "$\sigma = 0.25$ | $\epsilon = 2.00$"),

Line(ApproximateAccuracy("data/certify/cifar10/PGD_4steps_multiNoiseSamples/4-multitrain/eps_64/cifar10/resnet110/noise_0.50/test/sigma_0.50"), "$\sigma = 0.50$ | $\epsilon = 0.25$"),
Line(ApproximateAccuracy("data/certify/cifar10/PGD_4steps_multiNoiseSamples/4-multitrain/eps_127/cifar10/resnet110/noise_0.50/test/sigma_0.50"), "$\sigma = 0.50$ | $\epsilon = 0.50$"),
Line(ApproximateAccuracy("data/certify/cifar10/PGD_4steps_multiNoiseSamples/4-multitrain/eps_255/cifar10/resnet110/noise_0.50/test/sigma_0.50"), "$\sigma = 0.50$ | $\epsilon = 1.00$"),
Line(ApproximateAccuracy("data/certify/cifar10/PGD_4steps_multiNoiseSamples/4-multitrain/eps_512/cifar10/resnet110/noise_0.50/test/sigma_0.50"), "$\sigma = 0.50$ | $\epsilon = 2.00$"),

Line(ApproximateAccuracy("data/certify/cifar10/PGD_4steps_multiNoiseSamples/4-multitrain/eps_64/cifar10/resnet110/noise_1.00/test/sigma_1.00"), "$\sigma = 1.00$ | $\epsilon = 0.25$"),            
Line(ApproximateAccuracy("data/certify/cifar10/PGD_4steps_multiNoiseSamples/4-multitrain/eps_127/cifar10/resnet110/noise_1.00/test/sigma_1.00"), "$\sigma = 1.00$ | $\epsilon = 0.50$"),
Line(ApproximateAccuracy("data/certify/cifar10/PGD_4steps_multiNoiseSamples/4-multitrain/eps_255/cifar10/resnet110/noise_1.00/test/sigma_1.00"), "$\sigma = 1.00$ | $\epsilon = 1.00$"),
Line(ApproximateAccuracy("data/certify/cifar10/PGD_4steps_multiNoiseSamples/4-multitrain/eps_512/cifar10/resnet110/noise_1.00/test/sigma_1.00"), "$\sigma = 1.00$ | $\epsilon = 2.00$"),
]

PGD_4_steps_8_samples = [
Line(ApproximateAccuracy("data/certify/cifar10/PGD_4steps_multiNoiseSamples/8-multitrain/eps_64/cifar10/resnet110/noise_0.12/test/sigma_0.12"), "$\sigma = 0.12$ | $\epsilon = 0.25$"),
Line(ApproximateAccuracy("data/certify/cifar10/PGD_4steps_multiNoiseSamples/8-multitrain/eps_127/cifar10/resnet110/noise_0.12/test/sigma_0.12"), "$\sigma = 0.12$ | $\epsilon = 0.50$"),
Line(ApproximateAccuracy("data/certify/cifar10/PGD_4steps_multiNoiseSamples/8-multitrain/eps_255/cifar10/resnet110/noise_0.12/test/sigma_0.12"), "$\sigma = 0.12$ | $\epsilon = 1.00$"),
Line(ApproximateAccuracy("data/certify/cifar10/PGD_4steps_multiNoiseSamples/8-multitrain/eps_512/cifar10/resnet110/noise_0.12/test/sigma_0.12"), "$\sigma = 0.12$ | $\epsilon = 2.00$"),                       

Line(ApproximateAccuracy("data/certify/cifar10/PGD_4steps_multiNoiseSamples/8-multitrain/eps_64/cifar10/resnet110/noise_0.25/test/sigma_0.25"), "$\sigma = 0.25$ | $\epsilon = 0.25$"),
Line(ApproximateAccuracy("data/certify/cifar10/PGD_4steps_multiNoiseSamples/8-multitrain/eps_127/cifar10/resnet110/noise_0.25/test/sigma_0.25"), "$\sigma = 0.25$ | $\epsilon = 0.50$"),
Line(ApproximateAccuracy("data/certify/cifar10/PGD_4steps_multiNoiseSamples/8-multitrain/eps_255/cifar10/resnet110/noise_0.25/test/sigma_0.25"), "$\sigma = 0.25$ | $\epsilon = 1.00$"),
Line(ApproximateAccuracy("data/certify/cifar10/PGD_4steps_multiNoiseSamples/8-multitrain/eps_512/cifar10/resnet110/noise_0.25/test/sigma_0.25"), "$\sigma = 0.25$ | $\epsilon = 2.00$"),

Line(ApproximateAccuracy("data/certify/cifar10/PGD_4steps_multiNoiseSamples/8-multitrain/eps_64/cifar10/resnet110/noise_0.50/test/sigma_0.50"), "$\sigma = 0.50$ | $\epsilon = 0.25$"),
Line(ApproximateAccuracy("data/certify/cifar10/PGD_4steps_multiNoiseSamples/8-multitrain/eps_127/cifar10/resnet110/noise_0.50/test/sigma_0.50"), "$\sigma = 0.50$ | $\epsilon = 0.50$"),
Line(ApproximateAccuracy("data/certify/cifar10/PGD_4steps_multiNoiseSamples/8-multitrain/eps_255/cifar10/resnet110/noise_0.50/test/sigma_0.50"), "$\sigma = 0.50$ | $\epsilon = 1.00$"),
Line(ApproximateAccuracy("data/certify/cifar10/PGD_4steps_multiNoiseSamples/8-multitrain/eps_512/cifar10/resnet110/noise_0.50/test/sigma_0.50"), "$\sigma = 0.50$ | $\epsilon = 2.00$"),

Line(ApproximateAccuracy("data/certify/cifar10/PGD_4steps_multiNoiseSamples/8-multitrain/eps_64/cifar10/resnet110/noise_1.00/test/sigma_1.00"), "$\sigma = 1.00$ | $\epsilon = 0.25$"),            
Line(ApproximateAccuracy("data/certify/cifar10/PGD_4steps_multiNoiseSamples/8-multitrain/eps_127/cifar10/resnet110/noise_1.00/test/sigma_1.00"), "$\sigma = 1.00$ | $\epsilon = 0.50$"),
Line(ApproximateAccuracy("data/certify/cifar10/PGD_4steps_multiNoiseSamples/8-multitrain/eps_255/cifar10/resnet110/noise_1.00/test/sigma_1.00"), "$\sigma = 1.00$ | $\epsilon = 1.00$"),
Line(ApproximateAccuracy("data/certify/cifar10/PGD_4steps_multiNoiseSamples/8-multitrain/eps_512/cifar10/resnet110/noise_1.00/test/sigma_1.00"), "$\sigma = 1.00$ | $\epsilon = 2.00$"),
]

PGD_10_steps_2_samples = [
Line(ApproximateAccuracy("data/certify/cifar10/PGD_10steps_multiNoiseSamples/2-multitrain/eps_64/cifar10/resnet110/noise_0.12/test/sigma_0.12"), "$\sigma = 0.12$ | $\epsilon = 0.25$"),
Line(ApproximateAccuracy("data/certify/cifar10/PGD_10steps_multiNoiseSamples/2-multitrain/eps_127/cifar10/resnet110/noise_0.12/test/sigma_0.12"), "$\sigma = 0.12$ | $\epsilon = 0.50$"),
Line(ApproximateAccuracy("data/certify/cifar10/PGD_10steps_multiNoiseSamples/2-multitrain/eps_255/cifar10/resnet110/noise_0.12/test/sigma_0.12"), "$\sigma = 0.12$ | $\epsilon = 1.00$"),
Line(ApproximateAccuracy("data/certify/cifar10/PGD_10steps_multiNoiseSamples/2-multitrain/eps_512/cifar10/resnet110/noise_0.12/test/sigma_0.12"), "$\sigma = 0.12$ | $\epsilon = 2.00$"),            

Line(ApproximateAccuracy("data/certify/cifar10/PGD_10steps_multiNoiseSamples/2-multitrain/eps_64/cifar10/resnet110/noise_0.25/test/sigma_0.25"), "$\sigma = 0.25$ | $\epsilon = 0.25$"),
Line(ApproximateAccuracy("data/certify/cifar10/PGD_10steps_multiNoiseSamples/2-multitrain/eps_127/cifar10/resnet110/noise_0.25/test/sigma_0.25"), "$\sigma = 0.25$ | $\epsilon = 0.50$"),
Line(ApproximateAccuracy("data/certify/cifar10/PGD_10steps_multiNoiseSamples/2-multitrain/eps_255/cifar10/resnet110/noise_0.25/test/sigma_0.25"), "$\sigma = 0.25$ | $\epsilon = 1.00$"),
Line(ApproximateAccuracy("data/certify/cifar10/PGD_10steps_multiNoiseSamples/2-multitrain/eps_512/cifar10/resnet110/noise_0.25/test/sigma_0.25"), "$\sigma = 0.25$ | $\epsilon = 2.00$"),

Line(ApproximateAccuracy("data/certify/cifar10/PGD_10steps_multiNoiseSamples/2-multitrain/eps_64/cifar10/resnet110/noise_0.50/test/sigma_0.50"), "$\sigma = 0.50$ | $\epsilon = 0.25$"),
Line(ApproximateAccuracy("data/certify/cifar10/PGD_10steps_multiNoiseSamples/2-multitrain/eps_127/cifar10/resnet110/noise_0.50/test/sigma_0.50"), "$\sigma = 0.50$ | $\epsilon = 0.50$"),
Line(ApproximateAccuracy("data/certify/cifar10/PGD_10steps_multiNoiseSamples/2-multitrain/eps_255/cifar10/resnet110/noise_0.50/test/sigma_0.50"), "$\sigma = 0.50$ | $\epsilon = 1.00$"),
Line(ApproximateAccuracy("data/certify/cifar10/PGD_10steps_multiNoiseSamples/2-multitrain/eps_512/cifar10/resnet110/noise_0.50/test/sigma_0.50"), "$\sigma = 0.50$ | $\epsilon = 2.00$"),

Line(ApproximateAccuracy("data/certify/cifar10/PGD_10steps_multiNoiseSamples/2-multitrain/eps_64/cifar10/resnet110/noise_1.00/test/sigma_1.00"), "$\sigma = 1.00$ | $\epsilon = 0.25$"),            
Line(ApproximateAccuracy("data/certify/cifar10/PGD_10steps_multiNoiseSamples/2-multitrain/eps_127/cifar10/resnet110/noise_1.00/test/sigma_1.00"), "$\sigma = 1.00$ | $\epsilon = 0.50$"),
Line(ApproximateAccuracy("data/certify/cifar10/PGD_10steps_multiNoiseSamples/2-multitrain/eps_255/cifar10/resnet110/noise_1.00/test/sigma_1.00"), "$\sigma = 1.00$ | $\epsilon = 1.00$"),
Line(ApproximateAccuracy("data/certify/cifar10/PGD_10steps_multiNoiseSamples/2-multitrain/eps_512/cifar10/resnet110/noise_1.00/test/sigma_1.00"), "$\sigma = 1.00$ | $\epsilon = 2.00$"),
]


PGD_10_steps_4_samples = [
Line(ApproximateAccuracy("data/certify/cifar10/PGD_10steps_multiNoiseSamples/4-multitrain/eps_64/cifar10/resnet110/noise_0.12/test/sigma_0.12"), "$\sigma = 0.12$ | $\epsilon = 0.25$"),
Line(ApproximateAccuracy("data/certify/cifar10/PGD_10steps_multiNoiseSamples/4-multitrain/eps_127/cifar10/resnet110/noise_0.12/test/sigma_0.12"), "$\sigma = 0.12$ | $\epsilon = 0.50$"),
Line(ApproximateAccuracy("data/certify/cifar10/PGD_10steps_multiNoiseSamples/4-multitrain/eps_255/cifar10/resnet110/noise_0.12/test/sigma_0.12"), "$\sigma = 0.12$ | $\epsilon = 1.00$"),
Line(ApproximateAccuracy("data/certify/cifar10/PGD_10steps_multiNoiseSamples/4-multitrain/eps_512/cifar10/resnet110/noise_0.12/test/sigma_0.12"), "$\sigma = 0.12$ | $\epsilon = 2.00$"),            

Line(ApproximateAccuracy("data/certify/cifar10/PGD_10steps_multiNoiseSamples/4-multitrain/eps_64/cifar10/resnet110/noise_0.25/test/sigma_0.25"), "$\sigma = 0.25$ | $\epsilon = 0.25$"),
Line(ApproximateAccuracy("data/certify/cifar10/PGD_10steps_multiNoiseSamples/4-multitrain/eps_127/cifar10/resnet110/noise_0.25/test/sigma_0.25"), "$\sigma = 0.25$ | $\epsilon = 0.50$"),
Line(ApproximateAccuracy("data/certify/cifar10/PGD_10steps_multiNoiseSamples/4-multitrain/eps_255/cifar10/resnet110/noise_0.25/test/sigma_0.25"), "$\sigma = 0.25$ | $\epsilon = 1.00$"),
Line(ApproximateAccuracy("data/certify/cifar10/PGD_10steps_multiNoiseSamples/4-multitrain/eps_512/cifar10/resnet110/noise_0.25/test/sigma_0.25"), "$\sigma = 0.25$ | $\epsilon = 2.00$"),

Line(ApproximateAccuracy("data/certify/cifar10/PGD_10steps_multiNoiseSamples/4-multitrain/eps_64/cifar10/resnet110/noise_0.50/test/sigma_0.50"), "$\sigma = 0.50$ | $\epsilon = 0.25$"),
Line(ApproximateAccuracy("data/certify/cifar10/PGD_10steps_multiNoiseSamples/4-multitrain/eps_127/cifar10/resnet110/noise_0.50/test/sigma_0.50"), "$\sigma = 0.50$ | $\epsilon = 0.50$"),
Line(ApproximateAccuracy("data/certify/cifar10/PGD_10steps_multiNoiseSamples/4-multitrain/eps_255/cifar10/resnet110/noise_0.50/test/sigma_0.50"), "$\sigma = 0.50$ | $\epsilon = 1.00$"),
Line(ApproximateAccuracy("data/certify/cifar10/PGD_10steps_multiNoiseSamples/4-multitrain/eps_512/cifar10/resnet110/noise_0.50/test/sigma_0.50"), "$\sigma = 0.50$ | $\epsilon = 2.00$"),

Line(ApproximateAccuracy("data/certify/cifar10/PGD_10steps_multiNoiseSamples/4-multitrain/eps_64/cifar10/resnet110/noise_1.00/test/sigma_1.00"), "$\sigma = 1.00$ | $\epsilon = 0.25$"),            
Line(ApproximateAccuracy("data/certify/cifar10/PGD_10steps_multiNoiseSamples/4-multitrain/eps_127/cifar10/resnet110/noise_1.00/test/sigma_1.00"), "$\sigma = 1.00$ | $\epsilon = 0.50$"),
Line(ApproximateAccuracy("data/certify/cifar10/PGD_10steps_multiNoiseSamples/4-multitrain/eps_255/cifar10/resnet110/noise_1.00/test/sigma_1.00"), "$\sigma = 1.00$ | $\epsilon = 1.00$"),
Line(ApproximateAccuracy("data/certify/cifar10/PGD_10steps_multiNoiseSamples/4-multitrain/eps_512/cifar10/resnet110/noise_1.00/test/sigma_1.00"), "$\sigma = 1.00$ | $\epsilon = 2.00$"),
]

PGD_10_steps_8_samples = [
Line(ApproximateAccuracy("data/certify/cifar10/PGD_10steps_multiNoiseSamples/8-multitrain/eps_64/cifar10/resnet110/noise_0.12/test/sigma_0.12"), "$\sigma = 0.12$ | $\epsilon = 0.25$"),
Line(ApproximateAccuracy("data/certify/cifar10/PGD_10steps_multiNoiseSamples/8-multitrain/eps_127/cifar10/resnet110/noise_0.12/test/sigma_0.12"), "$\sigma = 0.12$ | $\epsilon = 0.50$"),
Line(ApproximateAccuracy("data/certify/cifar10/PGD_10steps_multiNoiseSamples/8-multitrain/eps_255/cifar10/resnet110/noise_0.12/test/sigma_0.12"), "$\sigma = 0.12$ | $\epsilon = 1.00$"),
Line(ApproximateAccuracy("data/certify/cifar10/PGD_10steps_multiNoiseSamples/8-multitrain/eps_512/cifar10/resnet110/noise_0.12/test/sigma_0.12"), "$\sigma = 0.12$ | $\epsilon = 2.00$"),                       

Line(ApproximateAccuracy("data/certify/cifar10/PGD_10steps_multiNoiseSamples/8-multitrain/eps_64/cifar10/resnet110/noise_0.25/test/sigma_0.25"), "$\sigma = 0.25$ | $\epsilon = 0.25$"),
Line(ApproximateAccuracy("data/certify/cifar10/PGD_10steps_multiNoiseSamples/8-multitrain/eps_127/cifar10/resnet110/noise_0.25/test/sigma_0.25"), "$\sigma = 0.25$ | $\epsilon = 0.50$"),
Line(ApproximateAccuracy("data/certify/cifar10/PGD_10steps_multiNoiseSamples/8-multitrain/eps_255/cifar10/resnet110/noise_0.25/test/sigma_0.25"), "$\sigma = 0.25$ | $\epsilon = 1.00$"),
Line(ApproximateAccuracy("data/certify/cifar10/PGD_10steps_multiNoiseSamples/8-multitrain/eps_512/cifar10/resnet110/noise_0.25/test/sigma_0.25"), "$\sigma = 0.25$ | $\epsilon = 2.00$"),

Line(ApproximateAccuracy("data/certify/cifar10/PGD_10steps_multiNoiseSamples/8-multitrain/eps_64/cifar10/resnet110/noise_0.50/test/sigma_0.50"), "$\sigma = 0.50$ | $\epsilon = 0.25$"),
Line(ApproximateAccuracy("data/certify/cifar10/PGD_10steps_multiNoiseSamples/8-multitrain/eps_127/cifar10/resnet110/noise_0.50/test/sigma_0.50"), "$\sigma = 0.50$ | $\epsilon = 0.50$"),
Line(ApproximateAccuracy("data/certify/cifar10/PGD_10steps_multiNoiseSamples/8-multitrain/eps_255/cifar10/resnet110/noise_0.50/test/sigma_0.50"), "$\sigma = 0.50$ | $\epsilon = 1.00$"),
Line(ApproximateAccuracy("data/certify/cifar10/PGD_10steps_multiNoiseSamples/8-multitrain/eps_512/cifar10/resnet110/noise_0.50/test/sigma_0.50"), "$\sigma = 0.50$ | $\epsilon = 2.00$"),

Line(ApproximateAccuracy("data/certify/cifar10/PGD_10steps_multiNoiseSamples/8-multitrain/eps_64/cifar10/resnet110/noise_1.00/test/sigma_1.00"), "$\sigma = 1.00$ | $\epsilon = 0.25$"),            
Line(ApproximateAccuracy("data/certify/cifar10/PGD_10steps_multiNoiseSamples/8-multitrain/eps_127/cifar10/resnet110/noise_1.00/test/sigma_1.00"), "$\sigma = 1.00$ | $\epsilon = 0.50$"),
Line(ApproximateAccuracy("data/certify/cifar10/PGD_10steps_multiNoiseSamples/8-multitrain/eps_255/cifar10/resnet110/noise_1.00/test/sigma_1.00"), "$\sigma = 1.00$ | $\epsilon = 1.00$"),
Line(ApproximateAccuracy("data/certify/cifar10/PGD_10steps_multiNoiseSamples/8-multitrain/eps_512/cifar10/resnet110/noise_1.00/test/sigma_1.00"), "$\sigma = 1.00$ | $\epsilon = 2.00$"),
]


all_imagenet_experiments = [
Line(ApproximateAccuracy("data/certify/imagenet/PGD_1step/imagenet/eps_127/resnet50/noise_0.25/test/sigma_0.25"), "$\sigma = 0.25$ | $\epsilon = 0.5$"),
Line(ApproximateAccuracy("data/certify/imagenet/PGD_1step/imagenet/eps_255/resnet50/noise_0.25/test/sigma_0.25"), "$\sigma = 0.25$ | $\epsilon = 1.0$"),
Line(ApproximateAccuracy("data/certify/imagenet/PGD_1step/imagenet/eps_512/resnet50/noise_0.25/test/sigma_0.25"), "$\sigma = 0.25$ | $\epsilon = 2.0$"),
Line(ApproximateAccuracy("data/certify/imagenet/PGD_1step/imagenet/eps_1024/resnet50/noise_0.25/test/sigma_0.25"), "$\sigma = 0.25$ | $\epsilon = 4.0$"),
Line(ApproximateAccuracy("data/certify/imagenet/PGD_1step/imagenet/eps_2048/resnet50/noise_0.25/test/sigma_0.25"), "$\sigma = 0.25$ | $\epsilon = 8.0$"),

Line(ApproximateAccuracy("data/certify/imagenet/PGD_1step/imagenet/eps_127/resnet50/noise_0.50/test/sigma_0.50"), "$\sigma = 0.50$ | $\epsilon = 0.5$"),
Line(ApproximateAccuracy("data/certify/imagenet/PGD_1step/imagenet/eps_255/resnet50/noise_0.50/test/sigma_0.50"), "$\sigma = 0.50$ | $\epsilon = 1.0$"),
Line(ApproximateAccuracy("data/certify/imagenet/PGD_1step/imagenet/eps_512/resnet50/noise_0.50/test/sigma_0.50"), "$\sigma = 0.50$ | $\epsilon = 2.0$"),
Line(ApproximateAccuracy("data/certify/imagenet/PGD_1step/imagenet/eps_1024/resnet50/noise_0.50/test/sigma_0.50"), "$\sigma = 0.50$ | $\epsilon = 4.0$"),
Line(ApproximateAccuracy("data/certify/imagenet/PGD_1step/imagenet/eps_2048/resnet50/noise_0.50/test/sigma_0.50"), "$\sigma = 0.50$ | $\epsilon = 8.0$"),

Line(ApproximateAccuracy("data/certify/imagenet/PGD_1step/imagenet/eps_127/resnet50/noise_1.00/test/sigma_1.00"), "$\sigma = 1.00$ | $\epsilon = 0.5$"),
Line(ApproximateAccuracy("data/certify/imagenet/PGD_1step/imagenet/eps_255/resnet50/noise_1.00/test/sigma_1.00"), "$\sigma = 1.00$ | $\epsilon = 1.0$"),
Line(ApproximateAccuracy("data/certify/imagenet/PGD_1step/imagenet/eps_512/resnet50/noise_1.00/test/sigma_1.00"), "$\sigma = 1.00$ | $\epsilon = 2.0$"),
Line(ApproximateAccuracy("data/certify/imagenet/PGD_1step/imagenet/eps_1024/resnet50/noise_1.00/test/sigma_1.00"), "$\sigma = 1.00$ | $\epsilon = 4.0$"),
Line(ApproximateAccuracy("data/certify/imagenet/PGD_1step/imagenet/eps_2048/resnet50/noise_1.00/test/sigma_1.00"), "$\sigma = 1.00$ | $\epsilon = 8.0$"),

Line(ApproximateAccuracy("data/certify/imagenet/DDN_2steps/eps_127/resnet50/noise_0.25/test/sigma_0.25"), "$\sigma = 0.25$ | $\epsilon = 0.5$"),
Line(ApproximateAccuracy("data/certify/imagenet/DDN_2steps/eps_255/resnet50/noise_0.25/test/sigma_0.25"), "$\sigma = 0.25$ | $\epsilon = 1.0$"),
Line(ApproximateAccuracy("data/certify/imagenet/DDN_2steps/eps_512/resnet50/noise_0.25/test/sigma_0.25"), "$\sigma = 0.25$ | $\epsilon = 2.0$"),
Line(ApproximateAccuracy("data/certify/imagenet/DDN_2steps/eps_1024/resnet50/noise_0.25/test/sigma_0.25"), "$\sigma = 0.25$ | $\epsilon = 4.0$"),

Line(ApproximateAccuracy("data/certify/imagenet/DDN_2steps/eps_127/resnet50/noise_0.50/test/sigma_0.50"), "$\sigma = 0.50$ | $\epsilon = 0.5$"),
Line(ApproximateAccuracy("data/certify/imagenet/DDN_2steps/eps_255/resnet50/noise_0.50/test/sigma_0.50"), "$\sigma = 0.50$ | $\epsilon = 1.0$"),
Line(ApproximateAccuracy("data/certify/imagenet/DDN_2steps/eps_512/resnet50/noise_0.50/test/sigma_0.50"), "$\sigma = 0.50$ | $\epsilon = 2.0$"),
Line(ApproximateAccuracy("data/certify/imagenet/DDN_2steps/eps_1024/resnet50/noise_0.50/test/sigma_0.50"), "$\sigma = 0.50$ | $\epsilon = 4.0$"),

Line(ApproximateAccuracy("data/certify/imagenet/DDN_2steps/eps_127/resnet50/noise_1.00/test/sigma_1.00"), "$\sigma = 1.00$ | $\epsilon = 0.5$"),
Line(ApproximateAccuracy("data/certify/imagenet/DDN_2steps/eps_255/resnet50/noise_1.00/test/sigma_1.00"), "$\sigma = 1.00$ | $\epsilon = 1.0$"),
Line(ApproximateAccuracy("data/certify/imagenet/DDN_2steps/eps_512/resnet50/noise_1.00/test/sigma_1.00"), "$\sigma = 1.00$ | $\epsilon = 2.0$"),
Line(ApproximateAccuracy("data/certify/imagenet/DDN_2steps/eps_1024/resnet50/noise_1.00/test/sigma_1.00"), "$\sigma = 1.00$ | $\epsilon = 4.0$"),
]


PGD_self_training = np.hstack([
[
Line(ApproximateAccuracy("data/certify/cifar10/self_training/PGD_{}steps/weight_{:.1f}/eps_{}/cifar10/resnet110/noise_0.12/test/sigma_0.12".format(num_pgd, weight, eps)), "$\sigma = 0.12$", plot_fmt='b'),
Line(ApproximateAccuracy("data/certify/cifar10/self_training/PGD_{}steps/weight_{:.1f}/eps_{}/cifar10/resnet110/noise_0.25/test/sigma_0.25".format(num_pgd, weight, eps)), "$\sigma = 0.25$", plot_fmt='orange'),
Line(ApproximateAccuracy("data/certify/cifar10/self_training/PGD_{}steps/weight_{:.1f}/eps_{}/cifar10/resnet110/noise_0.50/test/sigma_0.50".format(num_pgd, weight, eps)), "$\sigma = 0.50$", plot_fmt='g'),
Line(ApproximateAccuracy("data/certify/cifar10/self_training/PGD_{}steps/weight_{:.1f}/eps_{}/cifar10/resnet110/noise_1.00/test/sigma_1.00".format(num_pgd, weight, eps)), "$\sigma = 1.00$", plot_fmt='r'),

Line(ApproximateAccuracy("data/certify/cifar10/self_training/PGD_{}steps/weight_{:.1f}/eps_{}/cifar10/resnet110/noise_0.12/test/sigma_0.12".format(num_pgd, weight, eps)), "$\sigma = 0.12$", plot_fmt='b'),
Line(ApproximateAccuracy("data/certify/cifar10/self_training/PGD_{}steps/weight_{:.1f}/eps_{}/cifar10/resnet110/noise_0.25/test/sigma_0.25".format(num_pgd, weight, eps)), "$\sigma = 0.25$", plot_fmt='orange'),
Line(ApproximateAccuracy("data/certify/cifar10/self_training/PGD_{}steps/weight_{:.1f}/eps_{}/cifar10/resnet110/noise_0.50/test/sigma_0.50".format(num_pgd, weight, eps)), "$\sigma = 0.50$", plot_fmt='g'),
Line(ApproximateAccuracy("data/certify/cifar10/self_training/PGD_{}steps/weight_{:.1f}/eps_{}/cifar10/resnet110/noise_1.00/test/sigma_1.00".format(num_pgd, weight, eps)), "$\sigma = 1.00$", plot_fmt='r'),

Line(ApproximateAccuracy("data/certify/cifar10/self_training/PGD_{}steps/weight_{:.1f}/eps_{}/cifar10/resnet110/noise_0.12/test/sigma_0.12".format(num_pgd, weight, eps)), "$\sigma = 0.12$", plot_fmt='b'),
Line(ApproximateAccuracy("data/certify/cifar10/self_training/PGD_{}steps/weight_{:.1f}/eps_{}/cifar10/resnet110/noise_0.25/test/sigma_0.25".format(num_pgd, weight, eps)), "$\sigma = 0.25$", plot_fmt='orange'),
Line(ApproximateAccuracy("data/certify/cifar10/self_training/PGD_{}steps/weight_{:.1f}/eps_{}/cifar10/resnet110/noise_0.50/test/sigma_0.50".format(num_pgd, weight, eps)), "$\sigma = 0.50$", plot_fmt='g'),
Line(ApproximateAccuracy("data/certify/cifar10/self_training/PGD_{}steps/weight_{:.1f}/eps_{}/cifar10/resnet110/noise_1.00/test/sigma_1.00".format(num_pgd, weight, eps)), "$\sigma = 1.00$", plot_fmt='r'),

Line(ApproximateAccuracy("data/certify/cifar10/self_training/PGD_{}steps/weight_{:.1f}/eps_{}/cifar10/resnet110/noise_0.12/test/sigma_0.12".format(num_pgd, weight, eps)), "$\sigma = 0.12$", plot_fmt='b'),
Line(ApproximateAccuracy("data/certify/cifar10/self_training/PGD_{}steps/weight_{:.1f}/eps_{}/cifar10/resnet110/noise_0.25/test/sigma_0.25".format(num_pgd, weight, eps)), "$\sigma = 0.25$", plot_fmt='orange'),
Line(ApproximateAccuracy("data/certify/cifar10/self_training/PGD_{}steps/weight_{:.1f}/eps_{}/cifar10/resnet110/noise_0.50/test/sigma_0.50".format(num_pgd, weight, eps)), "$\sigma = 0.50$", plot_fmt='g'),
Line(ApproximateAccuracy("data/certify/cifar10/self_training/PGD_{}steps/weight_{:.1f}/eps_{}/cifar10/resnet110/noise_1.00/test/sigma_1.00".format(num_pgd, weight, eps)), "$\sigma = 1.00$", plot_fmt='r'),
]for num_pgd in [2, 4, 6, 8, 10] for weight in [0.1, 0.5, 1.0]for eps in [64, 127, 255, 512]
]).tolist()


PGD_imagenetPretraining_cifar10Fineuning_self_training = np.hstack([
[
Line(ApproximateAccuracy("data/certify/cifar10/finetune_cifar_from_imagenetPGD2steps/self_training_PGD_{}steps/weight_{:.1f}/eps_{}/cifar10/resnet110/noise_0.12/test/sigma_0.12".format(num_pgd, weight, eps)), "$\sigma = 0.12$", plot_fmt='b'),
Line(ApproximateAccuracy("data/certify/cifar10/finetune_cifar_from_imagenetPGD2steps/self_training_PGD_{}steps/weight_{:.1f}/eps_{}/cifar10/resnet110/noise_0.25/test/sigma_0.25".format(num_pgd, weight, eps)), "$\sigma = 0.25$", plot_fmt='orange'),
Line(ApproximateAccuracy("data/certify/cifar10/finetune_cifar_from_imagenetPGD2steps/self_training_PGD_{}steps/weight_{:.1f}/eps_{}/cifar10/resnet110/noise_0.50/test/sigma_0.50".format(num_pgd, weight, eps)), "$\sigma = 0.50$", plot_fmt='g'),
Line(ApproximateAccuracy("data/certify/cifar10/finetune_cifar_from_imagenetPGD2steps/self_training_PGD_{}steps/weight_{:.1f}/eps_{}/cifar10/resnet110/noise_1.00/test/sigma_1.00".format(num_pgd, weight, eps)), "$\sigma = 1.00$", plot_fmt='r'),

Line(ApproximateAccuracy("data/certify/cifar10/finetune_cifar_from_imagenetPGD2steps/self_training_PGD_{}steps/weight_{:.1f}/eps_{}/cifar10/resnet110/noise_0.12/test/sigma_0.12".format(num_pgd, weight, eps)), "$\sigma = 0.12$", plot_fmt='b'),
Line(ApproximateAccuracy("data/certify/cifar10/finetune_cifar_from_imagenetPGD2steps/self_training_PGD_{}steps/weight_{:.1f}/eps_{}/cifar10/resnet110/noise_0.25/test/sigma_0.25".format(num_pgd, weight, eps)), "$\sigma = 0.25$", plot_fmt='orange'),
Line(ApproximateAccuracy("data/certify/cifar10/finetune_cifar_from_imagenetPGD2steps/self_training_PGD_{}steps/weight_{:.1f}/eps_{}/cifar10/resnet110/noise_0.50/test/sigma_0.50".format(num_pgd, weight, eps)), "$\sigma = 0.50$", plot_fmt='g'),
Line(ApproximateAccuracy("data/certify/cifar10/finetune_cifar_from_imagenetPGD2steps/self_training_PGD_{}steps/weight_{:.1f}/eps_{}/cifar10/resnet110/noise_1.00/test/sigma_1.00".format(num_pgd, weight, eps)), "$\sigma = 1.00$", plot_fmt='r'),

Line(ApproximateAccuracy("data/certify/cifar10/finetune_cifar_from_imagenetPGD2steps/self_training_PGD_{}steps/weight_{:.1f}/eps_{}/cifar10/resnet110/noise_0.12/test/sigma_0.12".format(num_pgd, weight, eps)), "$\sigma = 0.12$", plot_fmt='b'),
Line(ApproximateAccuracy("data/certify/cifar10/finetune_cifar_from_imagenetPGD2steps/self_training_PGD_{}steps/weight_{:.1f}/eps_{}/cifar10/resnet110/noise_0.25/test/sigma_0.25".format(num_pgd, weight, eps)), "$\sigma = 0.25$", plot_fmt='orange'),
Line(ApproximateAccuracy("data/certify/cifar10/finetune_cifar_from_imagenetPGD2steps/self_training_PGD_{}steps/weight_{:.1f}/eps_{}/cifar10/resnet110/noise_0.50/test/sigma_0.50".format(num_pgd, weight, eps)), "$\sigma = 0.50$", plot_fmt='g'),
Line(ApproximateAccuracy("data/certify/cifar10/finetune_cifar_from_imagenetPGD2steps/self_training_PGD_{}steps/weight_{:.1f}/eps_{}/cifar10/resnet110/noise_1.00/test/sigma_1.00".format(num_pgd, weight, eps)), "$\sigma = 1.00$", plot_fmt='r'),

Line(ApproximateAccuracy("data/certify/cifar10/finetune_cifar_from_imagenetPGD2steps/self_training_PGD_{}steps/weight_{:.1f}/eps_{}/cifar10/resnet110/noise_0.12/test/sigma_0.12".format(num_pgd, weight, eps)), "$\sigma = 0.12$", plot_fmt='b'),
Line(ApproximateAccuracy("data/certify/cifar10/finetune_cifar_from_imagenetPGD2steps/self_training_PGD_{}steps/weight_{:.1f}/eps_{}/cifar10/resnet110/noise_0.25/test/sigma_0.25".format(num_pgd, weight, eps)), "$\sigma = 0.25$", plot_fmt='orange'),
Line(ApproximateAccuracy("data/certify/cifar10/finetune_cifar_from_imagenetPGD2steps/self_training_PGD_{}steps/weight_{:.1f}/eps_{}/cifar10/resnet110/noise_0.50/test/sigma_0.50".format(num_pgd, weight, eps)), "$\sigma = 0.50$", plot_fmt='g'),
Line(ApproximateAccuracy("data/certify/cifar10/finetune_cifar_from_imagenetPGD2steps/self_training_PGD_{}steps/weight_{:.1f}/eps_{}/cifar10/resnet110/noise_1.00/test/sigma_1.00".format(num_pgd, weight, eps)), "$\sigma = 1.00$", plot_fmt='r'),
# ]for num_pgd in [4] for weight in [0.1, 0.5, 1.0]for eps in [64, 127, 255, 512]
]for num_pgd in [2, 4, 6, 8, 10] for weight in [0.1, 0.5, 1.0] for eps in [64, 127, 255, 512]
]).tolist()


PGD_imagenetPretraining_cifar10Fineuning_1sample = np.hstack([
[
Line(ApproximateAccuracy("data/certify/cifar10/finetune_cifar_from_imagenetPGD2steps/PGD_{}steps_30epochs/eps_{}/cifar10/resnet110/noise_0.12/test/sigma_0.12".format(num_pgd, eps)), "$\sigma = 0.12$", plot_fmt='b'),
Line(ApproximateAccuracy("data/certify/cifar10/finetune_cifar_from_imagenetPGD2steps/PGD_{}steps_30epochs/eps_{}/cifar10/resnet110/noise_0.25/test/sigma_0.25".format(num_pgd, eps)), "$\sigma = 0.25$", plot_fmt='orange'),
Line(ApproximateAccuracy("data/certify/cifar10/finetune_cifar_from_imagenetPGD2steps/PGD_{}steps_30epochs/eps_{}/cifar10/resnet110/noise_0.50/test/sigma_0.50".format(num_pgd, eps)), "$\sigma = 0.50$", plot_fmt='g'),
Line(ApproximateAccuracy("data/certify/cifar10/finetune_cifar_from_imagenetPGD2steps/PGD_{}steps_30epochs/eps_{}/cifar10/resnet110/noise_1.00/test/sigma_1.00".format(num_pgd, eps)), "$\sigma = 1.00$", plot_fmt='r'),

Line(ApproximateAccuracy("data/certify/cifar10/finetune_cifar_from_imagenetPGD2steps/PGD_{}steps_30epochs/eps_{}/cifar10/resnet110/noise_0.12/test/sigma_0.12".format(num_pgd, eps)), "$\sigma = 0.12$", plot_fmt='b'),
Line(ApproximateAccuracy("data/certify/cifar10/finetune_cifar_from_imagenetPGD2steps/PGD_{}steps_30epochs/eps_{}/cifar10/resnet110/noise_0.25/test/sigma_0.25".format(num_pgd, eps)), "$\sigma = 0.25$", plot_fmt='orange'),
Line(ApproximateAccuracy("data/certify/cifar10/finetune_cifar_from_imagenetPGD2steps/PGD_{}steps_30epochs/eps_{}/cifar10/resnet110/noise_0.50/test/sigma_0.50".format(num_pgd, eps)), "$\sigma = 0.50$", plot_fmt='g'),
Line(ApproximateAccuracy("data/certify/cifar10/finetune_cifar_from_imagenetPGD2steps/PGD_{}steps_30epochs/eps_{}/cifar10/resnet110/noise_1.00/test/sigma_1.00".format(num_pgd, eps)), "$\sigma = 1.00$", plot_fmt='r'),

Line(ApproximateAccuracy("data/certify/cifar10/finetune_cifar_from_imagenetPGD2steps/PGD_{}steps_30epochs/eps_{}/cifar10/resnet110/noise_0.12/test/sigma_0.12".format(num_pgd, eps)), "$\sigma = 0.12$", plot_fmt='b'),
Line(ApproximateAccuracy("data/certify/cifar10/finetune_cifar_from_imagenetPGD2steps/PGD_{}steps_30epochs/eps_{}/cifar10/resnet110/noise_0.25/test/sigma_0.25".format(num_pgd, eps)), "$\sigma = 0.25$", plot_fmt='orange'),
Line(ApproximateAccuracy("data/certify/cifar10/finetune_cifar_from_imagenetPGD2steps/PGD_{}steps_30epochs/eps_{}/cifar10/resnet110/noise_0.50/test/sigma_0.50".format(num_pgd, eps)), "$\sigma = 0.50$", plot_fmt='g'),
Line(ApproximateAccuracy("data/certify/cifar10/finetune_cifar_from_imagenetPGD2steps/PGD_{}steps_30epochs/eps_{}/cifar10/resnet110/noise_1.00/test/sigma_1.00".format(num_pgd, eps)), "$\sigma = 1.00$", plot_fmt='r'),

Line(ApproximateAccuracy("data/certify/cifar10/finetune_cifar_from_imagenetPGD2steps/PGD_{}steps_30epochs/eps_{}/cifar10/resnet110/noise_0.12/test/sigma_0.12".format(num_pgd, eps)), "$\sigma = 0.12$", plot_fmt='b'),
Line(ApproximateAccuracy("data/certify/cifar10/finetune_cifar_from_imagenetPGD2steps/PGD_{}steps_30epochs/eps_{}/cifar10/resnet110/noise_0.25/test/sigma_0.25".format(num_pgd, eps)), "$\sigma = 0.25$", plot_fmt='orange'),
Line(ApproximateAccuracy("data/certify/cifar10/finetune_cifar_from_imagenetPGD2steps/PGD_{}steps_30epochs/eps_{}/cifar10/resnet110/noise_0.50/test/sigma_0.50".format(num_pgd, eps)), "$\sigma = 0.50$", plot_fmt='g'),
Line(ApproximateAccuracy("data/certify/cifar10/finetune_cifar_from_imagenetPGD2steps/PGD_{}steps_30epochs/eps_{}/cifar10/resnet110/noise_1.00/test/sigma_1.00".format(num_pgd, eps)), "$\sigma = 1.00$", plot_fmt='r'),
]for num_pgd in [2, 4, 6, 8, 10] for eps in [64, 127, 255, 512]
]).tolist()


DDN_imagenetPretraining_cifar10Fineuning_1sample = np.hstack([
[
Line(ApproximateAccuracy("data/certify/cifar10/finetune_cifar_from_imagenetPGD2steps/DDN_{}steps_30epochs/eps_{}/cifar10/resnet110/noise_0.12/test/sigma_0.12".format(num_ddn, eps)), "$\sigma = 0.12$", plot_fmt='b'),
Line(ApproximateAccuracy("data/certify/cifar10/finetune_cifar_from_imagenetPGD2steps/DDN_{}steps_30epochs/eps_{}/cifar10/resnet110/noise_0.25/test/sigma_0.25".format(num_ddn, eps)), "$\sigma = 0.25$", plot_fmt='orange'),
Line(ApproximateAccuracy("data/certify/cifar10/finetune_cifar_from_imagenetPGD2steps/DDN_{}steps_30epochs/eps_{}/cifar10/resnet110/noise_0.50/test/sigma_0.50".format(num_ddn, eps)), "$\sigma = 0.50$", plot_fmt='g'),
Line(ApproximateAccuracy("data/certify/cifar10/finetune_cifar_from_imagenetPGD2steps/DDN_{}steps_30epochs/eps_{}/cifar10/resnet110/noise_1.00/test/sigma_1.00".format(num_ddn, eps)), "$\sigma = 1.00$", plot_fmt='r'),

Line(ApproximateAccuracy("data/certify/cifar10/finetune_cifar_from_imagenetPGD2steps/DDN_{}steps_30epochs/eps_{}/cifar10/resnet110/noise_0.12/test/sigma_0.12".format(num_ddn, eps)), "$\sigma = 0.12$", plot_fmt='b'),
Line(ApproximateAccuracy("data/certify/cifar10/finetune_cifar_from_imagenetPGD2steps/DDN_{}steps_30epochs/eps_{}/cifar10/resnet110/noise_0.25/test/sigma_0.25".format(num_ddn, eps)), "$\sigma = 0.25$", plot_fmt='orange'),
Line(ApproximateAccuracy("data/certify/cifar10/finetune_cifar_from_imagenetPGD2steps/DDN_{}steps_30epochs/eps_{}/cifar10/resnet110/noise_0.50/test/sigma_0.50".format(num_ddn, eps)), "$\sigma = 0.50$", plot_fmt='g'),
Line(ApproximateAccuracy("data/certify/cifar10/finetune_cifar_from_imagenetPGD2steps/DDN_{}steps_30epochs/eps_{}/cifar10/resnet110/noise_1.00/test/sigma_1.00".format(num_ddn, eps)), "$\sigma = 1.00$", plot_fmt='r'),

Line(ApproximateAccuracy("data/certify/cifar10/finetune_cifar_from_imagenetPGD2steps/DDN_{}steps_30epochs/eps_{}/cifar10/resnet110/noise_0.12/test/sigma_0.12".format(num_ddn, eps)), "$\sigma = 0.12$", plot_fmt='b'),
Line(ApproximateAccuracy("data/certify/cifar10/finetune_cifar_from_imagenetPGD2steps/DDN_{}steps_30epochs/eps_{}/cifar10/resnet110/noise_0.25/test/sigma_0.25".format(num_ddn, eps)), "$\sigma = 0.25$", plot_fmt='orange'),
Line(ApproximateAccuracy("data/certify/cifar10/finetune_cifar_from_imagenetPGD2steps/DDN_{}steps_30epochs/eps_{}/cifar10/resnet110/noise_0.50/test/sigma_0.50".format(num_ddn, eps)), "$\sigma = 0.50$", plot_fmt='g'),
Line(ApproximateAccuracy("data/certify/cifar10/finetune_cifar_from_imagenetPGD2steps/DDN_{}steps_30epochs/eps_{}/cifar10/resnet110/noise_1.00/test/sigma_1.00".format(num_ddn, eps)), "$\sigma = 1.00$", plot_fmt='r'),

Line(ApproximateAccuracy("data/certify/cifar10/finetune_cifar_from_imagenetPGD2steps/DDN_{}steps_30epochs/eps_{}/cifar10/resnet110/noise_0.12/test/sigma_0.12".format(num_ddn, eps)), "$\sigma = 0.12$", plot_fmt='b'),
Line(ApproximateAccuracy("data/certify/cifar10/finetune_cifar_from_imagenetPGD2steps/DDN_{}steps_30epochs/eps_{}/cifar10/resnet110/noise_0.25/test/sigma_0.25".format(num_ddn, eps)), "$\sigma = 0.25$", plot_fmt='orange'),
Line(ApproximateAccuracy("data/certify/cifar10/finetune_cifar_from_imagenetPGD2steps/DDN_{}steps_30epochs/eps_{}/cifar10/resnet110/noise_0.50/test/sigma_0.50".format(num_ddn, eps)), "$\sigma = 0.50$", plot_fmt='g'),
Line(ApproximateAccuracy("data/certify/cifar10/finetune_cifar_from_imagenetPGD2steps/DDN_{}steps_30epochs/eps_{}/cifar10/resnet110/noise_1.00/test/sigma_1.00".format(num_ddn, eps)), "$\sigma = 1.00$", plot_fmt='r'),
]for num_ddn in [2, 4, 6, 8, 10] for eps in [64, 127, 255, 512]
]).tolist()


PGD_imagenetPretraining_cifar10Fineuning_multinoise = np.hstack([
[
Line(ApproximateAccuracy("data/certify/cifar10/finetune_cifar_from_imagenetPGD2steps/PGD_{}steps_30epochs_multinoise/{}-multitrain/eps_{}/cifar10/resnet110/noise_0.12/test/sigma_0.12".format(num_pgd, num_noise, eps)), "$\sigma = 0.12$", plot_fmt='b'),
Line(ApproximateAccuracy("data/certify/cifar10/finetune_cifar_from_imagenetPGD2steps/PGD_{}steps_30epochs_multinoise/{}-multitrain/eps_{}/cifar10/resnet110/noise_0.25/test/sigma_0.25".format(num_pgd, num_noise, eps)), "$\sigma = 0.25$", plot_fmt='orange'),
Line(ApproximateAccuracy("data/certify/cifar10/finetune_cifar_from_imagenetPGD2steps/PGD_{}steps_30epochs_multinoise/{}-multitrain/eps_{}/cifar10/resnet110/noise_0.50/test/sigma_0.50".format(num_pgd, num_noise, eps)), "$\sigma = 0.50$", plot_fmt='g'),
Line(ApproximateAccuracy("data/certify/cifar10/finetune_cifar_from_imagenetPGD2steps/PGD_{}steps_30epochs_multinoise/{}-multitrain/eps_{}/cifar10/resnet110/noise_1.00/test/sigma_1.00".format(num_pgd, num_noise, eps)), "$\sigma = 1.00$", plot_fmt='r'),

Line(ApproximateAccuracy("data/certify/cifar10/finetune_cifar_from_imagenetPGD2steps/PGD_{}steps_30epochs_multinoise/{}-multitrain/eps_{}/cifar10/resnet110/noise_0.12/test/sigma_0.12".format(num_pgd, num_noise, eps)), "$\sigma = 0.12$", plot_fmt='b'),
Line(ApproximateAccuracy("data/certify/cifar10/finetune_cifar_from_imagenetPGD2steps/PGD_{}steps_30epochs_multinoise/{}-multitrain/eps_{}/cifar10/resnet110/noise_0.25/test/sigma_0.25".format(num_pgd, num_noise, eps)), "$\sigma = 0.25$", plot_fmt='orange'),
Line(ApproximateAccuracy("data/certify/cifar10/finetune_cifar_from_imagenetPGD2steps/PGD_{}steps_30epochs_multinoise/{}-multitrain/eps_{}/cifar10/resnet110/noise_0.50/test/sigma_0.50".format(num_pgd, num_noise, eps)), "$\sigma = 0.50$", plot_fmt='g'),
Line(ApproximateAccuracy("data/certify/cifar10/finetune_cifar_from_imagenetPGD2steps/PGD_{}steps_30epochs_multinoise/{}-multitrain/eps_{}/cifar10/resnet110/noise_1.00/test/sigma_1.00".format(num_pgd, num_noise, eps)), "$\sigma = 1.00$", plot_fmt='r'),

Line(ApproximateAccuracy("data/certify/cifar10/finetune_cifar_from_imagenetPGD2steps/PGD_{}steps_30epochs_multinoise/{}-multitrain/eps_{}/cifar10/resnet110/noise_0.12/test/sigma_0.12".format(num_pgd, num_noise, eps)), "$\sigma = 0.12$", plot_fmt='b'),
Line(ApproximateAccuracy("data/certify/cifar10/finetune_cifar_from_imagenetPGD2steps/PGD_{}steps_30epochs_multinoise/{}-multitrain/eps_{}/cifar10/resnet110/noise_0.25/test/sigma_0.25".format(num_pgd, num_noise, eps)), "$\sigma = 0.25$", plot_fmt='orange'),
Line(ApproximateAccuracy("data/certify/cifar10/finetune_cifar_from_imagenetPGD2steps/PGD_{}steps_30epochs_multinoise/{}-multitrain/eps_{}/cifar10/resnet110/noise_0.50/test/sigma_0.50".format(num_pgd, num_noise, eps)), "$\sigma = 0.50$", plot_fmt='g'),
Line(ApproximateAccuracy("data/certify/cifar10/finetune_cifar_from_imagenetPGD2steps/PGD_{}steps_30epochs_multinoise/{}-multitrain/eps_{}/cifar10/resnet110/noise_1.00/test/sigma_1.00".format(num_pgd, num_noise, eps)), "$\sigma = 1.00$", plot_fmt='r'),

Line(ApproximateAccuracy("data/certify/cifar10/finetune_cifar_from_imagenetPGD2steps/PGD_{}steps_30epochs_multinoise/{}-multitrain/eps_{}/cifar10/resnet110/noise_0.12/test/sigma_0.12".format(num_pgd, num_noise, eps)), "$\sigma = 0.12$", plot_fmt='b'),
Line(ApproximateAccuracy("data/certify/cifar10/finetune_cifar_from_imagenetPGD2steps/PGD_{}steps_30epochs_multinoise/{}-multitrain/eps_{}/cifar10/resnet110/noise_0.25/test/sigma_0.25".format(num_pgd, num_noise, eps)), "$\sigma = 0.25$", plot_fmt='orange'),
Line(ApproximateAccuracy("data/certify/cifar10/finetune_cifar_from_imagenetPGD2steps/PGD_{}steps_30epochs_multinoise/{}-multitrain/eps_{}/cifar10/resnet110/noise_0.50/test/sigma_0.50".format(num_pgd, num_noise, eps)), "$\sigma = 0.50$", plot_fmt='g'),
Line(ApproximateAccuracy("data/certify/cifar10/finetune_cifar_from_imagenetPGD2steps/PGD_{}steps_30epochs_multinoise/{}-multitrain/eps_{}/cifar10/resnet110/noise_1.00/test/sigma_1.00".format(num_pgd, num_noise, eps)), "$\sigma = 1.00$", plot_fmt='r'),
] for num_pgd in [2, 4, 6, 8, 10] for num_noise in [2, 4, 8] for eps in [64, 127, 255, 512]
]).tolist()


PGD_imagenetPretraining_cifar10Fineuning = PGD_imagenetPretraining_cifar10Fineuning_1sample + \
                                        PGD_imagenetPretraining_cifar10Fineuning_multinoise + \
                                        DDN_imagenetPretraining_cifar10Fineuning_1sample


################################################
# All experiments 
all_cifar_experiments = PGD_2_steps_8_samples + \
                PGD_2_steps_4_samples + \
                PGD_2_steps_2_samples + \
                PGD_10_steps_8_samples + \
                PGD_10_steps_4_samples + \
                PGD_10_steps_2_samples + \
                DDN_4_steps_16_samples + \
                DDN_4_steps_8_samples + \
                DDN_4_steps_4_samples + \
                DDN_4_steps_2_samples + \
                DDN_10_steps_16_samples + \
                DDN_10_steps_8_samples + \
                DDN_10_steps_4_samples + \
                DDN_10_steps_2_samples + \
                PGD_10steps + \
                PGD_8steps + \
                PGD_6steps + \
                PGD_4steps + \
                PGD_2steps + \
                DDN_10steps + \
                DDN_8steps + \
                DDN_6steps + \
                DDN_4steps + \
                DDN_2steps

all_cifar_cohen=[
Line(ApproximateAccuracy("data_cohen/certify/cifar10/resnet110/noise_0.12/test/sigma_0.12"), "$\sigma = 0.12$"),
Line(ApproximateAccuracy("data_cohen/certify/cifar10/resnet110/noise_0.25/test/sigma_0.25"), "$\sigma = 0.25$"),
Line(ApproximateAccuracy("data_cohen/certify/cifar10/resnet110/noise_0.50/test/sigma_0.50"), "$\sigma = 0.50$"),
Line(ApproximateAccuracy("data_cohen/certify/cifar10/resnet110/noise_1.00/test/sigma_1.00"), "$\sigma = 1.00$"),
]

all_cifar_cohen_fulldataset=[
Line(ApproximateAccuracy("data/certify/best_models/cifar10/cohen/resnet110/noise_0.12/test/sigma_0.12"), "$\sigma = 0.12$"),
Line(ApproximateAccuracy("data/certify/best_models/cifar10/cohen/resnet110/noise_0.25/test/sigma_0.25"), "$\sigma = 0.25$"),
Line(ApproximateAccuracy("data/certify/best_models/cifar10/cohen/resnet110/noise_0.50/test/sigma_0.50"), "$\sigma = 0.50$"),
Line(ApproximateAccuracy("data/certify/best_models/cifar10/cohen/resnet110/noise_1.00/test/sigma_1.00"), "$\sigma = 1.00$"),
]

all_imagenet_cohen=[
Line(ApproximateAccuracy("data_cohen/certify/imagenet/resnet50/noise_0.25/test/sigma_0.25"), "$\sigma = 0.25$"),
Line(ApproximateAccuracy("data_cohen/certify/imagenet/resnet50/noise_0.50/test/sigma_0.50"), "$\sigma = 0.50$"),
Line(ApproximateAccuracy("data_cohen/certify/imagenet/resnet50/noise_1.00/test/sigma_1.00"), "$\sigma = 1.00$"),
]

all_imagenet_cohen_replicate=[
Line(ApproximateAccuracy("data/certify/imagenet/replication/resnet50/noise_0.25/test/sigma_0.25"), "$\sigma = 0.25$", plot_fmt='b'),
Line(ApproximateAccuracy("data/certify/imagenet/replication/resnet50/noise_0.50/test/sigma_0.50"), "$\sigma = 0.50$", plot_fmt='orange'),
Line(ApproximateAccuracy("data/certify/imagenet/replication/resnet50/noise_1.00/test/sigma_1.00"), "$\sigma = 1.00$", plot_fmt='g'),
]

all_imagenet_cohen_500samples=[
Line(ApproximateAccuracy("data/certify/best_models/imagenet/replication/resnet50/noise_0.25/test/sigma_0.25"), "$\sigma = 0.25$", plot_fmt='b'),
Line(ApproximateAccuracy("data/certify/best_models/imagenet/replication/resnet50/noise_0.50/test/sigma_0.50"), "$\sigma = 0.50$", plot_fmt='orange'),
Line(ApproximateAccuracy("data/certify/best_models/imagenet/replication/resnet50/noise_1.00/test/sigma_1.00"), "$\sigma = 1.00$", plot_fmt='g'),
]

## Our best models

best_cifar10 = [
    Line(ApproximateAccuracy("data/certify/best_models/cifar10/ours/cifar10/DDN_10steps_multiNoiseSamples/8-multitrain/eps_64/cifar10/resnet110/noise_0.12/test/sigma_0.12")),
    Line(ApproximateAccuracy("data/certify/best_models/cifar10/ours/cifar10/DDN_10steps_multiNoiseSamples/4-multitrain/eps_64/cifar10/resnet110/noise_0.12/test/sigma_0.12")),
    Line(ApproximateAccuracy("data/certify/best_models/cifar10/ours/cifar10/PGD_10steps_multiNoiseSamples/4-multitrain/eps_64/cifar10/resnet110/noise_0.12/test/sigma_0.12")),
    Line(ApproximateAccuracy("data/certify/best_models/cifar10/ours/cifar10/DDN_4steps_multiNoiseSamples/4-multitrain/eps_255/cifar10/resnet110/noise_0.12/test/sigma_0.12")),
    Line(ApproximateAccuracy("data/certify/best_models/cifar10/ours/cifar10/PGD_10steps_multiNoiseSamples/8-multitrain/eps_255/cifar10/resnet110/noise_0.25/test/sigma_0.25")),
    Line(ApproximateAccuracy("data/certify/best_models/cifar10/ours/cifar10/PGD_2steps_multiNoiseSamples/8-multitrain/eps_255/cifar10/resnet110/noise_0.25/test/sigma_0.25")),
    Line(ApproximateAccuracy("data/certify/best_models/cifar10/ours/cifar10/PGD_10steps_multiNoiseSamples/4-multitrain/eps_255/cifar10/resnet110/noise_0.25/test/sigma_0.25")),
    Line(ApproximateAccuracy("data/certify/best_models/cifar10/ours/cifar10/PGD_2steps_multiNoiseSamples/8-multitrain/eps_512/cifar10/resnet110/noise_0.25/test/sigma_0.25")),
    Line(ApproximateAccuracy("data/certify/best_models/cifar10/ours/cifar10/PGD_2steps_multiNoiseSamples/8-multitrain/eps_512/cifar10/resnet110/noise_0.50/test/sigma_0.50")),
    Line(ApproximateAccuracy("data/certify/best_models/cifar10/ours/cifar10/PGD_2steps_multiNoiseSamples/8-multitrain/eps_512/cifar10/resnet110/noise_0.50/test/sigma_0.50")),
    Line(ApproximateAccuracy("data/certify/best_models/cifar10/ours/cifar10/PGD_2steps_multiNoiseSamples/2-multitrain/eps_512/cifar10/resnet110/noise_0.50/test/sigma_0.50")),
    Line(ApproximateAccuracy("data/certify/best_models/cifar10/ours/cifar10/PGD_2steps_multiNoiseSamples/2-multitrain/eps_512/cifar10/resnet110/noise_0.50/test/sigma_0.50")),
    Line(ApproximateAccuracy("data/certify/best_models/cifar10/ours/cifar10/PGD_10steps_multiNoiseSamples/2-multitrain/eps_512/cifar10/resnet110/noise_0.50/test/sigma_0.50")),
    Line(ApproximateAccuracy("data/certify/best_models/cifar10/ours/cifar10/PGD_4steps/eps_512/cifar10/resnet110/noise_0.50/test/sigma_0.50")),
    Line(ApproximateAccuracy("data/certify/best_models/cifar10/ours/cifar10/PGD_4steps/eps_512/cifar10/resnet110/noise_0.50/test/sigma_0.50")),
    Line(ApproximateAccuracy("data/certify/best_models/cifar10/ours/cifar10/PGD_10steps_multiNoiseSamples/2-multitrain/eps_512/cifar10/resnet110/noise_1.00/test/sigma_1.00")),
    Line(ApproximateAccuracy("data/certify/best_models/cifar10/ours/cifar10/PGD_10steps_multiNoiseSamples/2-multitrain/eps_512/cifar10/resnet110/noise_1.00/test/sigma_1.00")),
    Line(ApproximateAccuracy("data/certify/best_models/cifar10/ours/cifar10/PGD_2steps_multiNoiseSamples/4-multitrain/eps_512/cifar10/resnet110/noise_1.00/test/sigma_1.00")),
    Line(ApproximateAccuracy("data/certify/best_models/cifar10/ours/cifar10/PGD_10steps_multiNoiseSamples/2-multitrain/eps_512/cifar10/resnet110/noise_1.00/test/sigma_1.00")),
]

best_cifar10_pretraining = [
    Line(ApproximateAccuracy("data/certify/best_models/cifar10/ours_pretraining/cifar10/finetune_cifar_from_imagenetPGD2steps/DDN_2steps_30epochs/eps_64/cifar10/resnet110/noise_0.12/test/sigma_0.12")),
    Line(ApproximateAccuracy("data/certify/best_models/cifar10/ours_pretraining/cifar10/finetune_cifar_from_imagenetPGD2steps/DDN_4steps_30epochs/eps_64/cifar10/resnet110/noise_0.12/test/sigma_0.12")),
    Line(ApproximateAccuracy("data/certify/best_models/cifar10/ours_pretraining/cifar10/finetune_cifar_from_imagenetPGD2steps/PGD_6steps_30epochs/eps_64/cifar10/resnet110/noise_0.12/test/sigma_0.12")),
    Line(ApproximateAccuracy("data/certify/best_models/cifar10/ours_pretraining/cifar10/finetune_cifar_from_imagenetPGD2steps/PGD_10steps_30epochs/eps_127/cifar10/resnet110/noise_0.12/test/sigma_0.12")),
    Line(ApproximateAccuracy("data/certify/best_models/cifar10/ours_pretraining/cifar10/finetune_cifar_from_imagenetPGD2steps/PGD_4steps_30epochs/eps_127/cifar10/resnet110/noise_0.25/test/sigma_0.25")),
    Line(ApproximateAccuracy("data/certify/best_models/cifar10/ours_pretraining/cifar10/finetune_cifar_from_imagenetPGD2steps/PGD_4steps_30epochs_multinoise/2-multitrain/eps_127/cifar10/resnet110/noise_0.25/test/sigma_0.25")),
    Line(ApproximateAccuracy("data/certify/best_models/cifar10/ours_pretraining/cifar10/finetune_cifar_from_imagenetPGD2steps/PGD_6steps_30epochs_multinoise/4-multitrain/eps_255/cifar10/resnet110/noise_0.25/test/sigma_0.25")),
    Line(ApproximateAccuracy("data/certify/best_models/cifar10/ours_pretraining/cifar10/finetune_cifar_from_imagenetPGD2steps/PGD_2steps_30epochs_multinoise/2-multitrain/eps_255/cifar10/resnet110/noise_0.25/test/sigma_0.25")),
    Line(ApproximateAccuracy("data/certify/best_models/cifar10/ours_pretraining/cifar10/finetune_cifar_from_imagenetPGD2steps/PGD_8steps_30epochs_multinoise/4-multitrain/eps_512/cifar10/resnet110/noise_0.50/test/sigma_0.50")),
    Line(ApproximateAccuracy("data/certify/best_models/cifar10/ours_pretraining/cifar10/finetune_cifar_from_imagenetPGD2steps/PGD_2steps_30epochs_multinoise/4-multitrain/eps_512/cifar10/resnet110/noise_0.50/test/sigma_0.50")),
    Line(ApproximateAccuracy("data/certify/best_models/cifar10/ours_pretraining/cifar10/finetune_cifar_from_imagenetPGD2steps/PGD_2steps_30epochs_multinoise/4-multitrain/eps_512/cifar10/resnet110/noise_0.50/test/sigma_0.50")),
    Line(ApproximateAccuracy("data/certify/best_models/cifar10/ours_pretraining/cifar10/finetune_cifar_from_imagenetPGD2steps/PGD_2steps_30epochs_multinoise/4-multitrain/eps_512/cifar10/resnet110/noise_0.50/test/sigma_0.50")),
    Line(ApproximateAccuracy("data/certify/best_models/cifar10/ours_pretraining/cifar10/finetune_cifar_from_imagenetPGD2steps/PGD_4steps_30epochs_multinoise/4-multitrain/eps_512/cifar10/resnet110/noise_0.50/test/sigma_0.50")),
    Line(ApproximateAccuracy("data/certify/best_models/cifar10/ours_pretraining/cifar10/finetune_cifar_from_imagenetPGD2steps/PGD_6steps_30epochs/eps_512/cifar10/resnet110/noise_0.50/test/sigma_0.50")),
    Line(ApproximateAccuracy("data/certify/best_models/cifar10/ours_pretraining/cifar10/finetune_cifar_from_imagenetPGD2steps/PGD_6steps_30epochs_multinoise/2-multitrain/eps_512/cifar10/resnet110/noise_0.50/test/sigma_0.50")),
    Line(ApproximateAccuracy("data/certify/best_models/cifar10/ours_pretraining/cifar10/finetune_cifar_from_imagenetPGD2steps/PGD_6steps_30epochs/eps_512/cifar10/resnet110/noise_1.00/test/sigma_1.00")),
    Line(ApproximateAccuracy("data/certify/best_models/cifar10/ours_pretraining/cifar10/finetune_cifar_from_imagenetPGD2steps/PGD_6steps_30epochs/eps_512/cifar10/resnet110/noise_1.00/test/sigma_1.00")),
    Line(ApproximateAccuracy("data/certify/best_models/cifar10/ours_pretraining/cifar10/finetune_cifar_from_imagenetPGD2steps/PGD_6steps_30epochs/eps_512/cifar10/resnet110/noise_1.00/test/sigma_1.00")),
    Line(ApproximateAccuracy("data/certify/best_models/cifar10/ours_pretraining/cifar10/finetune_cifar_from_imagenetPGD2steps/PGD_6steps_30epochs/eps_512/cifar10/resnet110/noise_1.00/test/sigma_1.00")),
]

best_cifar10_selftraining = [
    Line(ApproximateAccuracy("data/certify/best_models/cifar10/ours_selftraining/cifar10/self_training/PGD_4steps/weight_0.5/eps_64/cifar10/resnet110/noise_0.12/test/sigma_0.12")),
    Line(ApproximateAccuracy("data/certify/best_models/cifar10/ours_selftraining/cifar10/self_training/PGD_2steps/weight_1.0/eps_64/cifar10/resnet110/noise_0.12/test/sigma_0.12")),
    Line(ApproximateAccuracy("data/certify/best_models/cifar10/ours_selftraining/cifar10/self_training/PGD_4steps/weight_1.0/eps_64/cifar10/resnet110/noise_0.12/test/sigma_0.12")),
    Line(ApproximateAccuracy("data/certify/best_models/cifar10/ours_selftraining/cifar10/self_training/PGD_8steps/weight_1.0/eps_64/cifar10/resnet110/noise_0.12/test/sigma_0.12")),
    Line(ApproximateAccuracy("data/certify/best_models/cifar10/ours_selftraining/cifar10/self_training/PGD_4steps/weight_1.0/eps_127/cifar10/resnet110/noise_0.25/test/sigma_0.25")),
    Line(ApproximateAccuracy("data/certify/best_models/cifar10/ours_selftraining/cifar10/self_training/PGD_4steps/weight_1.0/eps_127/cifar10/resnet110/noise_0.25/test/sigma_0.25")),
    Line(ApproximateAccuracy("data/certify/best_models/cifar10/ours_selftraining/cifar10/self_training/PGD_2steps/weight_1.0/eps_255/cifar10/resnet110/noise_0.25/test/sigma_0.25")),
    Line(ApproximateAccuracy("data/certify/best_models/cifar10/ours_selftraining/cifar10/self_training/PGD_2steps/weight_1.0/eps_255/cifar10/resnet110/noise_0.25/test/sigma_0.25")),
    Line(ApproximateAccuracy("data/certify/best_models/cifar10/ours_selftraining/cifar10/self_training/PGD_2steps/weight_0.5/eps_255/cifar10/resnet110/noise_0.50/test/sigma_0.50")),
    Line(ApproximateAccuracy("data/certify/best_models/cifar10/ours_selftraining/cifar10/self_training/PGD_8steps/weight_1.0/eps_512/cifar10/resnet110/noise_0.50/test/sigma_0.50")),
    Line(ApproximateAccuracy("data/certify/best_models/cifar10/ours_selftraining/cifar10/self_training/PGD_4steps/weight_1.0/eps_512/cifar10/resnet110/noise_0.50/test/sigma_0.50")),
    Line(ApproximateAccuracy("data/certify/best_models/cifar10/ours_selftraining/cifar10/self_training/PGD_4steps/weight_1.0/eps_512/cifar10/resnet110/noise_0.50/test/sigma_0.50")),
    Line(ApproximateAccuracy("data/certify/best_models/cifar10/ours_selftraining/cifar10/self_training/PGD_4steps/weight_1.0/eps_512/cifar10/resnet110/noise_0.50/test/sigma_0.50")),
    Line(ApproximateAccuracy("data/certify/best_models/cifar10/ours_selftraining/cifar10/self_training/PGD_4steps/weight_1.0/eps_512/cifar10/resnet110/noise_0.50/test/sigma_0.50")),
    Line(ApproximateAccuracy("data/certify/best_models/cifar10/ours_selftraining/cifar10/self_training/PGD_4steps/weight_1.0/eps_512/cifar10/resnet110/noise_0.50/test/sigma_0.50")),
    Line(ApproximateAccuracy("data/certify/best_models/cifar10/ours_selftraining/cifar10/self_training/PGD_8steps/weight_0.5/eps_512/cifar10/resnet110/noise_1.00/test/sigma_1.00")),
    Line(ApproximateAccuracy("data/certify/best_models/cifar10/ours_selftraining/cifar10/self_training/PGD_8steps/weight_0.5/eps_512/cifar10/resnet110/noise_1.00/test/sigma_1.00")),
    Line(ApproximateAccuracy("data/certify/best_models/cifar10/ours_selftraining/cifar10/self_training/PGD_10steps/weight_1.0/eps_512/cifar10/resnet110/noise_1.00/test/sigma_1.00")),
    Line(ApproximateAccuracy("data/certify/best_models/cifar10/ours_selftraining/cifar10/self_training/PGD_8steps/weight_0.5/eps_512/cifar10/resnet110/noise_1.00/test/sigma_1.00")),
]

best_cifar10_pretraining_selftraining = [
    Line(ApproximateAccuracy("data/certify/best_models/cifar10/ours_pretraining_selftraining/cifar10/finetune_cifar_from_imagenetPGD2steps/self_training_PGD_2steps/weight_1.0/eps_64/cifar10/resnet110/noise_0.12/test/sigma_0.12")),
    Line(ApproximateAccuracy("data/certify/best_models/cifar10/ours_pretraining_selftraining/cifar10/finetune_cifar_from_imagenetPGD2steps/self_training_PGD_10steps/weight_1.0/eps_64/cifar10/resnet110/noise_0.12/test/sigma_0.12")),
    Line(ApproximateAccuracy("data/certify/best_models/cifar10/ours_pretraining_selftraining/cifar10/finetune_cifar_from_imagenetPGD2steps/self_training_PGD_2steps/weight_1.0/eps_64/cifar10/resnet110/noise_0.12/test/sigma_0.12")),
    Line(ApproximateAccuracy("data/certify/best_models/cifar10/ours_pretraining_selftraining/cifar10/finetune_cifar_from_imagenetPGD2steps/self_training_PGD_4steps/weight_1.0/eps_127/cifar10/resnet110/noise_0.12/test/sigma_0.12")),
    Line(ApproximateAccuracy("data/certify/best_models/cifar10/ours_pretraining_selftraining/cifar10/finetune_cifar_from_imagenetPGD2steps/self_training_PGD_2steps/weight_0.5/eps_64/cifar10/resnet110/noise_0.25/test/sigma_0.25")),
    Line(ApproximateAccuracy("data/certify/best_models/cifar10/ours_pretraining_selftraining/cifar10/finetune_cifar_from_imagenetPGD2steps/self_training_PGD_4steps/weight_1.0/eps_127/cifar10/resnet110/noise_0.25/test/sigma_0.25")),
    Line(ApproximateAccuracy("data/certify/best_models/cifar10/ours_pretraining_selftraining/cifar10/finetune_cifar_from_imagenetPGD2steps/self_training_PGD_6steps/weight_1.0/eps_255/cifar10/resnet110/noise_0.25/test/sigma_0.25")),
    Line(ApproximateAccuracy("data/certify/best_models/cifar10/ours_pretraining_selftraining/cifar10/finetune_cifar_from_imagenetPGD2steps/self_training_PGD_6steps/weight_1.0/eps_255/cifar10/resnet110/noise_0.25/test/sigma_0.25")),
    Line(ApproximateAccuracy("data/certify/best_models/cifar10/ours_pretraining_selftraining/cifar10/finetune_cifar_from_imagenetPGD2steps/self_training_PGD_6steps/weight_0.5/eps_512/cifar10/resnet110/noise_0.50/test/sigma_0.50")),
    Line(ApproximateAccuracy("data/certify/best_models/cifar10/ours_pretraining_selftraining/cifar10/finetune_cifar_from_imagenetPGD2steps/self_training_PGD_2steps/weight_0.5/eps_512/cifar10/resnet110/noise_0.50/test/sigma_0.50")),
    Line(ApproximateAccuracy("data/certify/best_models/cifar10/ours_pretraining_selftraining/cifar10/finetune_cifar_from_imagenetPGD2steps/self_training_PGD_2steps/weight_0.5/eps_512/cifar10/resnet110/noise_0.50/test/sigma_0.50")),
    Line(ApproximateAccuracy("data/certify/best_models/cifar10/ours_pretraining_selftraining/cifar10/finetune_cifar_from_imagenetPGD2steps/self_training_PGD_2steps/weight_0.5/eps_512/cifar10/resnet110/noise_0.50/test/sigma_0.50")),
    Line(ApproximateAccuracy("data/certify/best_models/cifar10/ours_pretraining_selftraining/cifar10/finetune_cifar_from_imagenetPGD2steps/self_training_PGD_2steps/weight_0.5/eps_512/cifar10/resnet110/noise_0.50/test/sigma_0.50")),
    Line(ApproximateAccuracy("data/certify/best_models/cifar10/ours_pretraining_selftraining/cifar10/finetune_cifar_from_imagenetPGD2steps/self_training_PGD_8steps/weight_0.5/eps_512/cifar10/resnet110/noise_0.50/test/sigma_0.50")),
    Line(ApproximateAccuracy("data/certify/best_models/cifar10/ours_pretraining_selftraining/cifar10/finetune_cifar_from_imagenetPGD2steps/self_training_PGD_8steps/weight_0.5/eps_512/cifar10/resnet110/noise_0.50/test/sigma_0.50")),
    Line(ApproximateAccuracy("data/certify/best_models/cifar10/ours_pretraining_selftraining/cifar10/finetune_cifar_from_imagenetPGD2steps/self_training_PGD_8steps/weight_0.1/eps_512/cifar10/resnet110/noise_0.50/test/sigma_0.50")),
    Line(ApproximateAccuracy("data/certify/best_models/cifar10/ours_pretraining_selftraining/cifar10/finetune_cifar_from_imagenetPGD2steps/self_training_PGD_6steps/weight_0.5/eps_512/cifar10/resnet110/noise_1.00/test/sigma_1.00")),
    Line(ApproximateAccuracy("data/certify/best_models/cifar10/ours_pretraining_selftraining/cifar10/finetune_cifar_from_imagenetPGD2steps/self_training_PGD_8steps/weight_1.0/eps_512/cifar10/resnet110/noise_1.00/test/sigma_1.00")),
    Line(ApproximateAccuracy("data/certify/best_models/cifar10/ours_pretraining_selftraining/cifar10/finetune_cifar_from_imagenetPGD2steps/self_training_PGD_4steps/weight_0.5/eps_512/cifar10/resnet110/noise_1.00/test/sigma_1.00")),
]

all_imagenet_experiments_500samples = [
    Line(ApproximateAccuracy("data/certify/best_models/imagenet/PGD_1step/eps_127/resnet50/noise_0.25/test/sigma_0.25"), "$\sigma = 0.25$ | $\epsilon = 0.5$"),
    Line(ApproximateAccuracy("data/certify/best_models/imagenet/PGD_1step/eps_255/resnet50/noise_0.25/test/sigma_0.25"), "$\sigma = 0.25$ | $\epsilon = 1.0$"),
    Line(ApproximateAccuracy("data/certify/best_models/imagenet/PGD_1step/eps_512/resnet50/noise_0.25/test/sigma_0.25"), "$\sigma = 0.25$ | $\epsilon = 2.0$"),
    Line(ApproximateAccuracy("data/certify/best_models/imagenet/PGD_1step/eps_1024/resnet50/noise_0.25/test/sigma_0.25"), "$\sigma = 0.25$ | $\epsilon = 4.0$"),

    Line(ApproximateAccuracy("data/certify/best_models/imagenet/PGD_1step/eps_127/resnet50/noise_0.50/test/sigma_0.50"), "$\sigma = 0.50$ | $\epsilon = 0.5$"),
    Line(ApproximateAccuracy("data/certify/best_models/imagenet/PGD_1step/eps_255/resnet50/noise_0.50/test/sigma_0.50"), "$\sigma = 0.50$ | $\epsilon = 1.0$"),
    Line(ApproximateAccuracy("data/certify/best_models/imagenet/PGD_1step/eps_512/resnet50/noise_0.50/test/sigma_0.50"), "$\sigma = 0.50$ | $\epsilon = 2.0$"),
    Line(ApproximateAccuracy("data/certify/best_models/imagenet/PGD_1step/eps_1024/resnet50/noise_0.50/test/sigma_0.50"), "$\sigma = 0.50$ | $\epsilon = 4.0$"),

    Line(ApproximateAccuracy("data/certify/best_models/imagenet/PGD_1step/eps_127/resnet50/noise_1.00/test/sigma_1.00"), "$\sigma = 1.00$ | $\epsilon = 0.5$"),
    Line(ApproximateAccuracy("data/certify/best_models/imagenet/PGD_1step/eps_255/resnet50/noise_1.00/test/sigma_1.00"), "$\sigma = 1.00$ | $\epsilon = 1.0$"),
    Line(ApproximateAccuracy("data/certify/best_models/imagenet/PGD_1step/eps_512/resnet50/noise_1.00/test/sigma_1.00"), "$\sigma = 1.00$ | $\epsilon = 2.0$"),
    Line(ApproximateAccuracy("data/certify/best_models/imagenet/PGD_1step/eps_1024/resnet50/noise_1.00/test/sigma_1.00"), "$\sigma = 1.00$ | $\epsilon = 4.0$"),

    Line(ApproximateAccuracy("data/certify/best_models/imagenet/DDN_2steps/eps_127/resnet50/noise_0.25/test/sigma_0.25"), "$\sigma = 0.25$ | $\epsilon = 0.5$"),
    Line(ApproximateAccuracy("data/certify/best_models/imagenet/DDN_2steps/eps_255/resnet50/noise_0.25/test/sigma_0.25"), "$\sigma = 0.25$ | $\epsilon = 1.0$"),
    Line(ApproximateAccuracy("data/certify/best_models/imagenet/DDN_2steps/eps_512/resnet50/noise_0.25/test/sigma_0.25"), "$\sigma = 0.25$ | $\epsilon = 2.0$"),
    Line(ApproximateAccuracy("data/certify/best_models/imagenet/DDN_2steps/eps_1024/resnet50/noise_0.25/test/sigma_0.25"), "$\sigma = 0.25$ | $\epsilon = 4.0$"),

    Line(ApproximateAccuracy("data/certify/best_models/imagenet/DDN_2steps/eps_127/resnet50/noise_0.50/test/sigma_0.50"), "$\sigma = 0.50$ | $\epsilon = 0.5$"),
    Line(ApproximateAccuracy("data/certify/best_models/imagenet/DDN_2steps/eps_255/resnet50/noise_0.50/test/sigma_0.50"), "$\sigma = 0.50$ | $\epsilon = 1.0$"),
    Line(ApproximateAccuracy("data/certify/best_models/imagenet/DDN_2steps/eps_512/resnet50/noise_0.50/test/sigma_0.50"), "$\sigma = 0.50$ | $\epsilon = 2.0$"),
    Line(ApproximateAccuracy("data/certify/best_models/imagenet/DDN_2steps/eps_1024/resnet50/noise_0.50/test/sigma_0.50"), "$\sigma = 0.50$ | $\epsilon = 4.0$"),

    Line(ApproximateAccuracy("data/certify/best_models/imagenet/DDN_2steps/eps_127/resnet50/noise_1.00/test/sigma_1.00"), "$\sigma = 1.00$ | $\epsilon = 0.5$"),
    Line(ApproximateAccuracy("data/certify/best_models/imagenet/DDN_2steps/eps_255/resnet50/noise_1.00/test/sigma_1.00"), "$\sigma = 1.00$ | $\epsilon = 1.0$"),
    Line(ApproximateAccuracy("data/certify/best_models/imagenet/DDN_2steps/eps_512/resnet50/noise_1.00/test/sigma_1.00"), "$\sigma = 1.00$ | $\epsilon = 2.0$"),
    Line(ApproximateAccuracy("data/certify/best_models/imagenet/DDN_2steps/eps_1024/resnet50/noise_1.00/test/sigma_1.00"), "$\sigma = 1.00$ | $\epsilon = 4.0$"),
]

if __name__ == "__main__":
    latex_table_abstention_rate(
        "analysis/latex_new/cohen_cifar10_abstention_rates", 0.25, 2.25, 0.25, all_cifar_cohen_fulldataset)   
  
    latex_table_abstention_rate(
        "analysis/latex_new/our_cifar10_abstention_rates", 0.25, 2.25, 0.25, best_cifar10)   

    latex_table_abstention_rate(
        "analysis/latex_new/our_cifar10_abstention_rates_self_training", 0.25, 2.25, 0.25, best_cifar10_selftraining)

    latex_table_abstention_rate(
        "analysis/latex_new/our_cifar10_abstention_rates_pretraining", 0.25, 2.25, 0.25, best_cifar10_pretraining)

    latex_table_abstention_rate(
        "analysis/latex_new/our_cifar10_abstention_rates_pretraining_self_training", 0.25, 2.25, 0.25, best_cifar10_pretraining_selftraining)

# Latex New (updated main latex tables: CIFAR10 full dataset / ImageeNet 500 samples)
    latex_table_certified_accuracy_upper_envelope(
        "analysis/latex_new/cohen_cifar10_certified_outer_envelop", 0.25, 2.25, 0.25, all_cifar_cohen_fulldataset)   

    latex_table_certified_accuracy_upper_envelope(
        "analysis/latex_new/cohen_imagenet_certified_outer_envelop", 0.5, 3.5, 0.5, all_imagenet_cohen_500samples)   

    latex_table_certified_accuracy_upper_envelope(
        "analysis/latex_new/our_cifar10_certified_outer_envelop", 0.25, 2.25, 0.25, best_cifar10)   

    latex_table_certified_accuracy_upper_envelope(
        "analysis/latex_new/our_imagenet_certified_outer_envelop", 0.5, 3.5, 0.5, all_imagenet_experiments_500samples)

    latex_table_certified_accuracy_upper_envelope(
        "analysis/latex_new/our_cifar10_certified_outer_envelop_self_training", 0.25, 2.25, 0.25, best_cifar10_selftraining)

    latex_table_certified_accuracy_upper_envelope(
        "analysis/latex_new/our_cifar10_certified_outer_envelop_pretraining", 0.25, 2.25, 0.25, best_cifar10_pretraining)

    latex_table_certified_accuracy_upper_envelope(
        "analysis/latex_new/our_cifar10_certified_outer_envelop_pretraining_Custom", 0.4347, 0.5, 0.25, best_cifar10_pretraining)

    latex_table_certified_accuracy_upper_envelope(
        "analysis/latex_new/our_cifar10_certified_outer_envelop_pretraining_self_training", 0.25, 2.25, 0.25, best_cifar10_pretraining_selftraining)

    radii_to_best_models(
        "analysis/radii_to_best_models/cifar10_original",
            all_cifar_cohen_fulldataset, max_radius=2.25, radius_step=0.125)

    radii_to_best_models(
        "analysis/radii_to_best_models/cifar10_ours", 
            best_cifar10, max_radius=2.25, radius_step=0.125)

    radii_to_best_models(
        "analysis/radii_to_best_models/cifar10_ours_pretraining", 
            best_cifar10_pretraining, max_radius=2.25, radius_step=0.125)

    radii_to_best_models(
        "analysis/radii_to_best_models/cifar10_ours_pretraining_selftraining",
            best_cifar10_pretraining_selftraining, max_radius=2.25, radius_step=0.125)

    radii_to_best_models(
        "analysis/radii_to_best_models/cifar10_ours_selftraining", 
            best_cifar10_selftraining, max_radius=2.25, radius_step=0.125)

    radii_to_best_models(
        "analysis/radii_to_best_models/imagenet_original", 
            all_imagenet_cohen_500samples, max_radius=3.5, radius_step=0.125)

    radii_to_best_models(
        "analysis/radii_to_best_models/imagenet_ours", 
            all_imagenet_experiments_500samples, max_radius=3.5, radius_step=0.125)
    exit()
# Latex
    latex_table_certified_accuracy(
        "analysis/latex/vary_noise_cifar10_cohen", 0.25, 1.5, 0.25, all_cifar_cohen)

    latex_table_certified_accuracy(
        "analysis/latex/vary_noise_imagenet_cohen", 0.5, 3.5, 0.5, all_imagenet_cohen)

    latex_table_certified_accuracy_upper_envelope(
        "analysis/latex/cohen_cifar10_certified_outer_envelop", 0.25, 2.25, 0.25, all_cifar_cohen)   

    latex_table_certified_accuracy_upper_envelope(
        "analysis/latex/cohen_imagenet_certified_outer_envelop", 0.5, 3.5, 0.5, all_imagenet_cohen)   

    latex_table_certified_accuracy_upper_envelope(
        "analysis/latex/our_cifar10_certified_outer_envelop", 0.25, 2.25, 0.25, all_cifar_experiments)   

    latex_table_certified_accuracy_upper_envelope(
        "analysis/latex/our_imagenet_certified_outer_envelop", 0.5, 3.5, 0.5, all_imagenet_experiments)

    latex_table_certified_accuracy_upper_envelope(
        "analysis/latex/our_cifar10_certified_outer_envelop_self_training", 0.25, 2.25, 0.25, PGD_self_training)

    latex_table_certified_accuracy_upper_envelope(
        "analysis/latex/our_cifar10_certified_outer_envelop_pretraining", 0.25, 2.25, 0.25, PGD_imagenetPretraining_cifar10Fineuning)

    latex_table_certified_accuracy_upper_envelope(
        "analysis/latex/our_cifar10_certified_outer_envelop_pretraining_Custom", 0.4347, 0.5, 0.25, PGD_imagenetPretraining_cifar10Fineuning)

    latex_table_certified_accuracy_upper_envelope(
        "analysis/latex/our_cifar10_certified_outer_envelop_pretraining_self_training", 0.25, 2.25, 0.25, PGD_imagenetPretraining_cifar10Fineuning_self_training)


    latex_table_certified_accuracy(
        "analysis/latex/vary_noise_imagenet_PGD_and_DDN", 0.0, 3.5, 0.5, all_imagenet_cohen + all_imagenet_experiments)
    

    #### PGD and DDN varying steps results
    ## PGD/DDN 2 step 1 sample vs Cohen's
    latex_table_certified_accuracy(
        "analysis/latex/vary_noise_cifar10_PGD_DDN_2steps", 0.0, 2.25, 0.25, all_cifar_cohen + PGD_2steps + DDN_2steps)


    ## PGD/DDN 4 step 1 sample vs Cohen's
    latex_table_certified_accuracy(
        "analysis/latex/vary_noise_cifar10_PGD_DDN_4steps", 0.0, 2.25, 0.25, all_cifar_cohen + PGD_4steps + DDN_4steps)

    ## PGD/DDN 6 step 1 sample vs Cohen's
    latex_table_certified_accuracy(
        "analysis/latex/vary_noise_cifar10_PGD_DDN_6steps", 0.0, 2.25, 0.25, all_cifar_cohen + PGD_6steps + DDN_6steps)


    ## PGD/DDN 8 step 1 sample vs Cohen's
    latex_table_certified_accuracy(
        "analysis/latex/vary_noise_cifar10_PGD_DDN_8steps", 0.0, 2.25, 0.25, all_cifar_cohen + PGD_8steps + DDN_8steps)

    ## PGD/DDN 10 step 1 sample vs Cohen's
    latex_table_certified_accuracy(
        "analysis/latex/vary_noise_cifar10_PGD_DDN_10steps", 0.0, 2.25, 0.25, all_cifar_cohen + PGD_10steps + DDN_10steps)


    #### DDN varying number of samples
    ## DDN 4 step 2/4/8/16 sample vs Cohen's
    latex_table_certified_accuracy(
        "analysis/latex/vary_noise_cifar10_DDN_4steps_multinoise", 0.0, 2.25, 0.25, all_cifar_cohen + DDN_4steps + DDN_4_steps_2_samples + DDN_4_steps_4_samples + DDN_4_steps_8_samples)


    ## DDN 10 step 2/4/8 sample vs Cohen's
    latex_table_certified_accuracy(
        "analysis/latex/vary_noise_cifar10_DDN_10steps_multinoise", 0.0, 2.25, 0.25, all_cifar_cohen + DDN_10steps + DDN_10_steps_2_samples + DDN_10_steps_4_samples + DDN_10_steps_8_samples)

    #### PGD varying number of samples
    ## PGD 2 step 2/4/8 sample vs Cohen's
    latex_table_certified_accuracy(
        "analysis/latex/vary_noise_cifar10_PGD_2steps_multinoise", 0.0, 2.25, 0.25, all_cifar_cohen + PGD_2steps + PGD_2_steps_2_samples + PGD_2_steps_4_samples + PGD_2_steps_8_samples)

    ## PGD 4 step 2/4/8 sample vs Cohen's
    latex_table_certified_accuracy(
        "analysis/latex/vary_noise_cifar10_PGD_4steps_multinoise", 0.0, 2.25, 0.25, all_cifar_cohen + PGD_4steps + PGD_4_steps_2_samples + PGD_4_steps_4_samples + PGD_4_steps_8_samples)

    ## PGD 10 step 2/4/8 sample vs Cohen's
    latex_table_certified_accuracy(
        "analysis/latex/vary_noise_cifar10_PGD_10steps_multinoise", 0.0, 2.25, 0.25, all_cifar_cohen + PGD_10steps + PGD_10_steps_2_samples + PGD_10_steps_4_samples + PGD_10_steps_8_samples)


################### PLOTS
# Paper plots
    plot_empirical_accuracy_vary_N(
        "analysis/plots/paper_figures/cohen_empirical_vary_N",
        None, 2.25,
        methods_certified=all_cifar_cohen,
        methods_empirical=[
        'data/predict/predict_cifar_cohen/N100',
        'data/predict/predict_cifar_cohen/N1000',
        'data/predict/predict_cifar_cohen/N10000',
        'data/predict/predict_cifar_cohen/N100000',
        ])

    plot_empirical_accuracy_nograd_trick(
        "analysis/plots/paper_figures/cohen_empirical_nograd_trick",
        None, 2.25,
        methods_certified=all_cifar_cohen,
        methods_empirical= 'data/predict/predict_cifar_cohen/N10000',
        methods_empirical_nograd_trick='data/predict/predict_cifar_cohen/N10000_nogradtrick')

    plot_empirical_accuracy_vary_N(
        "analysis/plots/paper_figures/ours_empirical_vary_N",
        None, 2.25,
        methods_certified=all_cifar_experiments,
        methods_empirical=[
        'data/predict/predict_cifar_ours/N100',
        'data/predict/predict_cifar_ours/N1000',
        'data/predict/predict_cifar_ours/N10000',
        'data/predict/predict_cifar_ours/N100000',
        ])

    plot_empirical_accuracy_upper_envelopes_vary_num_samples_during_attack(
        "analysis/plots/paper_figures/cohen_vary_samples_during_attack",
        None, 2.25,
        methods_certified=all_cifar_cohen,
        methods_empirical='data/predict/predict_cifar_cohen/N10000',
        )

    plot_empirical_accuracy_upper_envelopes_vary_num_samples_during_attack(
        "analysis/plots/paper_figures/ours_vary_samples_during_attack",
        None, 2.25,
        methods_certified=all_cifar_experiments,
        methods_empirical='data/predict/predict_cifar_ours/N100000',
        )
    
    plot_certified_accuracy_upper_envelopes(
        "analysis/plots/paper_figures/our_vs_cohen_certified_and_empirical_cifar",
        None, 2.25,
        methods_certified_ours=all_cifar_experiments, 
        methods_empirical_ours= 'data/predict/predict_cifar_ours/N100000',
        methods_certified_cohen=all_cifar_cohen,
        methods_empirical_cohen='data/predict/predict_cifar_cohen/N10000',
        )

    plot_certified_accuracy_upper_envelopes_all_methods(
        "analysis/plots/paper_figures/our_vs_cohen_certified_and_empirical_cifar_all_methods",
        None, 2.25,
        methods_certified_ours=all_cifar_experiments,
        methods_certified_ours_pretrain = PGD_imagenetPretraining_cifar10Fineuning, 
        methods_certified_ours_semisuper = PGD_self_training, 
        methods_certified_ours_pretrain_semisuper = PGD_imagenetPretraining_cifar10Fineuning_self_training, 
        methods_certified_cohen=all_cifar_cohen,
        )

    plot_certified_accuracy_upper_envelopes_vary_eps(
        "analysis/plots/paper_figures/our_vs_cohen_certified_vary_epsilon",
        None, 2.25,
        methods_certified_ours=all_cifar_experiments, 
        methods_empirical_ours= 'data/predict/predict_cifar_ours/N100000',
        methods_certified_cohen=all_cifar_cohen,
        methods_empirical_cohen='data/predict/predict_cifar_cohen/N10000',
        )


    plot_certified_accuracy_upper_envelopes(
        "analysis/plots/paper_figures/our_vs_cohen_certified_imagenet",
        None, 3.5,
        methods_certified_ours=all_imagenet_experiments, 
        methods_certified_cohen=all_imagenet_cohen_replicate,
        )

    plot_certified_accuracy_upper_envelopes_vary_m(
        "analysis/plots/paper_figures/PGD_DDN_bestSteps_vary_samples",
        None, 2.25,
        methods_certified_cohen=all_cifar_cohen, 
        methods_certified_N1=DDN_10steps + DDN_4steps + PGD_2steps + PGD_10steps, 
        methods_certified_N2=DDN_10_steps_2_samples + DDN_4_steps_2_samples + PGD_2_steps_2_samples + PGD_10_steps_2_samples,
        methods_certified_N4=DDN_10_steps_4_samples + DDN_4_steps_4_samples + PGD_2_steps_4_samples + PGD_10_steps_4_samples, 
        methods_certified_N8=DDN_10_steps_8_samples + DDN_4_steps_8_samples + PGD_2_steps_8_samples + PGD_10_steps_8_samples, 
        )

    plot_certified_accuracy_upper_envelopes_base_vs_ours_vs_cohen(
        "analysis/plots/paper_figures/ours_vs_cohen_vs_base",
        None, 2.25,
        methods_certified_cohen=all_cifar_cohen, 
        methods_certified_ours=PGD_with_noise_during_attack,
        methods_certified_base=PGD_no_noise_during_attack_and_training,
        methods_certified_base_with_noise=PGD_no_noise_during_attack,)


    plot_certified_accuracy_per_sigma_against_original(
        "analysis/plots/paper_figures/vary_noise_cifar_ours_vs_cohen", None, 4, 
        methods=all_cifar_experiments,
        methods_original=all_cifar_cohen)


    plot_certified_accuracy_per_sigma_against_original(
        "analysis/plots/paper_figures/vary_noise_imagenet_ours_vs_cohen", None, 4,
        methods=all_imagenet_experiments,
        methods_original=all_imagenet_cohen_replicate)


    plot_certified_accuracy_per_sigma_against_original_one_sample(
        "analysis/plots/paper_figures/vary_noise_cifar_ours_vs_cohen_one_sample", None, 4, 
        methods=[
            Line(ApproximateAccuracy("data/certify/cifar10/PGD_2steps_multiNoiseSamples/8-multitrain/eps_255/cifar10/resnet110/noise_0.12/test/sigma_0.12"), "$\sigma = 0.12$"),
            Line(ApproximateAccuracy("data/certify/cifar10/PGD_2steps_multiNoiseSamples/8-multitrain/eps_255/cifar10/resnet110/noise_0.25/test/sigma_0.25"), "$\sigma = 0.25$"),
            Line(ApproximateAccuracy("data/certify/cifar10/PGD_2steps_multiNoiseSamples/8-multitrain/eps_255/cifar10/resnet110/noise_0.50/test/sigma_0.50"), "$\sigma = 0.50$"),
            Line(ApproximateAccuracy("data/certify/cifar10/PGD_2steps_multiNoiseSamples/8-multitrain/eps_255/cifar10/resnet110/noise_1.00/test/sigma_1.00"), "$\sigma = 1.00$"),
        ],
        methods_original=all_cifar_cohen)


    plot_certified_accuracy_per_sigma_against_original_one_sample(
        "analysis/plots/paper_figures/vary_noise_imagenet_ours_vs_cohen_one_sample", None, 4,
        methods=[
            Line(ApproximateAccuracy("data/certify/imagenet/PGD_1step/imagenet/eps_255/resnet50/noise_0.25/test/sigma_0.25"), "$\sigma = 0.25$", plot_fmt='b'),
            Line(ApproximateAccuracy("data/certify/imagenet/PGD_1step/imagenet/eps_255/resnet50/noise_0.50/test/sigma_0.50"), "$\sigma = 0.50$", plot_fmt='orange'),
            Line(ApproximateAccuracy("data/certify/imagenet/PGD_1step/imagenet/eps_255/resnet50/noise_1.00/test/sigma_1.00"), "$\sigma = 1.00$", plot_fmt='g'),
        ],
        methods_original=all_imagenet_cohen_replicate)


    plot_certified_accuracy(
        "analysis/plots/paper_figures/github_readme_certified", "CIFAR-10, vary $\sigma$", 1.5, [
            Line(ApproximateAccuracy("data/certify/cifar10/finetune_cifar_from_imagenetPGD2steps/PGD_10steps_30epochs_multinoise/2-multitrain/eps_64/cifar10/resnet110/noise_0.12/test/sigma_0.12"), "$\sigma = 0.12$"),
            Line(ApproximateAccuracy("data/certify/cifar10/finetune_cifar_from_imagenetPGD2steps/PGD_10steps_30epochs_multinoise/2-multitrain/eps_64/cifar10/resnet110/noise_0.25/test/sigma_0.25"), "$\sigma = 0.25$"),
            Line(ApproximateAccuracy("data/certify/cifar10/finetune_cifar_from_imagenetPGD2steps/PGD_10steps_30epochs_multinoise/2-multitrain/eps_64/cifar10/resnet110/noise_0.50/test/sigma_0.50"), "$\sigma = 0.50$"),
            Line(ApproximateAccuracy("data/certify/cifar10/finetune_cifar_from_imagenetPGD2steps/PGD_10steps_30epochs_multinoise/2-multitrain/eps_64/cifar10/resnet110/noise_1.00/test/sigma_1.00"), "$\sigma = 1.00$"),
        ])



#################### Generate Cohen et al. results
    latex_table_certified_accuracy(
        "analysis/latex/vary_noise_cifar10", 0.25, 1.5, 0.25, [
            Line(ApproximateAccuracy("data_cohen/certify/cifar10/resnet110/noise_0.12/test/sigma_0.12"), "$\sigma = 0.12$"),
            Line(ApproximateAccuracy("data_cohen/certify/cifar10/resnet110/noise_0.25/test/sigma_0.25"), "$\sigma = 0.25$"),
            Line(ApproximateAccuracy("data_cohen/certify/cifar10/resnet110/noise_0.50/test/sigma_0.50"), "$\sigma = 0.50$"),
            Line(ApproximateAccuracy("data_cohen/certify/cifar10/resnet110/noise_1.00/test/sigma_1.00"), "$\sigma = 1.00$"),
        ])

    markdown_table_certified_accuracy(
        "analysis/markdown/vary_noise_cifar10", 0.25, 1.5, 0.25, [
            Line(ApproximateAccuracy("data_cohen/certify/cifar10/resnet110/noise_0.12/test/sigma_0.12"), "&sigma; = 0.12"),
            Line(ApproximateAccuracy("data_cohen/certify/cifar10/resnet110/noise_0.25/test/sigma_0.25"), "&sigma; = 0.25"),
            Line(ApproximateAccuracy("data_cohen/certify/cifar10/resnet110/noise_0.50/test/sigma_0.50"), "&sigma; = 0.50"),
            Line(ApproximateAccuracy("data_cohen/certify/cifar10/resnet110/noise_1.00/test/sigma_1.00"), "&sigma; = 1.00"),
        ])

    latex_table_certified_accuracy(
        "analysis/latex/vary_noise_imagenet", 0.5, 3.0, 0.5, [
            Line(ApproximateAccuracy("data_cohen/certify/imagenet/resnet50/noise_0.25/test/sigma_0.25"), "$\sigma = 0.25$"),
            Line(ApproximateAccuracy("data_cohen/certify/imagenet/resnet50/noise_0.50/test/sigma_0.50"), "$\sigma = 0.50$"),
            Line(ApproximateAccuracy("data_cohen/certify/imagenet/resnet50/noise_1.00/test/sigma_1.00"), "$\sigma = 1.00$"),
        ])

    markdown_table_certified_accuracy(
        "analysis/markdown/vary_noise_imagenet", 0.5, 3.0, 0.5, [
            Line(ApproximateAccuracy("data_cohen/certify/imagenet/resnet50/noise_0.25/test/sigma_0.25"), "&sigma; = 0.25"),
            Line(ApproximateAccuracy("data_cohen/certify/imagenet/resnet50/noise_0.50/test/sigma_0.50"), "&sigma; = 0.50"),
            Line(ApproximateAccuracy("data_cohen/certify/imagenet/resnet50/noise_1.00/test/sigma_1.00"), "&sigma; = 1.00"),
        ])

    plot_certified_accuracy(
        "analysis/plots/cohen_figures/vary_noise_cifar10", "CIFAR-10, vary $\sigma$", 1.5, [
            Line(ApproximateAccuracy("data_cohen/certify/cifar10/resnet110/noise_0.12/test/sigma_0.12"), "$\sigma = 0.12$"),
            Line(ApproximateAccuracy("data_cohen/certify/cifar10/resnet110/noise_0.25/test/sigma_0.25"), "$\sigma = 0.25$"),
            Line(ApproximateAccuracy("data_cohen/certify/cifar10/resnet110/noise_0.50/test/sigma_0.50"), "$\sigma = 0.50$"),
            Line(ApproximateAccuracy("data_cohen/certify/cifar10/resnet110/noise_1.00/test/sigma_1.00"), "$\sigma = 1.00$"),
        ])

    plot_certified_accuracy(
        "analysis/plots/cohen_figures/vary_train_noise_cifar_050", "CIFAR-10, vary train noise, $\sigma=0.5$", 1.5, [
            Line(ApproximateAccuracy("data_cohen/certify/cifar10/resnet110/noise_0.25/test/sigma_0.50"), "train $\sigma = 0.25$"),
            Line(ApproximateAccuracy("data_cohen/certify/cifar10/resnet110/noise_0.50/test/sigma_0.50"), "train $\sigma = 0.50$"),
            Line(ApproximateAccuracy("data_cohen/certify/cifar10/resnet110/noise_1.00/test/sigma_0.50"), "train $\sigma = 1.00$"),
        ])

    plot_certified_accuracy(
        "analysis/plots/cohen_figures/vary_train_noise_imagenet_050", "ImageNet, vary train noise, $\sigma=0.5$", 1.5, [
            Line(ApproximateAccuracy("data_cohen/certify/imagenet/resnet50/noise_0.25/test/sigma_0.50"), "train $\sigma = 0.25$"),
            Line(ApproximateAccuracy("data_cohen/certify/imagenet/resnet50/noise_0.50/test/sigma_0.50"), "train $\sigma = 0.50$"),
            Line(ApproximateAccuracy("data_cohen/certify/imagenet/resnet50/noise_1.00/test/sigma_0.50"), "train $\sigma = 1.00$"),
        ])

    plot_certified_accuracy(
        "analysis/plots/cohen_figures/vary_noise_imagenet", "ImageNet, vary $\sigma$", 4, [
            Line(ApproximateAccuracy("data_cohen/certify/imagenet/resnet50/noise_0.25/test/sigma_0.25"), "$\sigma = 0.25$"),
            Line(ApproximateAccuracy("data_cohen/certify/imagenet/resnet50/noise_0.50/test/sigma_0.50"), "$\sigma = 0.50$"),
            Line(ApproximateAccuracy("data_cohen/certify/imagenet/resnet50/noise_1.00/test/sigma_1.00"), "$\sigma = 1.00$"),
        ])

    plot_certified_accuracy(
        "analysis/plots/cohen_figures/high_prob", "Approximate vs. High-Probability", 2.0, [
            Line(ApproximateAccuracy("data_cohen/certify/imagenet/resnet50/noise_0.50/test/sigma_0.50"), "Approximate"),
            Line(HighProbAccuracy("data_cohen/certify/imagenet/resnet50/noise_0.50/test/sigma_0.50", 0.001, 0.001), "High-Prob"),
        ])