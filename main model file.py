import numpy as np
import pandas as pd
import math
import itertools
from scipy.optimize import minimize, basinhopping
import scipy.special as sps
from scipy.special import kv, gamma, factorial
from functools import lru_cache
import matplotlib.pyplot as plt
from scipy.integrate import quad
import seaborn as sns
from matplotlib.ticker import ScalarFormatter
import warnings

warnings.filterwarnings("ignore")


##############################
# Data Organization Functions
##############################
def add_pops_tolist(dct, cip):
    # Group the data in 'cip' by concentration.
    concentration = 0
    while concentration < 7:
        concentration_float = concentration / 10.0
        dct[concentration_float] = cip[cip['treatment'] == concentration_float]
        concentration += 2


def organize_data_2(data):
    genes_seen = {}
    unique_genes = {"geneID": [], "length": [], "syn": [], "nonsyn": [], "pops": []}
    num_genes = 0
    for index, row in data.iterrows():
        if row['geneID'] in genes_seen:
            location = genes_seen[row['geneID']]
            if row['snp_type'] == 'synonymous':
                unique_genes['syn'][location] += 1
            else:
                unique_genes['nonsyn'][location] += 1
            unique_genes['pops'][location].append(row['Well'])
        else:
            genes_seen[row['geneID']] = num_genes
            unique_genes['geneID'].append(row['geneID'])
            unique_genes['length'].append(row['gene_length'])
            unique_genes['pops'].append([row['Well']])
            if row['snp_type'] == 'synonymous':
                unique_genes['syn'].append(1)
                unique_genes['nonsyn'].append(0)
            else:
                unique_genes['nonsyn'].append(1)
                unique_genes['syn'].append(0)
            num_genes += 1
    return pd.DataFrame(unique_genes)


def summary_tables_non(data, data_length):
    data['geneID'] = pd.Categorical(data['geneID'])
    non_counts = data.groupby(['geneID'], observed=False)['nonsyn'].sum().reset_index()
    length_df = data[['geneID', 'length']].drop_duplicates()
    non_counts = pd.merge(non_counts, length_df, on='geneID', how='left')
    non_counts = pd.merge(data_length, non_counts, on='geneID', how='left')
    non_counts['nonsyn'] = non_counts['nonsyn'].fillna(0)
    non_counts['length'] = non_counts['length'].fillna(non_counts['gene_length'])
    non_counts['Non_Categories'] = pd.cut(non_counts['nonsyn'],
                                          bins=[-np.inf, 0, 1, 2, np.inf],
                                          labels=['0', '1', '2', '>2'])
    non_summary = non_counts.groupby('Non_Categories', observed=False).size().reset_index(name='Count')
    return non_counts, non_summary


def summary_tables_syn(data, data_length):
    data['geneID'] = pd.Categorical(data['geneID'])
    syn_counts = data.groupby(['geneID'], observed=False)['syn'].sum().reset_index()
    length_df = data[['geneID', 'length']].drop_duplicates()
    syn_counts = pd.merge(syn_counts, length_df, on='geneID', how='left')
    syn_counts = pd.merge(data_length, syn_counts, on='geneID', how='left')
    syn_counts['syn'] = syn_counts['syn'].fillna(0)
    syn_counts['length'] = syn_counts['length'].fillna(syn_counts['gene_length'])
    syn_counts['Syn_Categories'] = pd.cut(syn_counts['syn'],
                                          bins=[-np.inf, 0, 1, np.inf],
                                          labels=['0', '1', '>1'])
    syn_summary = syn_counts.groupby('Syn_Categories', observed=False).size().reset_index(name='Count')
    return syn_counts, syn_summary


##############################
# Model Functions
##############################
def model_1_poison_fixed(data, typeofmut='syn', sims=1000):
    mut_column = data.iloc[:, 2]
    lambda_val = mut_column.mean()
    num_genes = len(mut_column)

    synthetic_counts = np.random.poisson(lam=lambda_val, size=num_genes * sims)
    syn_pois_counts = np.bincount(synthetic_counts)
    syn_pois_freq = syn_pois_counts / sims

    if typeofmut == 'syn':
        # 3 categories: 0, 1, >1
        result = syn_pois_freq[:2].tolist()
        result.append(syn_pois_freq[2:].sum())
    else:
        # 4 categories: 0, 1, 2, >2
        result = syn_pois_freq[:3].tolist()
        result.append(syn_pois_freq[3:].sum())

    return result


def gamma_poison_mixture(r, k, p):
    return (gamma(r + k) / (math.factorial(int(k)) * gamma(r))) * ((1 - p) ** k) * (p ** r)


def likelihood_func(params, k_values):
    r, p = params
    if r <= 0 or p <= 0 or p >= 1:
        return np.inf
    log_likelihood = np.sum([np.log(gamma_poison_mixture(r, k, p)) for k in k_values])
    return -log_likelihood


def optimize(data):
    k_values = data.iloc[:, 2].values
    initial_guess = [1, 0.5]
    result = minimize(likelihood_func, initial_guess, args=(k_values,),
                      method='L-BFGS-B', bounds=[(1e-5, None), (1e-5, 1)])
    estimated_r, estimated_p = result.x
    return estimated_r, estimated_p


epsilon = 1e-40


def integrand(t_range, c1, c2, v):
    return (t_range ** c1) * np.exp(-c2 * (t_range ** 2)) * kv(v, t_range)


def bessel_integral(c1, c2, v):
    val, err = quad(integrand, 0.0, np.inf, args=(c1, c2, v), limit=200)
    return val


@lru_cache(maxsize=None)
def prob_poisson_mixed(k, a, b, v):
    if (a + v) / 2 <= 0 or (a - v) / 2 <= 0:
        return epsilon
    c1 = 2 * k + a - 1
    c2 = 1 / (4 * (b ** 2))
    integral = bessel_integral(c1, c2, v)
    gamma_term = sps.gamma((a + v) / 2) * sps.gamma((a - v) / 2)
    coeff = 4 / ((2 ** (2 * k + a)) * sps.factorial(k) * (b ** (2 * k)) * gamma_term)
    val = coeff * integral
    if (not np.isfinite(val)) or (val <= 0) or (val >= 1):
        val = epsilon
    return val


def log_likelihood(params, k_observed, a_x):
    a, b = params
    v = 2 * a_x - a
    prob_poisson_mixed_fun = lambda k: prob_poisson_mixed(k, a=a, b=b, v=v)
    prob_poisson_mixed_observed = np.vectorize(prob_poisson_mixed_fun)(k_observed)
    likelihood = -(np.sum(np.log(prob_poisson_mixed_observed + epsilon)))
    if not np.isfinite(likelihood):
        return np.inf
    print(likelihood)
    return likelihood if (likelihood >= 0) else np.inf


def optimize_2(data, a_x):
    k_values = data.iloc[:, 2].values.astype(float)
    a_init = 0.5
    b_init = 1.0
    initial_guess = [a_init, b_init]
    bounds = [(1e-3, 40), (1e-3, 40)]
    result = basinhopping(
        log_likelihood,
        x0=initial_guess,
        minimizer_kwargs={"method": "SLSQP", "args": (k_values, a_x), "bounds": bounds},
        niter=100,
        stepsize=0.5
    )
    a_est, b_est = result.x
    final_result = basinhopping(
        log_likelihood,
        x0=[a_est, b_est],
        minimizer_kwargs={"method": "SLSQP", "args": (k_values, a_x), "bounds": bounds},
        niter=100,
        stepsize=0.5
    )
    estimated_a, estimated_b = final_result.x
    estimated_v = 2 * a_x - estimated_a
    return estimated_a, estimated_b, estimated_v


def continous_sum(a, b, v):
    sum_lst = []
    cur_sum = 0
    for i in range(74):
        cur_val = prob_poisson_mixed(i, a, b, v)
        if cur_val < epsilon:
            break
        cur_sum += cur_val
        sum_lst.append(cur_sum)
    return sum_lst


def get_ks_optimized(sum_of_ks, size):
    random_values = np.random.uniform(0, 1, size)
    k_vec = np.searchsorted(sum_of_ks, random_values)
    return k_vec


def model_3_gamma_heterogeneity(data, typeofmut, sims=1000):
    r, p = optimize(data)
    size = int(len(data[typeofmut]) * sims)
    nb_samples = np.random.negative_binomial(r, p, size=size)
    nb_est = np.bincount(nb_samples) / sims
    if typeofmut == 'syn':
        nb_summary = np.concatenate((nb_est[:2], np.array([nb_est[2:].sum()])))
    else:
        nb_summary = np.concatenate((nb_est[:3], np.array([nb_est[3:].sum()])))
    if len(nb_summary.tolist()) < 3 and typeofmut == 'syn':
        return [0, 0, 0]
    elif len(nb_summary.tolist()) < 3 and typeofmut == 'nonsyn':
        return [0, 0, 0, 0]
    else:
        return nb_summary.tolist()


def model_4_mu_model(data_comp, data_length, non_counts, non_summary, typeofmut, antibiotics, sims=1000):
    # Use synonymous data to choose the rate parameter based on treatment.
    syn_counts_plus, syn_counts_minus = deal_with_Syn(data_comp, data_length)
    if antibiotics == 'plus':
        r, p = optimize(syn_counts_plus)
    else:
        r, p = optimize(syn_counts_minus)
    a, b, v = optimize_2(non_counts, r)
    size = int(len(non_counts[typeofmut]) * sims)
    nb_samples_probs = continous_sum(a, b, v)
    samples = get_ks_optimized(nb_samples_probs, size)
    nb_est = np.bincount(samples) / sims
    if typeofmut == 'syn':
        nb_summary = np.concatenate((nb_est[:2], np.array([nb_est[2:].sum()])))
    else:
        nb_summary = np.concatenate((nb_est[:3], np.array([nb_est[3:].sum()])))
    if len(nb_summary.tolist()) < 3 and typeofmut == 'syn':
        return [0, 0, 0]
    elif len(nb_summary.tolist()) < 3 and typeofmut == 'nonsyn':
        return [0, 0, 0, 0]
    else:
        return nb_summary.tolist()



def plot_syn_ax(ax, syn_sum, poison_vals_syn1, poison_vals_syn2, title):
    custom_scale = [0.1, 1, 10, 100, 1000, 10000]
    sns.barplot(x='Syn_Categories', y='Count', data=syn_sum,
                palette='viridis', edgecolor=None, ax=ax)
    ax.set_xlabel('No of synonymous mutations per gene', color='#3D3934')
    ax.set_ylabel('Count', color='#3D3934')
    ax.set_yscale('log')
    ax.set_yticks(custom_scale)
    ax.set_ylim([0.1, 10000])
    ax.yaxis.set_major_formatter(ScalarFormatter())
    ax.plot(syn_sum.index, poison_vals_syn1, color='blue', marker='o', markersize=10,
            linestyle='-', alpha=0.7, label='Fix rate model', linewidth=4)
    ax.plot(syn_sum.index, poison_vals_syn2, color='red', marker='s', markersize=10,
            linestyle='--', alpha=0.8, label='Neg bin model', linewidth=4)
    ax.legend(loc='upper right', fontsize='small')
    ax.set_title(title, color='#3D3934')


def plot_non_ax(ax, non_sum, poisson_1, poison_vals_non3, poison_vals_non4, title):
    custom_scale = [0.1, 1, 10, 100, 1000, 10000]
    sns.barplot(x='Non_Categories', y='Count', data=non_sum,
                palette='coolwarm', edgecolor=None, ax=ax)
    ax.set_xlabel('No of Non-Synonymous per gene', color='#3D3934')
    ax.set_ylabel('Count', color='#3D3934')
    ax.set_yscale('log')
    ax.set_yticks(custom_scale)
    ax.set_ylim([0.1, 10000])
    ax.yaxis.set_major_formatter(ScalarFormatter())
    ax.plot(non_sum.index, poisson_1, color='blue', marker='s', linestyle='-',
            alpha=0.7, markersize=7, label='Fixed rate model', linewidth=4)
    ax.plot(non_sum.index, poison_vals_non3, color='red', marker='o', linestyle='-',
            alpha=0.8, markersize=7, label='Neg bin model', linewidth=4)
    ax.plot(non_sum.index, poison_vals_non4, color='green', marker='^', linestyle='-',
            alpha=0.6, markersize=7, label='Double gamma model', linewidth=4)
    ax.legend(loc='upper right', fontsize='small')
    ax.set_title(title, color='#3D3934')



def separate_syn_plots(data, data_length):
    # Subset synonymous data by treatment.
    plus_syn = data[(data['cip'] == 'Plus') & (data['snp_type'] == 'synonymous')]
    minus_syn = data[(data['cip'] == 'Minus') & (data['snp_type'] == 'synonymous')]
    plus_syn = organize_data_2(plus_syn)
    minus_syn = organize_data_2(minus_syn)
    syn_counts_plus, syn_summary_plus = summary_tables_syn(plus_syn, data_length)
    syn_counts_minus, syn_summary_minus = summary_tables_syn(minus_syn, data_length)

    # Plot for Plus treatment:
    fig = plt.figure(figsize=(10, 6))
    poison_vals_syn1_plus = model_1_poison_fixed(syn_counts_plus, typeofmut='syn')
    poison_vals_syn2_plus = model_3_gamma_heterogeneity(syn_counts_plus, 'syn')
    ax = fig.add_subplot(1, 1, 1)
    plot_syn_ax(ax, syn_summary_plus, poison_vals_syn1_plus, poison_vals_syn2_plus, "Synonymous: Plus")
    fig.suptitle("Synonymous Mutation Model: Plus Treatment", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

    # Plot for Minus treatment:
    fig = plt.figure(figsize=(10, 6))
    poison_vals_syn1_minus = model_1_poison_fixed(syn_counts_minus, typeofmut='syn')
    poison_vals_syn2_minus = model_3_gamma_heterogeneity(syn_counts_minus, 'syn')
    ax = fig.add_subplot(1, 1, 1)
    plot_syn_ax(ax, syn_summary_minus, poison_vals_syn1_minus, poison_vals_syn2_minus, "Synonymous: Minus")
    fig.suptitle("Synonymous Mutation Model: Minus Treatment", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


def separate_non_plots(data, data_length):
    # group non-synonymous data by treatment and concentration.
    plus = data[(data['cip'] == 'Plus') & (data['snp_type'] != 'synonymous')]
    minus = data[(data['cip'] == 'Minus') & (data['snp_type'] != 'synonymous')]
    plus_pops_non = {}
    minus_pops_non = {}
    add_pops_tolist(plus_pops_non, plus)
    add_pops_tolist(minus_pops_non, minus)

    # For Plus treatments:
    for conc, subset_df in sorted(plus_pops_non.items(), key=lambda x: x[0]):
        fig = plt.figure(figsize=(10, 6))
        dataframe = organize_data_2(subset_df)
        non_counts, non_summary = summary_tables_non(dataframe, data_length)
        poisson_1 = model_1_poison_fixed(non_counts, typeofmut='nonsyn')
        poison_vals_non3 = model_3_gamma_heterogeneity(non_counts, 'nonsyn')
        poison_vals_non4 = model_4_mu_model(data, data_length, non_counts, non_summary, 'nonsyn', 'plus')
        title = f"Conc: {conc} | Plus"
        ax = fig.add_subplot(1, 1, 1)
        plot_non_ax(ax, non_summary, poisson_1, poison_vals_non3, poison_vals_non4, title)
        fig.suptitle("Non-Synonymous Mutation Model - Plus Treatment", fontsize=16)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()

    # For Minus treatments:
    for conc, subset_df in sorted(minus_pops_non.items(), key=lambda x: x[0]):
        fig = plt.figure(figsize=(10, 6))
        dataframe = organize_data_2(subset_df)
        non_counts, non_summary = summary_tables_non(dataframe, data_length)
        poisson_1 = model_1_poison_fixed(non_counts, typeofmut='nonsyn')
        poison_vals_non3 = model_3_gamma_heterogeneity(non_counts, 'nonsyn')
        poison_vals_non4 = model_4_mu_model(data, data_length, non_counts, non_summary, 'nonsyn', 'minus')
        title = f"Conc: {conc} | Minus"
        ax = fig.add_subplot(1, 1, 1)
        plot_non_ax(ax, non_summary, poisson_1, poison_vals_non3, poison_vals_non4, title)
        fig.suptitle("Non-Synonymous Mutation Model - Minus Treatment", fontsize=16)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()



def deal_with_Syn(data, data_length):
    # Organize synonymous data by treatment.
    plus_syn = data[(data['cip'] == 'Plus') & (data['snp_type'] == 'synonymous')]
    minus_syn = data[(data['cip'] == 'Minus') & (data['snp_type'] == 'synonymous')]
    plus_syn = organize_data_2(plus_syn)
    minus_syn = organize_data_2(minus_syn)
    syn_counts_plus, _ = summary_tables_syn(plus_syn, data_length)
    syn_counts_minus, _ = summary_tables_syn(minus_syn, data_length)
    return syn_counts_plus, syn_counts_minus



def main():
    data = pd.read_csv('all_evol_mut_uni - all_evol_mut_uni (1).csv')
    data_length = pd.read_csv('gene_list_file.csv')
    data_length = data_length[['gene_id', 'gene_length']]
    data_length = data_length.rename(columns={'gene_id': 'geneID'})

    separate_syn_plots(data, data_length)

    separate_non_plots(data, data_length)


if __name__ == '__main__':
    main()
