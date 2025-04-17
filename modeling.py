6
import numpy as np
import pandas as pd
import math
from scipy.optimize import minimize
import scipy.special as sps
from scipy.special import kv, gamma, factorial
from functools import lru_cache
import matplotlib.pyplot as plt
from scipy.integrate import quad
import seaborn as sns
from matplotlib.ticker import ScalarFormatter
import warnings
import itertools
from scipy.optimize import basinhopping

warnings.filterwarnings("ignore")
epsilon = 1e-40


def add_pops_tolist(dct, cip):
    concentration = 0
    while concentration < 7:
        concentration_float = concentration / 10.0
        dct[concentration_float] = cip[cip['treatment'] == concentration_float]
        concentration += 2


def organize_data_2(data):
    grouped = data.groupby('geneID').agg(
        length=('gene_length', 'first'),
        syn=('snp_type', lambda x: (x == 'synonymous').sum()),
        nonsyn=('snp_type', lambda x: (x != 'synonymous').sum())
    ).reset_index()
    return grouped


def summary_tables_syn(data, data_length):
    data['geneID'] = pd.Categorical(data['geneID'])
    syn_counts = data.groupby('geneID', observed=False)['syn'].sum().reset_index()
    length_df = data[['geneID', 'length']].drop_duplicates()
    syn_counts = pd.merge(syn_counts, length_df, on='geneID', how='left')
    syn_counts = pd.merge(data_length, syn_counts, on='geneID', how='left')
    syn_counts['syn'] = syn_counts['syn'].fillna(0)
    syn_counts['length'] = syn_counts['length'].fillna(syn_counts['gene_length'])
    syn_summary = syn_counts.groupby('syn', observed=False).size().reset_index(name='Count')
    return syn_counts, syn_summary


def summary_tables_non(data, data_length):
    data['geneID'] = pd.Categorical(data['geneID'])
    non_counts = data.groupby('geneID', observed=False)['nonsyn'].sum().reset_index()
    length_df = data[['geneID', 'length']].drop_duplicates()
    non_counts = pd.merge(non_counts, length_df, on='geneID', how='left')
    non_counts = pd.merge(data_length, non_counts, on='geneID', how='left')
    non_counts['nonsyn'] = non_counts['nonsyn'].fillna(0)
    non_counts['length'] = non_counts['length'].fillna(non_counts['gene_length'])
    non_summary = non_counts.groupby('nonsyn', observed=False).size().reset_index(name='Count')
    return non_counts, non_summary


def gamma_poison_mixture(r, k, p):
    return (gamma(r + k) / (math.factorial(int(k)) * gamma(r))) * ((1 - p) ** int(k)) * (p ** r)


def likelihood_func(params, k_values):
    r, p = params
    if r <= 0 or p <= 0 or p >= 1:
        return np.inf
    log_likelihood = np.sum([np.log(gamma_poison_mixture(r, k, p)) for k in k_values])
    return -log_likelihood


def fit_synonymous(data_syn):
    initial_guess = [1.0, 0.5]
    res = minimize(lambda params: likelihood_func(params, data_syn),
                   initial_guess,
                   bounds=[(1e-5, None), (1e-5, 1)],
                   method='L-BFGS-B')
    r_est, p_est = res.x
    return r_est, p_est


def integrand(t_range, c1, c2, v):
    return (t_range ** c1) * np.exp(-c2 * (t_range ** 2)) * kv(v, t_range)


def bessel_integral(c1, c2, v):
    val, err = quad(integrand, 0.0, np.inf, args=(c1, c2, v), limit=20000)
    return val


@lru_cache(maxsize=None)
def prob_poisson_mixed(k, a, b, v):
    if (a + v) / 2 <= 0 or (a - v) / 2 <= 0:
        return epsilon

    try:
        c1 = 2 * k + a - 1
        c2 = 1 / (4 * (b ** 2))

        if c1 > 50 or b > 50:
            max_t = 100
            val, _ = quad(integrand, 0.0, max_t, args=(c1, c2, v), limit=20000)
            integral = val
        else:
            integral = bessel_integral(c1, c2, v)

        log_gamma_term = np.log(sps.gamma((a + v) / 2)) + np.log(sps.gamma((a - v) / 2))
        log_coeff = np.log(4) - ((2 * k + a) * np.log(2)) - np.log(sps.factorial(k)) - (
                    2 * k * np.log(b)) - log_gamma_term

        val = np.exp(log_coeff) * integral

        if (not np.isfinite(val)) or (val <= 0) or (val >= 1):
            return epsilon

        return val
    except Exception:
        return epsilon


def negloglik_nonsyn_scaled(params, k_values, alpha_mu):
    # scale_factor = dynamic_scaling(alpha_mu)
    # alpha_mu = scale_factor*alpha_mu
    # alpha_s = scale_factor*alpha_s
    alpha_s, C_ns = params

    a = alpha_mu + alpha_s
    v = alpha_mu - alpha_s
    b = math.sqrt(alpha_mu * alpha_s / C_ns)

    nll = 0.0
    for k in k_values:
        prob = prob_poisson_mixed(int(k), float(a), float(b), float(v))
        nll -= math.log(prob + epsilon)
        # print(-nll)
    return nll


def fit_nonsynonymous_scaled(data_nonsyn, alpha_mu):
    bounds = [(1e-5, 50), (1e-5, 50)]  # [(alpha_s_min, alpha_s_max), (C_ns_min, C_ns_max)]

    def bounded_negloglik(params):
        alpha_s, C_ns = params
        if not (bounds[0][0] <= alpha_s <= bounds[0][1] and
                bounds[1][0] <= C_ns <= bounds[1][1]):
            return np.inf

        try:
            return negloglik_nonsyn_scaled([alpha_s, C_ns], data_nonsyn, alpha_mu)
        except Exception:
            return np.inf

    initial_guesses = [
        [0.01, 0.01],
        [1.0, 1.0],
        [10.0, 10.0]
    ]

    best_result = None
    best_nll = np.inf

    for initial_guess in initial_guesses:
        try:
            minimizer_kwargs = {
                "method": "L-BFGS-B",
                "bounds": bounds
            }

            result = basinhopping(
                bounded_negloglik,
                initial_guess,
                niter=1,
                T=1.0,
                stepsize=0.5,
                minimizer_kwargs=minimizer_kwargs,
                interval=10
            )

            if result.fun < best_nll and np.isfinite(result.fun):
                best_nll = result.fun
                best_result = result

        except Exception as e:
            print(f"Optimization attempt failed: {e}")
            continue

    if best_result is None or not np.isfinite(best_nll):
        raise ValueError("Basin-hopping optimization failed to find valid parameters")

    alpha_s_opt, C_ns_opt = best_result.x

    a_opt = alpha_mu + alpha_s_opt
    v_opt = alpha_mu - alpha_s_opt
    b_opt = math.sqrt(alpha_mu * alpha_s_opt / C_ns_opt)

    plot_fit(data_nonsyn, a_opt, b_opt, v_opt)
    return alpha_s_opt, C_ns_opt, a_opt, b_opt, v_opt, best_nll


def plot_fit(data_nonsyn, a, b, v):
    sns.histplot(data_nonsyn, discrete=True, stat='probability',
                 color='#00008B', alpha=0.6, label='Data')
    k_min, k_max = int(np.min(data_nonsyn)), int(np.max(data_nonsyn))
    k_vals = np.arange(k_min, k_max + 1)
    p_vals = [prob_poisson_mixed(k, a, b, v) for k in k_vals]
    plt.plot(k_vals, p_vals, 'o-', color='#301934',
             label=f'Model Fit (a={a:.3g}, b={b:.3g}, v={v:.3g})')
    plt.yscale('log')
    plt.gca().yaxis.set_major_formatter(ScalarFormatter())
    plt.xlabel("Mutation Count")
    plt.ylabel("Probability")
    plt.legend()
    plt.title("Non-synonymous Mutation Count Fit")
    plt.show()


def deal_with_Syn(data, data_length):
    plus_syn = data[(data['cip'] == 'Plus') & (data['snp_type'] == 'synonymous')]
    minus_syn = data[(data['cip'] == 'Minus') & (data['snp_type'] == 'synonymous')]

    plus_pops_syn = {}
    minus_pops_syn = {}

    add_pops_tolist(plus_pops_syn, plus_syn)
    add_pops_tolist(minus_pops_syn, minus_syn)

    counter = 0

    syn_counts_minus = {}
    syn_counts_plus = {}

    for key, value in itertools.chain(plus_pops_syn.items(), minus_pops_syn.items()):
        dataframe = organize_data_2(value)

        syn_counts, syn_summary = summary_tables_syn(dataframe, data_length)
        if counter < 4:
            syn_counts_plus[key] = syn_counts

        else:
            syn_counts_minus[key] = syn_counts
        counter += 1
    return syn_counts_plus, syn_counts_minus


def deal_with_non(data, data_length):
    plus = data[(data['cip'] == 'Plus') & (data['snp_type'] != 'synonymous')]
    minus = data[(data['cip'] == 'Minus') & (data['snp_type'] != 'synonymous')]

    plus_pops_non = {}
    minus_pops_non = {}

    add_pops_tolist(plus_pops_non, plus)
    add_pops_tolist(minus_pops_non, minus)

    counter = 0
    non_counts_minus = {}
    non_counts_plus = {}

    for key, value in itertools.chain(plus_pops_non.items(), minus_pops_non.items()):
        dataframe = organize_data_2(value)

        non_counts, non_summary = summary_tables_non(dataframe, data_length)
        if counter < 4:
            non_counts_plus[key] = non_counts

        else:
            non_counts_minus[key] = non_counts
        counter += 1
    return non_counts_plus, non_counts_minus


def model_4_double_gamma(data, data_length):
    syn_counts_plus, syn_counts_minus = deal_with_Syn(data, data_length)
    non_counts_plus, non_counts_minus = deal_with_non(data, data_length)
    results = []

    def process_combination(syn_counts, non_counts, treatment):
        for concentration in syn_counts.keys():
            print(concentration)
            syn_counts_df = syn_counts[concentration].iloc[:, 2].values.astype(float)
            r_est, p_est = fit_synonymous(syn_counts_df)
            alpha_mu = r_est
            mean_syn = r_est * (1 - p_est) / p_est
            var_syn = r_est * (1 - p_est) / (p_est ** 2)
            mean_syn_actual = np.mean(syn_counts_df)

            try:
                non_counts_df = non_counts[concentration]
                data_nonsyn = non_counts_df.iloc[:, 2].values.astype(float)
                alpha_s_opt, C_ns_opt, a_opt, b_opt, v_opt, final_nll = fit_nonsynonymous_scaled(data_nonsyn, alpha_mu)
                b_x = alpha_s_opt / C_ns_opt
                b_y = alpha_s_opt
                data_mean = np.mean(data_nonsyn)
                data_var = np.var(data_nonsyn)
                X_mean = C_ns_opt
                X_var = C_ns_opt ** 2 / alpha_mu
                Y_mean = 1.0
                Y_var = 1 / alpha_s_opt
                Model_Mean = X_mean * Y_mean
                Model_mean_2 = ((a_opt + v_opt) * (a_opt - v_opt)) / (4 * b_opt ** 2)
                Model_Var = Model_Mean + X_mean ** 2 * (1 / alpha_s_opt + 1 / alpha_mu + 1 / (alpha_mu * alpha_s_opt))
                results.append({
                    'Treatment': treatment,
                    'Concentration': concentration,
                    'alpha_mu': alpha_mu,
                    'Mean_syn': mean_syn,
                    'mean_actual': mean_syn_actual,
                    'Var_syn': var_syn,
                    'a_y': alpha_s_opt,
                    'b_y': b_y,
                    'a': a_opt,
                    'b': b_opt,
                    'v': v_opt,
                    'Data_Mean_nonsyn': data_mean,
                    'C_ns': C_ns_opt,
                    'model_mean': Model_mean_2,
                    'Data_Var_nonsyn': data_var,
                    'Model_Var': Model_Var,
                    'X_Mean': X_mean,
                    'X_Var': X_var,
                    'Y_Mean': Y_mean,
                    'Final_NLL': final_nll,
                    'Y_Var': Y_var
                })
            except Exception as e:
                print(f"Non-synonymous optimization failed for concentration {concentration} ({treatment}): {e}")
                continue

    process_combination(syn_counts_plus, non_counts_plus, 'Plus')
    process_combination(syn_counts_minus, non_counts_minus, 'Minus')
    results_df = pd.DataFrame(results)
    results_df.to_csv('model_4_double_gamma_results_grouped.csv', index=False)
    return results_df


def main():
    data = pd.read_csv('all_evol_mut_uni - all_evol_mut_uni (1).csv')
    data_length = pd.read_csv('gene_list_file.csv')
    data_length = data_length[['gene_id', 'gene_length']].rename(columns={'gene_id': 'geneID'})
    results_df = model_4_double_gamma(data, data_length)
    return results_df


if __name__ == '__main__':
    results = main()
