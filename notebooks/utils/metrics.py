from __future__ import (
    division, print_function, unicode_literals, absolute_import
    )
import numpy as np
from scipy.special import gammaln
from scipy.integrate import quad
import pandas as pd

def get_purity(y_true, y_pred, threshold):

    num = ((y_true == 1) & (y_pred >= threshold)).sum()
    denom = (y_pred >= threshold).sum()

    if denom:
        purity = num / denom
    else:
        purity = np.nan

    return purity

def get_completeness(y_true, y_pred, threshold):

    num = ((y_true == 1) & (y_pred >= threshold)).sum()
    denom = (y_true == 1).sum()

    if denom:
        completeness = num / denom
    else:
        completeness = np.nan

    return completeness

def threshold_at_purity_of(y_true, y_pred, threshold):
    
    thresholds = np.sort(y_pred)
    purity = np.zeros_like(thresholds)
    
    for i, t in enumerate(thresholds):
        purity[i] = get_purity(y_true, y_pred, t)

    purity = purity[~np.isnan(purity)] 

    idx = np.argmin(np.abs(purity - threshold))
    
    return thresholds[idx], purity[idx]

def threshold_at_completeness_of(y_true, y_pred, threshold):
    
    thresholds = np.sort(y_pred)
    completeness = np.zeros_like(thresholds)
    
    for i, t in enumerate(thresholds):
        completeness[i] = get_completeness(y_true, y_pred, t)
 
    completeness = completeness[~np.isnan(completeness)]
    
    idx = np.argmin(np.abs(completeness - threshold))

    return thresholds[idx], completeness[idx]


def bayes_conf(N, k, conf=0.683, tol=1.0e-3, step=1.0e-3, a0=None, dx0=None, output=True):
    """
    http://inspirehep.net/record/669498/files/fermilab-tm-2286.PDF
    """
    
    epsilon = k / N
    
    if a0 is None:
        a0 = epsilon
        
    if dx0 is None:
        dx0 = step
    
    bins = np.arange(0, 1 + step, step)
    
    def get_log_p(N, k):
        p = gammaln(N + 2) - gammaln(k + 1) - gammaln(N - k + 1) + k * np.log(bins) + (N - k) * np.log(1 - bins)
        return p
    
    alpha = np.arange(0, a0, step)
    beta = np.arange(epsilon, 1, step)
    
    log_p = get_log_p(N, k)

    def func(x):
        i = np.argmin(np.abs(bins - x))
        return np.exp(log_p[i])

    found = False

    area_best = 1
    alpha_best = alpha[-1]
    beta_best = 1.0

    dxs = np.arange(dx0, 1, step)
    
    for ix, dx in enumerate(dxs):
        
        for ia, a in enumerate(alpha[::-1]):

            b = a + dx
            
            if b > 1 or b < epsilon:
                break
         
            area, err = quad(func, a, b)
                      
            if np.abs(area - conf) < tol:
                area_best = area
                alpha_best = a
                beta_best = b
                found = True
                break
                
            if area > conf:
                # go back a step, recalculate with smaller step
                alpha_best, beta_best, area_best = bayes_conf(
                    N, k, step=0.8*step, a0=a + step, dx0=dx - step, output=False
                    )

                found = True
                # exit the inner for loop for a
                break
                    
        # exit the outer for loop for dx    
        if found:
            break
    
    if output:
        print("Done. N = {0}, k = {1}, area: {2:.3f}, alpha: {3:.4f}, beta: {4:.4f}"
              "".format(N, k, area_best, alpha_best, beta_best, step))
            
    return alpha_best, beta_best, area_best

def make_df(y_true, y_pred):
    df = pd.DataFrame()
    df['y_true'] = y_true
    df['y_pred'] = y_pred
    df = df.sort(columns='y_pred').reset_index(drop=True)
    return df

def bin_df(y_true, y_prob, n_bins=5):
    
    df = make_df(y_true, y_prob)
    bins = np.linspace(0, 1 + 1.e-8, n_bins + 1)
    df['group'] = pd.cut(df.y_pred, bins, labels=list(range(n_bins)))
    
    return df

def get_bayes_interval(y_true, y_prob, n_bins=5, step=0.004, tol=0.001):
    
    df = bin_df(y_true, y_prob, n_bins=n_bins)

    med = np.zeros(n_bins)
    low = np.zeros(n_bins)
    high = np.zeros(n_bins)
    
    for i in range(n_bins):
        bin_ = df[df.group == i]
        N = len(bin_)
        k = (bin_.y_true == 1).sum()
        med[i] = k / N
        low[i], high[i], _ = bayes_conf(N, k, step=step, tol=tol)
        
    return low, med, high

def get_interval(y, n_bins=5):
    
    df = bin_df(y, y, n_bins=n_bins)
    
    pred_med = np.zeros(n_bins)
    pred_low = np.zeros(n_bins)
    pred_high = np.zeros(n_bins)
    
    for i in range(n_bins):
        pred_med[i] = df.y_pred[df.group == i].median()
        pred_low[i] = df.y_pred[df.group == i].quantile(0.16)
        pred_high[i] = df.y_pred[df.group == i].quantile(0.84)

    return pred_low, pred_med, pred_high

def hosmer_lemeshow_table(y_true, y_pred, n_groups=20):
    if n_groups < 2:
        raise ValueError('Number of groups must be greater than or equal to 2')

    if n_groups > len(y_true):
        raise ValueError('Number of predictions must exceed number of groups')

    df = make_df(y_true, y_pred)

    table = pd.DataFrame(columns=('group_size', 'obs_freq', 'pred_freq', 'mean_prob'))
    
    for i in range(n_groups):
        step = len(df) // n_groups
        idx0 = i * step
        group = df[idx0: idx0 + step]
        table.loc[i, 'group_size'] = len(group)
        table.loc[i, 'obs_freq'] = group.y_true.values.sum()
        table.loc[i, 'pred_freq'] = group.y_pred.values.sum()
        table.loc[i, 'mean_prob'] = group.y_pred.mean()
        
    return table

def hosmer_lemeshow_test(y_true, y_pred, n_groups=20):
    
    table = hosmer_lemeshow_table(y_true, y_pred, n_groups=n_groups)

    num = np.square(table.obs_freq.values - table.pred_freq.values)
    den = table.group_size.values * table.mean_prob.values * (1 - table.mean_prob.values)

    mask = (den > 0.0)
    C_hat = np.sum(num[mask] / den[mask])
    df = len(mask) - 2
    p = 1 - sp.stats.chi2.cdf(C_hat, len(mask) - 2)
    
    return C_hat, p

def calibration_error(y_true, y_pred, s=100):
    
    df = make_df(y_true, y_pred)
    
    error = []
    
    for i in range(len(df) - s):
        this_bin = df.loc[i: i + s]
        p_gal = (this_bin.y_true.values == 1).sum() / s
        error.append(np.abs(this_bin.y_pred.values - p_gal).sum() / len(this_bin))
    
    cal = np.mean(error)
    
    return cal
