from __future__ import (
    division, print_function, unicode_literals, absolute_import
    )
import numpy as np
from sklearn.neighbors import KernelDensity
from .metrics import threshold_at_completeness_of, threshold_at_purity_of

def get_log_density(x, bins):

    x_kde = bins[:, np.newaxis]
    bandwidth = 1.06 * np.std(x) * np.power(len(x), -0.2)
    kde = KernelDensity(kernel='gaussian', bandwidth=bandwidth).fit(x[:, np.newaxis])
    log_density = kde.score_samples(x_kde)

    return log_density

def kde_completeness(y_true, y_pred, mag, threshold=0.5, bins=None):
    
    if bins is None:
        bins = np.linspace(mag.min(), mag.max(), 100)

    true_class = mag[y_true == 1]
    log_dens_true = get_log_density(true_class, bins)

    tp = mag[(y_pred >= threshold) & (y_true == 1)]
    log_dens_tp = get_log_density(tp, bins)

    completeness = (
        len(tp) * np.exp(log_dens_tp) / np.exp(log_dens_tp).sum()
        / (len(true_class) * np.exp(log_dens_true) / np.exp(log_dens_true).sum())
        )

    return completeness

def kde_purity(y_true, y_pred, mag, threshold=0.5, bins=None):
    
    if bins is None:
        bins = np.linspace(mag.min(), mag.max(), 100)

    true_class = mag[y_pred >= threshold]
    log_dens_true = get_log_density(true_class, bins)

    tp = mag[(y_pred >= threshold) & (y_true == 1)]
    log_dens_tp = get_log_density(tp, bins)

    purity = (
        len(tp) * np.exp(log_dens_tp) / np.exp(log_dens_tp).sum()
        / (len(true_class) * np.exp(log_dens_true) / np.exp(log_dens_true).sum())
        )

    return purity

def get_stellar_fraction(y_true, y_pred, mag, bins=None):

    if bins is None:
        bins = np.linspace(mag.min(), mag.max(), 100)

    log_dens_mag = get_log_density(mag, bins)

    sx = mag[y_true == 1]
    log_dens_sx = get_log_density(sx, bins)

    stellar_fraction = (
        len(sx) * np.exp(log_dens_sx) / np.exp(log_dens_sx).sum()
        / (len(mag) * np.exp(log_dens_mag) / np.exp(log_dens_mag).sum())
        )
    
    return stellar_fraction

def confidence_band(func, y_true, y_pred, mag, p_cut=0.5, n_boots=100, bins=None):
    
    boots = [func(y_true, y_pred, mag, bins=bins)]
    
    print("Bootstrapping...")
    
    for i in range(1, n_boots):
        rows = np.floor(np.random.rand(len(y_true)) * len(y_true)).astype(int)
        mag_boots = mag[rows]
        pred_boots = y_pred[rows]
        true_boots = y_true[rows]
        boots.append(func(true_boots, pred_boots, mag_boots, bins=bins))
        
        if i % (n_boots / 10) == 0:
            print("{0:.0f} percent complete...".format(i / n_boots * 100))
    
    lower = np.percentile(boots, 0.16, axis=0)
    median = np.percentile(boots, 0.50, axis=0)
    upper = np.percentile(boots, 0.84, axis=0)
    
    print("Complete.")
    
    return median, lower, upper
