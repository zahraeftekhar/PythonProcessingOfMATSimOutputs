# %matplotlib inline

import warnings
import numpy as np
import pandas as pd
import scipy.stats as st
import statsmodels as sm
import matplotlib
import matplotlib.pyplot as plt
import scipy
import scipy.stats
from scipy.stats import gamma
import numpy as np
import math
from scipy.stats import gaussian_kde
from sklearn.neighbors import KernelDensity

# ______________________best fit for duration of home:st.foldcauchy,exponnorm(K,loc,scale),lognormal____________________________________
from _4_groundTruthAnalysis_locationActivityDistribution import homeDurations, homeStarts, workDurations, workStarts, \
    nHomeActivity, nTotalActivity, nWorkActivity

# homeStarts.reshape(-1, homeStarts.shape[-1])
# data = scipy.int_((homeStarts['start_time(sec)'][homeStarts['start_time(sec)']>7200]))
data = ((homeDurations['duration(sec)'][homeDurations['duration(sec)'] > 0]) / 3600)
plt.rcParams['figure.figsize'] = (16.0, 12.0)
plt.style.use('ggplot')


# Create models from data
def best_fit_distribution(data, bins=288, ax=None):
    """Model data by finding best fit distribution to data"""
    # Get histogram of original data
    y, x = np.histogram(data, bins=bins, density=True)
    x = (x + np.roll(x, -1))[:-1] / 2.0

    # Distributions to check
    DISTRIBUTIONS = [
        st.alpha,st.anglit,st.arcsine,st.beta,st.betaprime,st.bradford,st.burr,st.cauchy,st.chi,st.chi2,st.cosine,st.dgamma,st.dweibull,
        st.erlang,st.expon,st.exponnorm,st.exponweib,st.exponpow,st.f,st.fatiguelife,
        st.foldnorm,st.frechet_r,st.frechet_l,st.genlogistic,st.genpareto,st.genexpon,
        st.gausshyper,st.gamma,st.gengamma,st.genhalflogistic,st.gennorm,st.gilbrat,st.gompertz,st.gumbel_r,
        st.gumbel_l,st.halfcauchy,st.halflogistic,st.halfnorm,st.halfgennorm,st.hypsecant,st.invgamma,st.invgauss,
        st.invweibull,st.johnsonsb,st.johnsonsu,st.kstwobign,st.laplace,
        st.logistic,st.loggamma,st.loglaplace,st.lognorm,st.lomax,st.maxwell,st.mielke,st.nakagami,
        st.norm,st.pareto,st.pearson3,st.powerlaw,st.powerlognorm,st.powernorm,st.rdist,st.reciprocal,
        st.rayleigh,st.rice,st.semicircular,st.t,st.triang,st.truncexpon,
        st.uniform,st.vonmises,st.vonmises_line,st.wald,st.weibull_min,st.weibull_max,st.wrapcauchy
    ]
    # DISTRIBUTIONS = [st.logistic, st.norm
    #                  # st.dgamma
    #                  # st.gamma
    #                  # st.dgamma, st.expon, st.exponnorm,
    #                  # st.genlogistic, st.gennorm, st.gausshyper, st.gamma, st.gengamma, st.halflogistic, st.halfnorm, st.halfgennorm,
    #                  # st.invgamma,
    #                  # st.laplace, st.logistic, st.loggamma, st.lognorm,
    #                  # st.norm, st.powerlaw, st.powerlognorm,
    #                  # st.rayleigh, st.t
    #                  ]

    # Best holders
    best_distribution = st.norm
    best_params = (0.0, 1.0)
    best_sse = np.inf

    # Estimate distribution parameters from data
    for distribution in DISTRIBUTIONS:
        print(distribution.name)
        # Try to fit the distribution
        try:
            # Ignore warnings from data that can't be fit
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore')

                # fit dist to data
                params = distribution.fit(data)

                # Separate parts of parameters
                arg = params[:-2]
                loc = params[-2]
                scale = params[-1]

                # Calculate fitted PDF and error with fit in distribution
                pdf = distribution.pdf(x, loc=loc, scale=scale, *arg)
                sse = np.sum(np.power(y - pdf, 2.0))

                # if axis pass in add to plot
                try:
                    if ax:
                        pd.Series(pdf, x).plot(ax=ax, label='{}'.format(distribution.name))

                except Exception:
                    pass

                # identify if this distribution is better
                if best_sse > sse > 0:
                    best_distribution = distribution
                    best_params = params
                    best_sse = sse

        except Exception:
            pass

    return (best_distribution.name, best_params)


def make_pdf(dist, params, size=10000):
    """Generate distributions's Probability Distribution Function """

    # Separate parts of parameters
    arg = params[:-2]
    loc = params[-2]
    scale = params[-1]

    # Get sane start and end points of distribution
    start = dist.ppf(0.01, *arg, loc=loc, scale=scale) if arg else dist.ppf(0.01, loc=loc, scale=scale)
    end = dist.ppf(0.99, *arg, loc=loc, scale=scale) if arg else dist.ppf(0.99, loc=loc, scale=scale)

    # Build PDF and turn into pandas Series
    x = np.linspace(start, end, size)
    y = dist.pdf(x, loc=loc, scale=scale, *arg)
    pdf = pd.Series(y, x)

    return pdf


# Load data from statsmodels datasets
data = pd.Series(data)

# Plot for comparison
plt.figure(figsize=(12, 8))
ax = data.plot(kind='hist', bins=288, density=True, alpha=0.5)

# Save plot limits
dataYLim = ax.get_ylim()

# Find best fit distribution
best_fit_name, best_fit_params = best_fit_distribution(data, 288, ax)
best_dist = getattr(st, best_fit_name)

# Update plots
ax.set_ylim(dataYLim)
plt.xticks((np.arange(0, 24, step=1)))
plt.legend(loc='upper right')
ax.set_title(u'home stay duration.\n All Fitted Distributions')
ax.set_xlabel(u'Duration (hour)')
ax.set_ylabel('Frequency')

# Make PDF with best params
pdf = make_pdf(best_dist, best_fit_params)

# Display
plt.figure(figsize=(12, 8))
plt.xticks((np.arange(0, 24, step=1)))
ax = pdf.plot(lw=2, label='PDF', legend=True)
data.plot(kind='hist', bins=288, density=True, alpha=0.5, label='Data', legend=True, ax=ax)
param_names = (best_dist.shapes + ', loc, scale').split(', ') if best_dist.shapes else ['loc', 'scale']
param_str = ', '.join(['{}={:0.2f}'.format(k, v) for k, v in zip(param_names, best_fit_params)])
dist_str = '{}({})'.format(best_fit_name, param_str)

ax.set_title(u'home stay duration . with best fit distribution \n' + dist_str)
ax.set_xlabel(u'Duration (hour)')
ax.set_ylabel('Frequency')
# _____________________________________________________________________________________________________
####################################_____________________###############################################
####################################| Bayesian Inference |###############################################
####################################|____________________|###############################################
homeData = pd.DataFrame()
homeData['Duration(hour)'] = (homeDurations['duration(sec)'])/3600
homeData['DurationProb'] = None
homeData['DurationProb'] = st.exponnorm.pdf(x=homeData['Duration(hour)'],K=best_fit_params[0],
                                          loc = best_fit_params[1], scale = best_fit_params[2])
############################# best fit for duration of work: dgamma, dweibull ########################################
# ########## 3)gennormal(beta=best_fit_params[0],loc = best_fit_params[1], scale = best_fit_params[2]) ##################
workDuration = pd.Series(workDurations)
data = ((workDuration[workDuration > 0]) / 3600)
plt.rcParams['figure.figsize'] = (16.0, 12.0)
plt.style.use('ggplot')
# Load data from statsmodels datasets
data = pd.Series(data)

# Plot for comparison
plt.figure(figsize=(12, 8))
ax = data.plot(kind='hist', bins=288, density=True, alpha=0.5)

# Save plot limits
dataYLim = ax.get_ylim()

# Find best fit distribution
best_fit_name, best_fit_params = best_fit_distribution(data, 288, ax)
best_dist = getattr(st, best_fit_name)

# Update plots
ax.set_ylim(dataYLim)
plt.xticks((np.arange(0, 24, step=1)))
plt.legend(loc='upper right')
ax.set_title(u'work stay duration.\n All Fitted Distributions')
ax.set_xlabel(u'Duration (hour)')
ax.set_ylabel('Frequency')

# Make PDF with best params
pdf = make_pdf(best_dist, best_fit_params)

# Display
plt.figure(figsize=(12, 8))
plt.xticks((np.arange(0, 24, step=1)))
ax = pdf.plot(lw=2, label='PDF', legend=True)
data.plot(kind='hist', bins=288, density=True, alpha=0.5, label='Data', legend=True, ax=ax)
param_names = (best_dist.shapes + ', loc, scale').split(', ') if best_dist.shapes else ['loc', 'scale']
param_str = ', '.join(['{}={:0.2f}'.format(k, v) for k, v in zip(param_names, best_fit_params)])
dist_str = '{}({})'.format(best_fit_name, param_str)
ax.set_title(u'work stay duration . with best fit distribution \n' + dist_str)
ax.set_xlabel(u'Duration (hour)')
ax.set_ylabel('Frequency')

# _____________________________________________________________________________________________________
####################################_____________________###############################################
####################################| Bayesian Inference |###############################################
####################################|____________________|###############################################
workData = pd.DataFrame()
workData['Duration(hour)'] = (pd.Series(workDurations)/3600)
workData['DurationProb'] = None
workData['DurationProb'] = st.gennorm.pdf(x=workData['Duration(hour)'],beta=best_fit_params[0],
                                          loc = best_fit_params[1], scale = best_fit_params[2])















# _______________________best fit for start of home stay: genextreme,loggamma(c,loc,scale), logistic ____________________________________
from _4_groundTruthAnalysis_locationActivityDistribution import homeDurations, homeStarts

data = ((homeStarts['start_time(sec)'][homeStarts['start_time(sec)'] > 7200]) / 3600)
# data = scipy.int_((homeDurations['duration(sec)'][homeDurations['duration(sec)']>0]))
plt.rcParams['figure.figsize'] = (16.0, 12.0)
plt.style.use('ggplot')

# Load data from statsmodels datasets
data = pd.Series(data)

# Plot for comparison
plt.figure(figsize=(12, 8))

# plt.rcParams['axes.color_cycle'] =
ax = data.plot(kind='hist', bins=288, density=True, alpha=0.5)

# Save plot limits
dataYLim = ax.get_ylim()

# Find best fit distribution
best_fit_name, best_fit_params = best_fit_distribution(data, 288, ax)
best_dist = getattr(st, best_fit_name)

# Update plots
ax.set_ylim(dataYLim)
plt.xticks((np.arange(2, 26, step=1)))
plt.legend(loc='upper right')
ax.set_title(u'home stay start time.\n All Fitted Distributions')
ax.set_xlabel(u'start time (hour)')
ax.set_ylabel('Frequency')

# Make PDF with best params
pdf = make_pdf(best_dist, best_fit_params)

# Display
plt.figure(figsize=(12, 8))
plt.xticks((np.arange(2, 26, step=1)))
ax = pdf.plot(lw=2, label='PDF', legend=True)
data.plot(kind='hist', bins=288, density=True, alpha=0.5, label='Data', legend=True, ax=ax)

param_names = (best_dist.shapes + ', loc, scale').split(', ') if best_dist.shapes else ['loc', 'scale']
param_str = ', '.join(['{}={:0.2f}'.format(k, v) for k, v in zip(param_names, best_fit_params)])
dist_str = '{}({})'.format(best_fit_name, param_str)

ax.set_title(u'home stay start time . with best fit distribution \n' + dist_str)
ax.set_xlabel(u'start time (hour)')
ax.set_ylabel('Frequency')
# _____________________________________________________________________________________________________
####################################_____________________###############################################
####################################| Bayesian Inference |###############################################
####################################|____________________|###############################################

homeData['StartProb'] = None
homeData['start(hour)'] = homeStarts['start_time(sec)']/3600
homeData['StartProb'] = st.loggamma.pdf(x=homeData['start(hour)'],c=best_fit_params[0],
                                       loc = best_fit_params[1], scale = best_fit_params[2])
# ____________best fit for start of work stay : t(df,loc,scale),fisk(c,loc,scale) or Log-logistic distribution,loglaplace____________________________________
from _4_groundTruthAnalysis_locationActivityDistribution import homeDurations, homeStarts

data = ((workStarts['start_time(sec)'][workStarts['start_time(sec)'] > 7200]) / 3600)
# data = scipy.int_((homeDurations['duration(sec)'][homeDurations['duration(sec)']>0]))
plt.rcParams['figure.figsize'] = (16.0, 12.0)
plt.style.use('ggplot')

# Load data from statsmodels datasets
data = pd.Series(data)

# Plot for comparison
plt.figure(figsize=(12, 8))

# plt.rcParams['axes.color_cycle'] =
ax = data.plot(kind='hist', bins=288, density=True, alpha=0.5)

# Save plot limits
dataYLim = ax.get_ylim()

# Find best fit distribution
best_fit_name, best_fit_params = best_fit_distribution(data, 288, ax)
best_dist = getattr(st, best_fit_name)

# Update plots
ax.set_ylim(dataYLim)
plt.xticks((np.arange(2, 26, step=1)))
plt.legend(loc='upper right')
ax.set_title(u'work stay start time.\n All Fitted Distributions')
ax.set_xlabel(u'start time (hour)')
ax.set_ylabel('Frequency')

# Make PDF with best params
pdf = make_pdf(best_dist, best_fit_params)

# Display
plt.figure(figsize=(12, 8))
plt.xticks((np.arange(2, 26, step=1)))
ax = pdf.plot(lw=2, label='PDF', legend=True)
data.plot(kind='hist', bins=288, density=True, alpha=0.5, label='Data', legend=True, ax=ax)

param_names = (best_dist.shapes + ', loc, scale').split(', ') if best_dist.shapes else ['loc', 'scale']
param_str = ', '.join(['{}={:0.2f}'.format(k, v) for k, v in zip(param_names, best_fit_params)])
dist_str = '{}({})'.format(best_fit_name, param_str)

ax.set_title(u'work stay start time . with best fit distribution \n' + dist_str)
ax.set_xlabel(u'start time (hour)')
ax.set_ylabel('Frequency')

#******************************************
x2 = np.linspace(0, 26, 10000)
y2 = gaussian_kde(data).pdf(x2) #*******
pdf2 = pd.Series(y2, x2)
y1 = st.t.pdf(x=x2,df=best_fit_params[0],loc = best_fit_params[1], scale = best_fit_params[2])
pdf1 = pd.Series(y1, x2)
plt.figure(figsize=(12, 8))
plt.xticks((np.arange(2, 26, step=1)))
ax = pdf1.plot(lw=2, label='PDF', legend=True)
ax1 = pdf2.plot(lw=2, label='gaussian Kernel Density Estimation', legend=True, ax=ax)
data.plot(kind='hist', bins=288, density=True, alpha=0.5, label='Data', legend=True, ax=ax)

param_names = (st.t.shapes + ', loc, scale').split(', ') if st.t.shapes else ['loc', 'scale']
param_str = ', '.join(['{}={:0.2f}'.format(k, v) for k, v in zip(param_names, st.t.fit(data))])
dist_str = '{}({})'.format(st.t.name, param_str)
ax.set_title(u'work stay start time . with best fit distribution ' + dist_str+ '\n ,with Gaussian KDE, and regular fit distribution (___)')
ax.set_xlabel(u'start time (hour)')
ax.set_ylabel('Frequency')




# _____________________________________________________________________________________________________
####################################_____________________###############################################
####################################| Bayesian Inference |###############################################
####################################|____________________|###############################################

workData['StartProb'] = None
workData['start(hour)'] = workStarts['start_time(sec)']/3600
workData['StartProb'] = st.t.pdf(x=workData['start(hour)'],df=best_fit_params[0],
                                       loc = best_fit_params[1], scale = best_fit_params[2])









































# data.plot(kind='hist', bins=288, density=True, alpha=0.5, label='Data', legend=True, ax=ax)
# Build PDF and turn into pandas Series
x2 = np.linspace(0, 26, 10000)
y2 = gaussian_kde(data).pdf(x2)
pdf2 = pd.Series(y2, x2)
y1 = st.t.pdf(x=x2,df=best_fit_params[0],loc = best_fit_params[1], scale = best_fit_params[2])
pdf1 = pd.Series(y1, x2)
plt.figure(figsize=(12, 8))
plt.xticks((np.arange(2, 26, step=1)))
ax = pdf1.plot(lw=2, label='PDF', legend=True)
ax1 = pdf2.plot(lw=2, label='gaussian Kernel Density Estimation', legend=True, ax=ax)
data.plot(kind='hist', bins=288, density=True, alpha=0.5, label='Data', legend=True, ax=ax)

param_names = (st.t.shapes + ', loc, scale').split(', ') if st.t.shapes else ['loc', 'scale']
param_str = ', '.join(['{}={:0.2f}'.format(k, v) for k, v in zip(param_names, st.t.fit(data))])
dist_str = '{}({})'.format(st.t.name, param_str)
ax.set_title(u'work stay start time . with best fit distribution ' + dist_str+ '\n ,with Gaussian KDE, and regular fit distribution (___)')
ax.set_xlabel(u'start time (hour)')
ax.set_ylabel('Frequency')