import numpy as np
from scipy.signal import fftconvolve
from scipy.stats import randint, binom, poisson
import scipy.stats as sps

# There should be no main() in this file!!! 
# Nothing should start running when you import this file somewhere.
# You may add other supporting functions to this file.
#
# Important rules:
# 1) Function pa_bc must return tensor which has dimensions (#a x #b x #c),
#    where #v is a number of different values of the variable v.
#    For input variables #v = how many input values of this variable you give to the function.
#    For output variables #v = number of all possible values of this variable.
#    Ex. for pb_a: #b = bmax-bmin+1,   #a is arbitrary.
# 2) Random variables in function names must be written in alphabetic order
#    e.g. pda_cb is an improper function name (pad_bc must be used instead)
# 3) Single dimension must be explicitly stated:
#    if you give only one value of a variable a to the function pb_a, i.e. #a=1, 
#    then the function pb_a must return tensor of shape (#b, 1), not (#b,).
#
# The format of all the functions for distributions is the following:
# Inputs:
# params - dictionary with keys 'amin', 'amax', 'bmin', 'bmax', 'p1', 'p2', 'p3'
# model - model number, number from 1 to 4
# all other parameters - values of the conditions (variables a, b, c, d).
#                        Numpy vectors of size (k,), where k is an arbitrary number.
#                        For variant 3: c and d must be numpy arrays of size (k,N),
#                        where N is a number of lectures.
# Outputs:
# prob, val
# prob - probabilities for different values of the output variable with different input conditions
#        prob[i,...] = p(v=val[i]|...)
# val - support of a distribution, numpy vector of size (#v,) for variable v
#
# Example 1:
#    Function pc_ab - distribution p(c|a,b)
#    Input: a of size (k_a,) and b of size (k_b,)
#    Result: prob of size (cmax-cmin+1,k_a,k_b), val of size (cmax-cmin+1,) 
#
# Example 2 (for variant 3):
#    Function pb_ad - distribution p(b|a,d_1,...,d_N)
#    Input: a of size (k_a,) and d of size (k_d,N)
#    Result: prob of size (bmax-bmin+1,k_a,k_d), val of size (bmax-bmin+1,)
#
# The format the generation function from variant 3 is the following:
# Inputs:
# N - how many points to generate
# all other inputs have the same format as earlier
# Outputs:
# d - generated values of d, numpy array of size (N,#a,#b)

# In variant 1 the following functions are required:
# def pa(params, model):
    
# def pb(params, model):
    
# def pc(params, model):

# def pd(params, model):
    
# def pc_a(a, params, model):

# def pc_b(b, params, model):
    
# def pc_d(d, params, model):

# def pc_ab(a, b, params, model):
    
# def pc_abd(a, b, d, params, model):
    
# # In variant 2 the following functions are required:
# def pa(params, model):
    
# def pb(params, model):
    
# def pc(params, model):

# def pd(params, model):

# def pc_a(a, params, model):

# def pc_b(b, params, model):
    
# def pb_a(a, params, model):

# def pb_d(d, params, model):
    
# def pb_ad(a, d, params, model):


# In variant 3 the following functions are required:

def clamp_prob(prob):
    prob[np.isnan(prob)] = 0
    prob[np.where(prob < 0)] = 0
    return prob

def pa(params, model):
    amin = params["amin"]
    amax = params["amax"]
    values = np.arange(amin, amax + 1)

    return randint.pmf(values, amin, amax + 1), values


def pb(params, model):
    new_params = dict(
        amin=params["bmin"],
        amax=params["bmax"]
    )
    return pa(new_params, model)

def pc_ab(a, b, params, model):
    prob = None
    values = np.arange(0, params['amax'] + params['bmax'] + 1)

    if model == 4:
        a = a[:, None, None]
        b = b[None, :, None]
        c = values[None, None, :]

        lmbda = a * params['p1'] + b * params['p2']
        prob = poisson.pmf(c, lmbda)
    else:
        a = a.reshape((-1, 1))
        b = b.reshape((-1, 1))
        bin_a_probs = binom.pmf(np.arange(params["amax"] + 1)[None, :], a, params["p1"])
        bin_b_probs = binom.pmf(np.arange(params["bmax"] + 1)[None, :], b, params["p2"])
    
        bin_a_probs_3d = np.ones((a.shape[0], b.shape[0], params["amax"] + 1)) * bin_a_probs[:, None, :]
        bin_b_probs_3d = np.ones((a.shape[0], b.shape[0], params["bmax"] + 1)) * bin_b_probs[None, :, :]
    
#         prob = fftconvolve(bin_a_probs_3d, bin_b_probs_3d, axes=2)
#         prob = np.apply_along_axis(lambda m: np.convolve(m, bin_b_probs_3d), axis=2, arr=bin_a_probs_3d)
    
        prob = np.ones((a.shape[0], b.shape[0], values.shape[0]))
        for i in range(a.shape[0]):
            for j in range(b.shape[0]):
#                 prob[i, j] = fftconvolve(bin_a_probs_3d[i, j], bin_b_probs_3d[i, j])
                prob[i, j] = np.convolve(bin_a_probs_3d[i, j].flatten(), bin_b_probs_3d[i, j].flatten())
    prob = np.moveaxis(prob, -1, 0)
    return clamp_prob(prob), values

def pc(params, model):
    p_a, a_values = pa(params, model)
    p_b, b_values = pb(params, model)
 
    p_c_ab, c_values = pc_ab(a_values, b_values, params, model)
    p_c = p_c_ab.dot(p_b)
    p_c = p_c.dot(p_a)
 
    return clamp_prob(p_c), c_values

def pd_c(c, params, model):
    c = c.reshape((-1, 1))
    values = np.arange(0, 2 * (params["amax"] + params["bmax"]) + 1)[None, :]
    all_values = np.ones((c.shape[0], values.shape[1])) * values - c
    prob = binom.pmf(all_values, c, params["p3"])

    prob = np.moveaxis(prob, -1, 0)
    return clamp_prob(prob), values.flatten()

def pd(params, model):
    p_c, c_values = pc(params, model)
    p_d_c, d_values = pd_c(c_values, params, model)
    
    p_c = p_c.reshape((1, -1))
    p_d = (p_d_c * p_c).sum(axis=1).flatten()
    return clamp_prob(p_d), d_values
    
def pd_ab(a, b, params, model):
    p_c_ab, c_values = pc_ab(a, b, params, model)
    p_d_c, d_values = pd_c(c_values, params, model)
    
    p_c_ab = np.moveaxis(p_c_ab, 0, -1)
    p_d_c = np.moveaxis(p_d_c, 0, -1)

    p_d = p_c_ab.dot(p_d_c)
    p_d = np.moveaxis(p_d, -1, 0)

    return clamp_prob(p_d), d_values

def pd_a(a, params, model):
    p_b, b_values = pb(params, model)
    p_d_ab, d_values = pd_ab(a, b_values, params, model)

    p_d_a = p_d_ab.dot(p_b)
    return clamp_prob(p_d_a), d_values

def pb_ad(a, d, params, model):
    k_a = a.shape[0]
    k_d, N = d.shape

    p_b, b_values = pb(params, model)
    p_d_ab, d_values = pd_ab(a, b_values, params, model)
    p_d_a, _, = pd_a(a, params, model)
    k_b = b_values.shape[0]
    
    p_b_ad = np.take(p_d_ab, d, axis=0).prod(axis=1)
    
    assert p_b_ad.shape == (k_d, k_a, k_b)
    
    p_b_ad *= p_b[None, None, :]
    p_b_ad /= p_b_ad.sum(axis=2)[:, :, None]
    p_b_ad = p_b_ad.transpose(2, 1, 0)
    
    return clamp_prob(p_b_ad), b_values

def pb_d(d, params, model):
    _, a_values = pa(params, model)
    _, b_values = pb(params, model)
    p_d_ab, _ = pd_ab(a_values, b_values, params, model)
    
    k_a = a_values.shape[0]
    k_b = b_values.shape[0]
    k_d = d.shape[0]
    
    p_b_ad = np.take(p_d_ab, d, axis=0).prod(axis=1).transpose(2, 1, 0)
    assert p_b_ad.shape == (k_b, k_a, k_d)
    p_b_d = np.sum(p_b_ad, axis=1)
    assert p_b_d.shape == (k_b, k_d)
    p_b_d /= np.sum(p_b_d, axis=0).reshape((1, -1))

    return clamp_prob(p_b_d), b_values

def generate(N, a, b, params, model):
    p_d_ab, d_values = pd_ab(a, b, params, model)

    k_d, k_a, k_b = p_d_ab.shape

    d = np.zeros((N, k_a, k_b))
    for i in range(k_a):
        for j in range(k_b):
            d[:, i, j] = np.random.choice(d_values, N, p=p_d_ab[:, i, j])
    # d = np.random.choice(values.flatten(), (N, k_a, k_b) , p=p_d_ab.flatten())
    return d
