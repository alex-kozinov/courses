import numpy as np
from scipy.signal import fftconvolve

def squared_distance_from_gt(X, B, F):
    """
    (X - \mu_{i, j})**2

    Parameters
    ----------
    X : array, shape (H, W)
        K images of size H x W.
    F : array, shape (h, w)
        Estimate of villain's face.
    B : array, shape (H, W)
        Estimate of background.
    q : array shape (H-h+1, W-w+1)
        q[dh,dw] - estimate of posterior of displacement (dh,dw)
        of villain's face given image Xk
    
    Returns
    -------
    res : array, shape(H-h+1, W-w+1)
    """
    H, W = X.shape
    h, w = F.shape
    assert B.shape == (H, W)
    
    res = fftconvolve(X**2, np.ones_like(F), mode='valid')
    res -= 2 * fftconvolve(X, F[::-1, ::-1], mode='valid')
    res += np.sum(F**2)

    res += np.sum((X - B)**2)
    res -= fftconvolve((X - B) ** 2, np.ones_like(F), mode='valid')
    return res

def calculate_log_probability(X, F, B, s):
    """
    Calculates log p(X_k|d_k,F,B,s) for all images X_k in X and
    all possible displacements d_k.

    Parameters
    ----------
    X : array, shape (H, W, K)
        K images of size H x W.
    F : array, shape (h, w)
        Estimate of villain's face.
    B : array, shape (H, W)
        Estimate of background.
    s : float
        Estimate of standard deviation of Gaussian noise.

    Returns
    -------
    ll : array, shape(H-h+1, W-w+1, K)
        ll[dh,dw,k] - log-likelihood of observing image X_k given
        that the villain's face F is located at displacement (dh, dw)
    """
    H, W, K = X.shape
    h, w = F.shape
    
    ll = np.zeros((H-h+1, W-w+1, K), dtype=np.float64)
    for k in range(K):
        ll[:, :, k] = -squared_distance_from_gt(X[:, :, k], B, F) / (2 * s**2)
        ll[:, :, k] -= W * H * np.log(2 * np.pi * s**2) / 2

    return ll


def calculate_lower_bound(X, F, B, s, A, q, use_MAP=False):
    """
    Calculates the lower bound L(q,F,B,s,A) for the marginal log likelihood.

    Parameters
    ----------
    X : array, shape (H, W, K)
        K images of size H x W.
    F : array, shape (h, w)
        Estimate of villain's face.
    B : array, shape (H, W)
        Estimate of background.
    s : float
        Estimate of standard deviation of Gaussian noise.
    A : array, shape (H-h+1, W-w+1)
        Estimate of prior on displacement of face in any image.
    q : array
        If use_MAP = False: shape (H-h+1, W-w+1, K)
            q[dh,dw,k] - estimate of posterior of displacement (dh,dw)
            of villain's face given image Xk
        If use_MAP = True: shape (2, K)
            q[0,k] - MAP estimates of dh for X_k
            q[1,k] - MAP estimates of dw for X_k
    use_MAP : bool, optional
        If true then q is a MAP estimates of displacement (dh,dw) of
        villain's face given image Xk.

    Returns
    -------
    L : float
        The lower bound L(q,F,B,s,A) for the marginal log likelihood.
    """
    H, W, K = X.shape
    h, w = F.shape
    eps = 1e-64
        
    ll = calculate_log_probability(X, F, B, s)
    L = ll[q[0], q[1], np.arange(K)] if use_MAP else ll * q
    L += np.log(A[q[0], q[1]] + eps) if use_MAP else q * (np.log(A[:, :, np.newaxis] + eps) - np.log(q + eps))
    
    return L.sum()


def run_e_step(X, F, B, s, A, use_MAP=False):
    """
    Given the current esitmate of the parameters, for each image Xk
    esitmates the probability p(d_k|X_k,F,B,s,A).

    Parameters
    ----------
    X : array, shape(H, W, K)
        K images of size H x W.
    F  : array_like, shape(h, w)
        Estimate of villain's face.
    B : array shape(H, W)
        Estimate of background.
    s : scalar, shape(1, 1)
        Eestimate of standard deviation of Gaussian noise.
    A : array, shape(H-h+1, W-w+1)
        Estimate of prior on displacement of face in any image.
    use_MAP : bool, optional,
        If true then q is a MAP estimates of displacement (dh,dw) of
        villain's face given image Xk.

    Returns
    -------
    q : array
        If use_MAP = False: shape (H-h+1, W-w+1, K)
            q[dh,dw,k] - estimate of posterior of displacement (dh,dw)
            of villain's face given image Xk
        If use_MAP = True: shape (2, K)
            q[0,k] - MAP estimates of dh for X_k
            q[1,k] - MAP estimates of dw for X_k
    """
    
    H, W, K = X.shape
    eps = 1e-64

    ll = calculate_log_probability(X, F, B, s)
    ll_A = ll + np.log(A + eps)[:, :, np.newaxis]
    ll_A = ll_A - ll_A.max(axis=(0, 1))[np.newaxis, np.newaxis, :]

    q = np.exp(ll_A)
    q = q / (q.sum(axis=(0, 1))[np.newaxis, np.newaxis, :])

    if use_MAP:
        new_q = np.zeros((2, K), dtype=int)
        for k in range(K):
            new_q[:,k] = np.unravel_index(np.argmax(q[:, :, k]), A.shape)
        q = new_q

    return q


def run_m_step(X, q, h, w, use_MAP=False):
    """
    Estimates F,B,s,A given esitmate of posteriors defined by q.

    Parameters
    ----------
    X : array, shape(H, W, K)
        K images of size H x W.
    q  :
        if use_MAP = False: array, shape (H-h+1, W-w+1, K)
           q[dh,dw,k] - estimate of posterior of displacement (dh,dw)
           of villain's face given image Xk
        if use_MAP = True: array, shape (2, K)
            q[0,k] - MAP estimates of dh for X_k
            q[1,k] - MAP estimates of dw for X_k
    h : int
        Face mask height.
    w : int
        Face mask width.
    use_MAP : bool, optional
        If true then q is a MAP estimates of displacement (dh,dw) of
        villain's face given image Xk.

    Returns
    -------
    F : array, shape (h, w)
        Estimate of villain's face.
    B : array, shape (H, W)
        Estimate of background.
    s : float
        Estimate of standard deviation of Gaussian noise.
    A : array, shape (H-h+1, W-w+1)
        Estimate of prior on displacement of face in any image.
    """
    H, W, K = X.shape
    F, B, s, A = None, None, None, None
    eps = 1e-64
    q_full = None
    if use_MAP:
        new_q = np.zeros(shape=(H - h + 1, W - w + 1, K))
        for k in range(K):
            new_q[q[0, k]][q[1, k]][k] = 1
        q_full = new_q
        
    # Compute A
    if use_MAP:
        A = q_full.sum(axis=2)
    else:
        A = q.sum(axis=2)
    A /= K
    
    # Compute F
    F = np.zeros((h, w))
    for k in range(K):
        if use_MAP:
            F += X[q[0, k]:q[0, k] + h, q[1, k]:q[1, k] + w, k]
        else:
            F += fftconvolve(X[:, :, k], q[::-1, ::-1, k], mode='valid')
    F /= K
    
    # Compute B
    numirator = np.zeros((H, W))
    denominator = np.zeros((H, W))
    B = np.zeros((H, W))
    
    if use_MAP:
        numirator = np.sum(X, axis=2) - np.sum(X * fftconvolve(q_full, np.ones(shape=(h, w, 1)), mode='full'), axis=2)
        denominator = K - np.sum(fftconvolve(q_full, np.ones(shape=(h, w, 1)), mode='full'), axis=2)
    else:
        numirator = np.sum(X, axis=2) - np.sum(X * fftconvolve(q, np.ones(shape=(h, w, 1)), mode='full'), axis=2)
        denominator = K - np.sum(fftconvolve(q, np.ones(shape=(h, w, 1)), mode='full'), axis=2)
    valid_mask = np.where(denominator > 0)
    B[valid_mask] = numirator[valid_mask] / denominator[valid_mask]

    # Compute s2
    numirator = 0
    for k in range(K):
        if use_MAP:
            gt_image = B.copy()
            gt_image[q[0, k] : q[0, k] + h, q[1, k] : q[1, k] + w] = F
            numirator += (X[:, :, k] - gt_image) ** 2
        else:
            numirator += squared_distance_from_gt(X[:, :, k], B, F) * q[:, :, k]
    s = np.sqrt(numirator.sum() / (K * W * H))

    return F, B, s, A

# def run_m_step(X, q, h=100, w=75, use_MAP=False):
#     H = X.shape[0]
#     W = X.shape[1]
#     K = X.shape[2]

#     if use_MAP:
#         new_q = np.zeros(shape=(H - h + 1, W - w + 1, K))
#         for k in range(K):
#             new_q[q[0, k]][q[1, k]][k] = 1
#         q = new_q
#     A = np.sum(q, axis=2) / K

#     F = np.squeeze(fftconvolve(X, q[::-1, ::-1, ::-1],
#                                mode='valid')) / K

#     B_numerator = np.sum(X, axis=2) - \
#         np.sum(X * fftconvolve(q, np.ones(shape=(h, w, 1)),
#                                mode='full'), axis=2)
#     B_denominator = K - \
#         np.sum(fftconvolve(q, np.ones(shape=(h, w, 1)),
#                            mode='full'), axis=2)  # + 1e-8
#     B = B_numerator / (B_denominator + 1e-64)
#     repeated_B = B[:, :, np.newaxis]
#     repeated_F = F[:, :, np.newaxis]
#     s = 0
#     background = (X - repeated_B) ** 2
#     background_sum = np.sum(background, axis=(0, 1))
#     ll = np.zeros(shape=(H - h + 1, W - w + 1, K))
#     for d_h in range(H-h+1):
#         for d_w in range(W-w+1):
#             x_nu = background_sum.copy()
#             x_nu -= np.sum(background[d_h:d_h+h, d_w:d_w+w, :], axis=(0, 1))
#             x_nu += np.sum((X[d_h:d_h+h, d_w:d_w+w, :] -
#                             repeated_F) ** 2, axis=(0, 1))
#             ll[d_h, d_w, :] = q[d_h, d_w, :] * x_nu
#     s = np.sqrt(np.sum(ll) / (H * W * K))
#     return F, B, s, A

def run_EM(X, h, w, F=None, B=None, s=None, A=None, tolerance=0.001,
           max_iter=50, use_MAP=False):
    """
    Runs EM loop until the likelihood of observing X given current
    estimate of parameters is idempotent as defined by a fixed
    tolerance.

    Parameters
    ----------
    X : array, shape (H, W, K)
        K images of size H x W.
    h : int
        Face mask height.
    w : int
        Face mask width.
    F : array, shape (h, w), optional
        Initial estimate of villain's face.
    B : array, shape (H, W), optional
        Initial estimate of background.
    s : float, optional
        Initial estimate of standard deviation of Gaussian noise.
    A : array, shape (H-h+1, W-w+1), optional
        Initial estimate of prior on displacement of face in any image.
    tolerance : float, optional
        Parameter for stopping criterion.
    max_iter  : int, optional
        Maximum number of iterations.
    use_MAP : bool, optional
        If true then after E-step we take only MAP estimates of displacement
        (dh,dw) of villain's face given image Xk.

    Returns
    -------
    F, B, s, A : trained parameters.
    LL : array, shape(number_of_iters,)
        L(q,F,B,s,A) after each EM iteration (1 iteration = 1 e-step + 1 m-step); 
        number_of_iters is actual number of iterations that was done.
    """
    
    H, W, K = X.shape

    F = np.random.rand(h, w) if F is None else F
    B = np.random.rand(H, W) if B is None else B
    A = np.ones((H-h+1, W-w+1)) / ((H-h+w) * (W-w+1)) if A is None else A
    s = np.random.random_sample() if s is None else s

    LL = [np.inf, ]
    for i in range(max_iter):
        q = run_e_step(X, F, B, s, A, use_MAP)
        F, B, s, A = run_m_step(X, q, h, w, use_MAP)
        
        L = calculate_lower_bound(X, F, B, s, A, q, use_MAP)
        LL.append(L)
        if (LL[-1] - LL[-2] < tolerance):
            break

    return F, B, s, A, np.array(LL[1:])


def run_EM_with_restarts(X, h, w, tolerance=0.001, max_iter=50, use_MAP=False,
                         n_restarts=10):
    """
    Restarts EM several times from different random initializations
    and stores the best estimate of the parameters as measured by
    the L(q,F,B,s,A).

    Parameters
    ----------
    X : array, shape (H, W, K)
        K images of size H x W.
    h : int
        Face mask height.
    w : int
        Face mask width.
    tolerance, max_iter, use_MAP : optional parameters for EM.
    n_restarts : int
        Number of EM runs.

    Returns
    -------
    F : array,  shape (h, w)
        The best estimate of villain's face.
    B : array, shape (H, W)
        The best estimate of background.
    s : float
        The best estimate of standard deviation of Gaussian noise.
    A : array, shape (H-h+1, W-w+1)
        The best estimate of prior on displacement of face in any image.
    L : float
        The best L(q,F,B,s,A).
    """
    F_best, B_best, s_best, A_best, LL = run_EM(
        X=X, h=h, w=w,
        tolerance=tolerance,
        max_iter=max_iter,
        use_MAP=use_MAP
    )
    L_best = LL[-1]

    for _ in range(n_restarts - 1):
        F, B, s, A, LL = run_EM(
            X=X, h=h, w=w,
            tolerance=tolerance,
            max_iter=max_iter,
            use_MAP=use_MAP
        )
        L = LL[-1]
        
        if L > L_best:
            F_best, B_best, s_best, A_best, L_best = F, B, s, A, L

    return F_best, B_best, s_best, A_best, L_best
