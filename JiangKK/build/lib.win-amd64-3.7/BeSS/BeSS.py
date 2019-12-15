from BeSS.cbess import pywrap_bess_lm


def bess_lm(X, y, T0, max_steps, beta, weights, normal = True):
    result = pywrap_bess_lm(X, y, T0, max_steps, beta, weights, len(beta), T0, normal)
    return result


