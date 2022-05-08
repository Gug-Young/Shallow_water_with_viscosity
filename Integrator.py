import numpy as np
from numba import jit

# @jit()
def HEUN2(f, y0, t, args=()):
        n = len(t)
        y = np.zeros((n, len(y0)))
        y[0] = y0
        
        for i in range(n - 1):
            h = t[i+1] - t[i]
            k1 = f(t[i], y[i], *args)
            k2 = f(t[i] + 2* h / 3., y[i] + 2 * k1 * h / 3., *args)
            y[i+1] = y[i] + (h / 4.) * (k1 + 3*k2)
        return y
# @jit()
def RK4(f, y0, t, args=()):
        n = len(t)
        y = np.zeros((n, len(y0)))
        y[0] = y0
        
        for i in range(n - 1):
            h = t[i+1] - t[i]
            k1 = f(t[i],y[i], *args)
            k2 = f(t[i] + h / 2., y[i] + k1 * h / 2.,  *args)
            k3 = f(t[i] + h / 2., y[i] + k2 * h / 2.,  *args)
            k4 = f(t[i] + h, y[i] + k3 * h, *args)
            y[i+1] = y[i] + (h / 6.) * (k1 + 2*k2 + 2*k3 + k4)
        return y