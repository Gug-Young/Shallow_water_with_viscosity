import numpy as np

def Euler(f,y0,t,args=()):
    n = len(t)
    h=t[1]-t[0]
    y = np.zeros((n, len(y0)))
    y[0] = y0
    for i in range(n - 1):
        y[i+1]=y[i] + h*f(y[i],t[i],*args)
    return y

def RK2(f,y0,t,args=()):
    n = len(t)
    y = np.zeros((n, len(y0)))
    y[0] = y0
    for i in range(n - 1):
        h = t[i+1] - t[i]
        k1 = f(y[i], t[i], *args)
        k2 = f(y[i] + k1 * h / 2., t[i] + h / 2., *args)
        y[i+1] = y[i] + k2 * h
    return y

def RK4(f, y0, t, args=()):
    n = len(t)
    y = np.zeros((n, len(y0)))
    y[0] = y0

    for i in range(n - 1):
        h = t[i+1] - t[i]
        k1 = f(y[i], t[i], *args)
        k2 = f(y[i] + k1 * h / 2., t[i] + h / 2., *args)
        k3 = f(y[i] + k2 * h / 2., t[i] + h / 2., *args)
        k4 = f(y[i] + k3 * h, t[i] + h, *args)
        y[i+1] = y[i] + (h / 6.) * (k1 + 2*k2 + 2*k3 + k4)
    return y

def Modified_Euler(f,y0,t,args=()):
    h=t[1]-t[0]
    n = len(t)
    y = np.zeros((n, len(y0)))
    y[0] = y0
    for i in range( n - 1 ):
        k1 = h * f(y[i] , t[i] , *args)
        k2 = h * f(y[i] + k1, t[i] + h, *args)
        y[i+1] = y[i]+(k1 + k2)*0.5
    return y
def leapfrog(f,df,y0,v0,t,args=()):
    h = t[1]-t[0]
    n = len(t)
    y = np.zeros((n, len(y0)))
    v = np.zeros((n+1, len(v0)))
    y[0] = y0
    v[0] = v0
    v[1] = v[0] + 0.5 * h * df(y0,t[0]) #velocity at 1/2 delta t
    for i in range(1,n):
        y[i] = y[i-1] + h * f(v[i],t[i]+h/2) #pos at t+ delta t
        v[i+1] = v[i] + h * df(y[i],t[i]) #velocity at
    return y,v[1:,:]

        
    

def Error(origin,method,f,y0,t,args=()):
    origin_arr = np.array(origin(t,*args))
    method_arr = np.array(method(f,y0,t,args))
    if origin_arr.shape == method_arr.shape:
        Error_arr = np.fabs(origin_arr-method_arr)
    else:
        Error_arr = np.fabs(origin_arr-method_arr.T)
    max_Error = np.max(Error_arr)
    return max_Error,Error_arr