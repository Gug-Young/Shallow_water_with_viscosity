import numpy as np
import matplotlib.pyplot as plt

def padding_1d(u,padding ='edge'):
    uc = u.copy()
    if padding == 'edge':
        u_pad = np.pad(u,(1,1),'edge')
    else:
        u_pad = np.pad(u,(1,1),'constant',constant_values=0)
    uw = u_pad[:-2];ue = u_pad[2:]
    return uc,uw,ue

def slicing_1d(u):
    uc = u.copy()
    uw = u[:-1]
    ue = u[1:]
    return uc,uw,ue

def h_update_1d(u,h,H,dx):
    hc,hw,he = padding_1d(h)
    uc,uw,ue = padding_1d(u)
    Hc,Hw,He = padding_1d(H)
    
    eta_e = np.where(uc>=0,hc+Hc,he+He)
    eta_w = np.where(uc>=0,hw+Hw,hc+Hc)

    uhwe = (uc*eta_e - uw*eta_w)/dx
    return uhwe

def udu_update_1d(u,dx):
    uc,uw,ue = padding_1d(u)
    
    u_e = np.where(uc>=0,uc,ue)
    u_w = np.where(uc>=0,uw,uc)

    udu = (uc)*(u_e - u_w)/dx
    return udu

def get_hat(f,u):
    fc,fw,fe = padding_1d(f,'edge')
    
    fhat = np.where(u>=0,fc,fe)
    return fhat 

def get_F_E(h,u):
    hpad = np.pad(h,(1,0),'constant',constant_values=0)
    upad = np.pad(u,(1,0),'constant',constant_values=0)
    
    hhat = get_hat(hpad,upad)
    F_E = hhat*upad
    return F_E

def update_h(h,u,dx,dt):
    hn = h.copy();un = u.copy()
    # un = np.where(hn>=z,un,0)
    F_E = get_F_E(hn,un)
    h = hn -  dt/dx*(F_E[1:] - F_E[:-1])
    return h

def get_F_i(F_E):
    Fi = 0.5*(F_E[:-1]+F_E[1:])
    return Fi

def get_h_half(h):
    h_half = 0.5*(h[:-1]+h[1:])
    return h_half

def get_u_ihat(F_i,u):
    uc,uw,ue = padding_1d(u,'constant')
    uhat = np.where(F_i>=0,uw,uc)
    return uhat
def get_kappa_vsv(h_half,kappa,mu):
    kvsv = kappa/(1+kappa*h_half/(3*mu))
    return kvsv

def get_dt(F_E,h,dx,c_num,g=9.81):
    nu = c_num * dx
    de = np.max(np.abs(F_E[1:]+F_E[:-1])*(2*h) + np.sqrt(g*h))
    dt = nu/de
    if dt == np.nan: dt = 0.005
    return dt
    
def shallow_water_viscous_sorce(h,u,z,dx,dt,kappa,mu,hs,c_num=0.7,g=9.81):
    hn = h.copy()
    un = u.copy()
    # if hs>err : 
    #     un[1] = np.sqrt(g*hs)/hn[1]
    #     un[0] = np.sqrt(g*hs)/hn[1]
    hn_half = get_h_half(hn)# i in Eint
    
    h = update_h(hn,un,dx,dt)# i in M
    if hs<=0 : 
        source =0
        hs =0
    else :
        source = np.sqrt(g*hs)*dt/dx
        hs -=  (np.sqrt(g*hs)*dt)/(20)
    h[0] += source
    
    h_half = get_h_half(h)  # i in Eint
    F_E = get_F_E(hn,un)    # i in E
    F_i = get_F_i(F_E)      # i in M
    uin = get_u_ihat(F_i,un)# i in M

    kvsv = get_kappa_vsv(h_half,kappa,mu)
    dt = get_dt(F_E,hn,dx,c_num,g=g)
    u_m = np.insert(u[:-2],0,0)
    viscos_term = 4*mu/dx*(h[1:]*(u[1:]-u[:-1])/dx - h[:-1]*(u[:-1]-u_m)/dx)
    hu = hn_half*un[:-1] - dt/dx*(F_i[1:]*uin[1:]-F_i[:-1]*uin[:-1]
                                  +0.5*g*(h[1:]*h[1:]-h[:-1]*h[:-1])
                                  +g*h_half*(z[1:]-z[:-1])) + viscos_term*dt
    if kappa==0:kvsv = kvsv + 1e-8
    u[:-1] = hu/(kvsv + h_half)
    u[-1] = 0
    return h,u,dt,hs

L_x = 1e+2
g = 9.81

N_x = 1000
N_t = 5000
err = 1e-4
dx = L_x/(N_x - 1)
x = np.linspace(0,L_x,N_x)

def sim_max_u_x(cSt,N_t,Draw = False):
    h = np.ones(N_x)*2
    slope = lambda x,b,L,c: ((x>=b)&(x<c))*(x-b)*(L/(L_x-b-(L_x-c))) + L*(x>=c)

    u = np.zeros(N_x)
    z = np.ones(N_x)*0
    s = slope(x,20,2,70)
    z = z+s
    z_ = np.zeros(N_x)

    hs = 2.4
    h = np.where(h>=z,h-z+err,0)
    dt = 0.001
    h_list = []
    u_list = []
    t_list = []
    hs_list = []
    t = 0
    mu = 1e-6*  cSt
    for i in range(N_t):
        hn = h.copy()
        un = u.copy()
        h_list.append(h)
        u_list.append(u)
        t_list.append(t)
        hs_list.append(hs)
        t +=dt
        h,u,dt,hs =shallow_water_viscous_sorce(hn,un,z,dx,dt,0.3,mu=mu,hs=hs)
    max_ux = np.max(u_list,axis=0)
    max_uxind = np.argmax(max_ux)
    max_x = x[max_uxind]
    
    np.array(u_list)[:,max_uxind]
    max_t = t_list[np.argmax(np.array(u_list)[:,max_uxind])]
    if Draw == True:
        plt.plot(h +z)
        plt.plot(z)
        plt.plot(u)
        
    return max_ux[2:-2],max_x,max_t, t_list[-1]