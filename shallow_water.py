import numpy as np
import matplotlib.pyplot as plt


L_x = 3e+3
g = 9.81

N_x = 1000
dx = L_x/(N_x - 1)
dt = 0.05
N_t = 5000
err = 1e-4
x = np.linspace(0,L_x,N_x)
x_todraw = np.linspace(-50,L_x+50,N_x)
# cSts ={}
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



def shallow_water_viscous(h,u,z,dx,dt,kappa,mu,c_num=0.7,g=9.81):
    hn = h.copy()
    un = u.copy()
    hn_half = get_h_half(hn)# i in Eint
    
    h = update_h(hn,un,dx,dt)# i in M
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
    return h,u,dt


def set_u_h():
    h = np.ones(N_x)*0.5
    drop = np.exp(-((x-L_x*(0.7))**2/(2*(0.05E+2)**2)))*1.5
    h_ = 21
    # h = h+ drop
    h[:int(200//dx)]= h_
    slope = lambda x,b,L: (x>=b)*(x-b)*(L/(L_x-b))
    dam  = lambda x,a,b,L: ((x>=a)&(x<=b))*L

    u = np.zeros(N_x)
    z = np.ones(N_x)*1
    s = slope(x,0,1.5)
    d = dam(x,20,28,1.6)
    z = z 
    z_to_draw = z + dam(x_todraw,-50,0,h_*1.1)+  dam(x_todraw,L_x,L_x+50,h_*1.1)
    z_to_draw2 = z + dam(x_todraw,-50,0,h_*1.1)
    z_ = np.zeros(N_x)

    h = np.where(h>=z,h-z,0)
    return u,h,z
    
def set_figure(cSt):
    h = np.ones(N_x)*0.5
    drop = np.exp(-((x-L_x*(0.7))**2/(2*(0.05E+2)**2)))*1.5
    h_ = 21
    # h = h+ drop
    h[:int(200//dx)]= h_
    slope = lambda x,b,L: (x>=b)*(x-b)*(L/(L_x-b))
    dam  = lambda x,a,b,L: ((x>=a)&(x<=b))*L

    u = np.zeros(N_x)
    z = np.ones(N_x)*1
    s = slope(x,0,1.5)
    d = dam(x,20,28,1.6)
    z = z 
    z_to_draw = z + dam(x_todraw,-50,0,h_*1.1)+  dam(x_todraw,L_x,L_x+50,h_*1.1)
    z_to_draw2 = z + dam(x_todraw,-50,0,h_*1.1)
    z_ = np.zeros(N_x)

    h = np.where(h>=z,h-z,0)
    fig, (axh,axu) = plt.subplots(2, 1,facecolor ='white')
    line_u,=axu.plot(x,u)
    line_h,=axh.plot(x,h+z)
    axh.fill_between(x_todraw,z_to_draw,z_,color = 'tab:orange',zorder=2)
    axu.set_xlabel('Horizontal distance[m]',fontsize=13)
    axh.set_xlim(-50,L_x+50)
    axh.set_ylim(-0,h_*1.1)
    axh.set_ylabel('Height[m]',fontsize=13)
    axu.set_xlim(-50,L_x+50)

    axu.set_ylabel('Velocity[m/s]',fontsize=13)
    axh.set_title(r'$\nu$ = {} cSt, time = {:.02f}s'.format(cSt,0.),fontsize=15)
    plt.tight_layout()
    return fig, axh, axu,h,z,z_


def Shallow_water_check(cSt, N_t,check_time,Draw= False):
    u,h,z= set_u_h()
    h_list = []
    u_list = []
    t_list = []
    dt = 0.001
    t = 0
    # cSt = 1.0034
    mu = 1e-6 * cSt 
    kappa = 0.3 #논문에서 h/L 의 10배정도가 Nonslip boundry에 근접했음, 그래서 20배를 해줘서 Non slip boundary 보장
    # h,u,dt =shallow_water_viscous(u,h,z,dx,dt,0.01,0.1)
    count = 0
    for i in range(N_t):
        hn = h.copy()
        un = u.copy()
        h_list.append(h)
        u_list.append(u)
        t_list.append(t)
        t +=dt
        h,u,dt =shallow_water_viscous(hn,un,z,dx,dt,kappa=kappa,mu=mu,c_num=0.7,g = 9.81)
        # if (t >= check_time) & count == 0:
        #     count+=1
        #     h_check,u_check = h.copy(),u.copy()
        # h = np.where(h>=z,h)
    search_index = np.searchsorted(t_list,check_time)
    h_check = np.array(h_list)[search_index]
    u_check = np.array(u_list)[search_index]
    if Draw == True:
        plt.plot(h +z)
        plt.plot(z)
        plt.plot(u)
    return h_list, u_list, t_list, h_check, u_check