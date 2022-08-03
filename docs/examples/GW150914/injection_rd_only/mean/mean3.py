import numpy as np
import qnm
import ringdown as rd
import scipy.signal as sig
import matplotlib.pyplot as pl
from pylab import *
from joblib import Parallel, delayed

def h22new(a,mass,n,t):
    t_unit1=mass*2950./2/299792458
    htot=[]
#     htot+=[np.exp(0*t)]
    for i in range(n+1):
        grav = qnm.modes_cache(s=-2,l=2,m=2,n=i)
        omega, A1, C = grav(a=a)
        htot+=[np.exp(-1j*(omega)/t_unit1*abs(t))]
    return np.array(htot).T

def construct_qnm(n,chi,M_est,time):
    t_unit1=M_est*2950./2/299792458
    datalist=h22new(chi,M_est,n,time)
    construct=np.sum(datalist*mag_from_nr[:(n+1)],axis=1)#*np.heaviside(time+0*t_unit1,1)
    construct_new=np.nan_to_num(construct)
    return np.real(construct_new)

def get_signal_overtone(n,chi,M_est,time):
    t_unit1=M_est*2950./2/299792458
    grav = qnm.modes_cache(s=-2,l=2,m=2,n=n)
    omega, A1, C = grav(a=chi)
    construct=np.exp(-1j*(omega)/t_unit1*abs(time))*mag_from_nr[n]
    construct_new=np.nan_to_num(construct)
    return np.real(construct_new)

def get_signal(n,chi,M_est,time):
    t_unit1=M_est*2950./2/299792458
    construct_new=construct_qnm(n,chi,M_est,time)
    fpsi422=np.fft.rfft(construct_new,norm='ortho')
    ffreq=np.fft.rfftfreq(len(construct_new),d=(time[1]-time[0])/t_unit1)*2*np.pi
    return rd.Data(construct_new, index=time),ffreq,fpsi422

def set_data(M_est,chi_est,t_init):
    fit1 = rd.Fit(model='mchi_aligned', modes=[(1, -2, 2, 2, 1)])
    fit1.add_data(data)
    t_unit=M_est*2950./2/299792458
    ts_ins=0.125
    fit1.set_target(0-ts_ins, ra=1.95, dec=-1.27, psi=0.82, duration=T+ts_ins)
    #fit1.condition_data(ds=int(round(data.fsamp/srate)), flow=20)
    fit1.filter_data(chi_est,M_est,2,2,0)
#     fit1.filter_data(chi_est,M_est,2,2,1)
#     fit1.filter_data(chi_est,M_est,2,2,2)
#     fit1.filter_data(chi_est,M_est,2,2,3)
    fit1.set_target(0+t_init*1e-3, ra=1.95, dec=-1.27, psi=0.82, duration=T)
    #fit1.condition_data(ds=1, flow=20)
    wd1 = fit1.analysis_data
    return fit1,wd1

def compute_L_inv(fit1):
    #fit1.compute_acfs()
    #Ls=fit1.obtain_L()
    Ls=np.array([np.identity(fit.n_analyze)])
    Ls_inv=[]
    for i in range(len(Ls)):
        norm=np.sqrt(np.sum(abs(np.dot(np.linalg.inv(Ls[i]),Ls[i])-np.identity(len(Ls[i])))**2))
        if abs(norm)>1e-8:
            raise ValueError("inverse of L is not correct")
        Ls_inv.append(np.linalg.inv(Ls[i]))
    return np.array(Ls_inv)

def compute_likelihood(wd1,Ls_inv):
    strains=np.array([s.values for s in wd1.values()])
    times=np.array([np.array(d.time) for d in wd1.values()])
    likelihood=0
    for i in range(len(strains)):
        whitened=np.dot(Ls_inv[i],strains[i])
        likelihood-=0.5*np.dot(whitened,whitened)
    return likelihood

def total(Ls_inv,M_est,chi_est,t_init):
    fit1,wd1=set_data(M_est,chi_est,t_init)
    likelihood=compute_likelihood(wd1,Ls_inv)
    return likelihood

# define sampling rate and duration (make these powers of two) 
fsamp = 8192
duration = 4
t0 = 0

delta_t = 1/fsamp
tlen = int(round(duration / delta_t))
epoch = t0 - 0.5*tlen*delta_t

time = np.arange(tlen)*delta_t + epoch

chi=0.692085186818
M_est=68.5
t_unit=M_est*2950./2/299792458
dis=1
mag_from_nr=dis*np.array([  0.13657646 +0.95055905j,   3.14657959 -2.71266413j,
       -10.84170677 +3.607406j  ,  22.67958241 -8.01885436j,
       -30.44795174+19.91714543j,  22.09429848-26.19835642j,
        -7.43071591+15.89029183j,   1.0005728  -3.63046079j])
signal,ffreq,fpsi422 = get_signal(1,chi,M_est,time)
rng = np.random.default_rng(12345)
data = signal + rng.normal(0, 1, len(signal))

T = 0.2
srate = 2048

chispace=np.arange(0.0,0.95,0.02)
massspace=np.arange(34,240,0.5)
X, Y = np.meshgrid(massspace,chispace)
mass_max_clu=[]
spin_max_clu=[]
bayes_clu=[]
tssss=np.arange(20,32,0.5)
for t_init in tssss:
        print(t_init)
        fit,_=set_data(massspace[0],chispace[0],t_init)
        Ls_inv=compute_L_inv(fit)
        finalfinal=[]
        for j in chispace:
            final=Parallel(n_jobs=24)(delayed(total)(Ls_inv,i,j,t_init) for i in massspace)
            finalfinal.append(final)
        finalfinal=np.array(finalfinal)
        finalfinal+=800
        bayes=np.sum(np.exp(finalfinal))
        finalfinalnorm=finalfinal.flatten()-np.max(finalfinal.flatten())
        mass_max=np.sum((X.flatten())*np.exp(finalfinalnorm)/np.sum(np.exp(finalfinalnorm)))
        spin_max=np.sum((Y.flatten())*np.exp(finalfinalnorm)/np.sum(np.exp(finalfinalnorm)))
        mass_max_clu.append(mass_max)
        spin_max_clu.append(spin_max)
        bayes_clu.append(bayes)
np.savetxt('time_rest/mass3',mass_max_clu)
np.savetxt('time_rest/spin3',spin_max_clu)
np.savetxt('time_rest/tinit3',tssss)
np.savetxt('time_rest/bayes3',bayes_clu)
