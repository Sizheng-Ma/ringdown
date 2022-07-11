from pylab import *
from scipy import stats
from scipy import optimize
import arviz as az
import h5py
import pandas as pd
import seaborn as sns
import lal
from scipy.interpolate import interp1d

import ringdown
from joblib import Parallel, delayed

def planck_window(t, tMin=0., tMax=1., tol=.005, rolloff=False):
    """
A smooth function f mapping [tMin, tMax] to [0, 1], with
f(tMin)=0, f(tMax)=1, all derivatives -> 0 as t->0 or t->tMax.
f(t) = 1/(1 + exp(z)) where z = 1/t - 1/(1-t) for the case
tMin=0, tMax=1.
tol controls the minimum deviation of t from 0 or tMax, with
a tol of .005 giving abs(z) < ~200, where exp(z) is 1.e86 or 1.e-86

If rolloff, instead has f(0)=1, f(tMax)=0.
    """
    if rolloff:
        return helper.planck_window(-t, tMin=-tMax, tMax=-tMin, tol=tol)

    if tMax==1. and tMin==0.:
        # If t <= .005, z >= 200. - 1./199. and exp(z) > 1.e86 ~ infinity
        # Similar for t > .995 but exp(z) < 1.e-86 ~ 0
        safe = (t>tol)*(t<(1.-tol))
        safeT = safe*t + (1. - safe)*.5 # use t = 0.5 temporarily for unsafe values
        safeZ = 1./safeT - 1./(1. - safeT)
        return safe*1./(1. + np.exp(safeZ)) + (t >= (1.-tol))

    return planck_window((t-tMin)/(tMax-tMin))

def load_data(iota,beta,t_unit):
    h2m2=np.loadtxt('Y_l2_m-2.dat')
    h22=np.loadtxt('Y_l2_m2.dat')
    time=h22[:,0]-3692.7480095302817

    h2=(h22[:,1]+1j*h22[:,2])*lal.SpinWeightedSphericalHarmonic(iota,beta,-2,2,2)\
    +(h2m2[:,1]+1j*h2m2[:,2])*lal.SpinWeightedSphericalHarmonic(iota,beta,-2,2,-2)
#     h2=(h22[:,1]+1j*h22[:,2])
    ts=np.arange(time[0]+50,time[-1],0.1)
    dtcut=ts[1]-ts[0]
    h2int=interp1d(time,h2)(ts)
    partition=2
    padlen=2**(3+int(np.ceil(np.log2(len(h2int)))))-len(h2int)
    ini_filter=planck_window(ts, tMin=ts[0], tMax=ts[0]+400)
    h2pad=np.pad(h2int*ini_filter,(padlen//partition,padlen-(padlen//partition)),'constant', constant_values=(0, 0))
    end1=ts[-1]+(padlen-(padlen//partition))*dtcut
    end2=ts[0]-(padlen//partition)*dtcut

    tpad=np.pad(ts,(padlen//partition,padlen-(padlen//partition)),'linear_ramp', end_values=(end2, end1))
#     tpad-=tpad[0]
    return tpad*t_unit,(h2pad)

def NR_injection_into_Bilby(time,tpad,h2pad,M_tot,solar_to_distance,dis, **waveform_kwargs):
    hplus_interp_func = interp1d(tpad, np.real(h2pad), bounds_error=False, fill_value=0)
    hcross_interp_func = interp1d(tpad, -np.imag(h2pad), bounds_error=False, fill_value=0)
    hplus = hplus_interp_func(time)/dis#*M_tot*solar_to_distance
    hcross = hcross_interp_func(time)/dis#*M_tot*solar_to_distance
    return {'plus': hplus, 'cross': hcross}

def inject_gau(distance,M_tot):

    solar_to_distance=2950./2
    solar_to_time=2950./2/299792458
    t_unit=M_tot*solar_to_time
    Mpc=30860000000000004*1000104.4887813144596
    dis=distance
    

    duration = 4
    sampling_frequency = 2048.*4

    
    iota=np.pi/3
    beta=np.pi/3*0

    tpad,h2pad=load_data(iota,beta,t_unit)
    
    tfinal=np.arange(-duration/2,duration/2,1./sampling_frequency)
    h_int=NR_injection_into_Bilby(tfinal,tpad,h2pad,M_tot,solar_to_distance,dis)
    
    rng = np.random.default_rng(12345)
    signal=h_int['plus']+ rng.normal(0, 1, len(tfinal))

    
    fpsi422=np.fft.rfft(h_int['plus'],norm='ortho')
    ffreq=np.fft.rfftfreq(len(signal),d=(tfinal[1]-tfinal[0])/(Mf*t_unit))*2*np.pi
    return tfinal,signal,ffreq,fpsi422,t_unit


def set_data(M_est,chi_est,t_init):
    srate = 2048
    acf = ringdown.AutoCovariance(zeros_like(h_raw_strain),delta_t=h_raw_strain.delta_t)
    acf.iloc[0] = 1
    fit1 = ringdown.Fit(model='mchi_aligned', modes=[(1, -2, 2, 2, 1)])
    fit1.add_data(h_raw_strain, acf=acf)
    
    t_unit=M_est_total*Mf*2950./2/299792458
    t0=t_init*t_unit
    T = 0.08
    fit1.set_target(t0, duration=T)
    fit1.condition_data(ds=int(round(h_raw_strain.fsamp/srate)),flow=20,preserve_acfs=True)
    
    fit1.filter_data(chi_est,M_est,2,2,0)
    fit1.filter_data(chi_est,M_est,2,2,1)
    wd1 = fit1.analysis_data
    return fit1,wd1

def compute_likelihood(fit1,wd1):
    Ls=fit1.obtain_L()
    strains=np.array([s.values for s in wd1.values()])
    times=np.array([array(d.time) for d in wd1.values()])
    likelihood=0
    for i in range(len(strains)):
        norm=np.sqrt(np.sum(abs(np.dot(np.linalg.inv(Ls[i]),Ls[i])-np.identity(len(Ls[i])))**2))
        if abs(norm)>1e-8:
            raise ValueError("inverse of L is not correct")
        whitened=np.dot(np.linalg.inv(Ls[i]),strains[i])
        likelihood-=0.5*np.dot(whitened,whitened)
    return likelihood

def total(M_est,chi_est,t_init):
    fit1,wd1=set_data(M_est,chi_est,t_init)
    likelihood=compute_likelihood(fit1,wd1)
    return likelihood

Mf=0.952032939704
M_est_total=68.5/Mf
tfinal,signal,ffreq,fpsi422,t_unit=inject_gau(distance=0.008,M_tot=M_est_total)
h_raw_strain =ringdown.Data(signal, index=tfinal)

chispace=np.arange(0.1,0.95,0.02)
massspace=np.arange(34,100,0.5)
X, Y = np.meshgrid(massspace,chispace)
distance=[]
mass_max_clu=[]
spin_max_clu=[]
for t_init in np.arange(-5,60,0.5):
        finalfinal=[]
        for j in chispace:
            final=Parallel(n_jobs=24)(delayed(total)(i,j,t_init) for i in massspace)
            finalfinal.append(final)
        finalfinal=np.array(finalfinal)
        finalfinalnorm=finalfinal.flatten()-np.max(finalfinal.flatten())
        mass_max=np.sum((X.flatten())*np.exp(finalfinalnorm)/np.sum(np.exp(finalfinalnorm)))
        spin_max=np.sum((Y.flatten())*np.exp(finalfinalnorm)/np.sum(np.exp(finalfinalnorm)))
        mass_max_clu.append(mass_max)
        spin_max_clu.append(spin_max)
np.savetxt('time_rest/mass_overtone',mass_max_clu)
np.savetxt('time_rest/spin_overtone',spin_max_clu)

