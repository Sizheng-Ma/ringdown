import numpy as np
import lal
from scipy.interpolate import interp1d
from gwpy.timeseries import TimeSeries
import qnm
from pylab import *

import arviz as az
import pandas as pd
import seaborn as sns
import bilby
import matplotlib.pyplot as pl
from joblib import Parallel, delayed
import ringdown

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
    ts=np.arange(time[0]+3100,time[-1],0.1)
#     ts=np.arange(time[0]+50,time[-1],0.1)
    dtcut=ts[1]-ts[0]
    h2int=interp1d(time,h2)(ts)
    partition=2
    padlen=2**(3+int(np.ceil(np.log2(len(h2int)))))-len(h2int)
    ini_filter=planck_window(ts, tMin=ts[0], tMax=ts[0]+400)
    h2pad=np.pad(h2int,(padlen//partition,padlen-(padlen//partition)),'constant',constant_values=(0, 0))
    end1=ts[-1]+(padlen-(padlen//partition))*dtcut
    end2=ts[0]-(padlen//partition)*dtcut

    tpad=np.pad(ts,(padlen//partition,padlen-(padlen//partition)),'linear_ramp',end_values=(end2, end1))
#     tpad-=tpad[0]
    return tpad*t_unit,(h2pad)

def NR_injection_into_Bilby(time,tpad,h2pad,dis, **waveform_kwargs):
    hplus_interp_func = interp1d(tpad, np.real(h2pad), bounds_error=False,fill_value=0)
    hcross_interp_func = interp1d(tpad, -np.imag(h2pad), bounds_error=False,fill_value=0)
    hplus = hplus_interp_func(time)/dis#*M_tot*solar_to_distance
    hcross = hcross_interp_func(time)/dis#*M_tot*solar_to_distance
    return {'plus': hplus, 'cross': hcross}

def sim_ligo_noise(duration,sampling_frequency,distance,tpad,h2pad):
    injection_parameters = dict(ra=0,dec=0,psi=0,geocent_time=0.,mass_1=140,mass_2=130\
                                ,dis=distance,tpad=tpad,h2pad=h2pad)
    
    waveform = bilby.gw.waveform_generator.WaveformGenerator(
        duration=duration, sampling_frequency=sampling_frequency,
        time_domain_source_model=NR_injection_into_Bilby,
        parameters=injection_parameters,
        start_time=-duration/2)
    
    ifos = bilby.gw.detector.InterferometerList(['H1', 'L1'])
    ifos.set_strain_data_from_power_spectral_densities(
    sampling_frequency=sampling_frequency, duration=duration,
    start_time=0)
    
    ifos.inject_signal(waveform_generator=waveform,
    parameters=injection_parameters);
    
    signal=ifos[0].strain_data.time_domain_strain
    print(ifos[0].meta_data['matched_filter_SNR'])
    tfinal=np.array([-duration/2+i/sampling_frequency for i in range(len(ifos[0].strain_data.time_domain_strain))])
    
    return tfinal,signal

def inject_gau(distance,M_tot):

    solar_to_distance=2950./2
    solar_to_time=2950./2/299792458
    t_unit=M_tot*solar_to_time
    Mpc=30860000000000004*1000104.4887813144596
    dis=distance
    

    duration = 32
    sampling_frequency = 16384.

    
    iota=np.pi/3*1
    beta=np.pi/3*0

    tpad,h2pad=load_data(iota,beta,t_unit)
    
    tfinal,signal=sim_ligo_noise(duration,sampling_frequency,distance,tpad,h2pad)
    
    return ringdown.Data(signal, index=tfinal+0.015625),t_unit

def set_data(M_est,chi_est,t_init):
    srate = 2048
    fit1 = ringdown.Fit(model='mchi_aligned', modes=[(1, -2, 2, 2, 1)])
    fit1.add_data(h_raw_strain)

    t_unit=M_est_total*Mf*2950./2/299792458
    t0=t_init*t_unit
    T = 0.2
    fit1.set_target(t_init*1e-3, duration=T)
    fit1.condition_data(ds=int(round(h_raw_strain.fsamp/srate)),flow=20)

    fit1.filter_data(chi_est,M_est,2,2,0)
    wd1 = fit1.analysis_data
    return fit1,wd1

def compute_L_inv(fit1):
    fit1.compute_acfs()
    Ls=fit1.obtain_L()
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

Mf=0.952032939704
chi=0.692085186818
M_est_total=68.5/Mf
disdis=3.6
t_init=0
h_raw_strain,t_unit=inject_gau(distance=disdis*1e20,M_tot=M_est_total)

chispace=np.arange(0.0,0.95,0.005)
massspace=np.arange(34,100,0.1)
finalfinal=[]

fit,_=set_data(massspace[0],chispace[0],t_init)
Ls_inv=compute_L_inv(fit)
for j in chispace:
    final=Parallel(n_jobs=24)(delayed(total)(Ls_inv,i,j,t_init) for i in massspace)
    finalfinal.append(final)
finalfinal=np.array(finalfinal)
np.savetxt('rest/t_'+str(t_init)+'_'+str(disdis),finalfinal)
