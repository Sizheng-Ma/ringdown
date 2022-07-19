from pylab import *
from scipy import stats
from scipy import optimize
import arviz as az
import h5py
import pandas as pd
import seaborn as sns

import ringdown
from joblib import Parallel, delayed


T = 2
srate = 2048

tfinal=np.arange(-T/2,T/2,1./srate)
rng = np.random.default_rng(12345)
signal=rng.normal(0, 1, len(tfinal))
h_raw_strain =ringdown.Data(signal, index=tfinal)

def set_data(M_est,chi_est,t_init):
    acf = ringdown.AutoCovariance(zeros_like(signal),delta_t=h_raw_strain.delta_t)
    acf.iloc[0] = 1
    fit1 = ringdown.Fit(model='mchi_aligned', modes=[(1, -2, 2, 2, 1)])
    fit1.add_data(h_raw_strain, acf=acf)
    
    
    t_unit=M_est*2950./2/299792458
    t0=t_init*1e-3
    fit1.filter_data(chi_est,M_est,2,2,0)
#     fit1.filter_data(chi_est,M_est,2,2,1)
    fit1.set_target(t0, duration=0.08)
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

#t_init=0.8

chispace=np.arange(0.1,0.95,0.02)
massspace=np.arange(34,240,0.5)
X, Y = np.meshgrid(massspace,chispace)
mass_max_clu=[]
spin_max_clu=[]
distance=[]
tssss=np.arange(48,64,0.5)
for t_init in tssss:
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
np.savetxt('time_rest/mass5_overtone',mass_max_clu)
np.savetxt('time_rest/spin5_overtone',spin_max_clu)
np.savetxt('time_rest/tinit5_overtone',tssss)
