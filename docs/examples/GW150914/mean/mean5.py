from pylab import *
from scipy import stats
from scipy import optimize
import arviz as az
import h5py
import pandas as pd
import seaborn as sns

import ringdown
from joblib import Parallel, delayed

def read_strain(file, dname):
    with h5py.File(file, 'r') as f:
        t0 = f['meta/GPSstart'][()]
        T = f['meta/Duration'][()]
        h = f['strain/Strain'][:]
    
        dt = T/len(h)
    
        raw_strain = ringdown.Data(h, index=t0 + dt*arange(len(h)), ifo=dname)
        
        return raw_strain
    
h_raw_strain = read_strain('H-H1_GWOSC_16KHZ_R1-1126259447-32.hdf5', 'H1')
l_raw_strain = read_strain('L-L1_GWOSC_16KHZ_R1-1126259447-32.hdf5', 'L1')

T = 0.2
srate = 2048

def set_data(M_est,chi_est,t_init):
    fit1 = ringdown.Fit(model='mchi_aligned', modes=[(1, -2, 2, 2, 1)])
    fit1.add_data(h_raw_strain)
    fit1.add_data(l_raw_strain)
    t_unit=M_est*2950./2/299792458
    ts_ins=0.125
    fit1.set_target(1126259462.4083147-ts_ins, ra=1.95, dec=-1.27, psi=0.82, duration=T+ts_ins)
    fit1.condition_data(ds=int(round(h_raw_strain.fsamp/srate)), flow=20)
    fit1.filter_data(chi_est,M_est,2,2,0)
    #fit1.filter_data(chi_est,M_est,2,2,1)
#     fit1.filter_data(chi_est,M_est,2,2,2)
#     fit1.filter_data(chi_est,M_est,2,2,3)
    fit1.set_target(1126259462.4083147+t_init*1e-3, ra=1.95, dec=-1.27, psi=0.82, duration=T)
    #fit1.condition_data(ds=1, flow=20)
    fit1.compute_acfs()
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

chispace=np.arange(0.0,0.95,0.02)
massspace=np.arange(34,240,0.5)
X, Y = np.meshgrid(massspace,chispace)
mass_max_clu=[]
spin_max_clu=[]
bayes_clu=[]
distance=[]
tssss=np.arange(56,80,1.)
for t_init in tssss:
        finalfinal=[]
        print(t_init)
        for j in chispace:
            final=Parallel(n_jobs=24)(delayed(total)(i,j,t_init) for i in massspace)
            finalfinal.append(final)
        finalfinal=np.array(finalfinal)
        bayes=np.sum(np.exp(finalfinal))
        finalfinalnorm=finalfinal.flatten()-np.max(finalfinal.flatten())
        mass_max=np.sum((X.flatten())*np.exp(finalfinalnorm)/np.sum(np.exp(finalfinalnorm)))
        spin_max=np.sum((Y.flatten())*np.exp(finalfinalnorm)/np.sum(np.exp(finalfinalnorm)))
        mass_max_clu.append(mass_max)
        spin_max_clu.append(spin_max)
        bayes_clu.append(bayes)
np.savetxt('time_rest/mass5',mass_max_clu)
np.savetxt('time_rest/spin5',spin_max_clu)
np.savetxt('time_rest/tinit5',tssss)
np.savetxt('time_rest/bayes5',bayes_clu)
