from pylab import *

import arviz as az
import h5py
import pandas as pd
import seaborn as sns

import ringdown

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

M_est = 68.5
chi_est = 0.69

T = 0.2
srate = 2048

tinit=0.84

fit = ringdown.Fit(model='mchi_aligned', modes=[(1, -2, 2, 2,1)])
fit.add_data(h_raw_strain)
fit.add_data(l_raw_strain)
t_unit=M_est*2950./2/299792458
ts_ins=0.125

fit.set_target(1126259462.4083147+tinit*1e-3, ra=1.95, dec=-1.27, psi=0.82,duration=T)
fit.condition_data(ds=int(round(h_raw_strain.fsamp/srate)), flow=20)
fit.filter_data(chi_est,M_est,2,2,0)

fit.compute_acfs()
wd = fit.whiten(fit.analysis_data)
fit.update_prior(A_scale=5e-21, M_min=35.0, M_max=140.0, cosi_max=-0.99,flat_A=True)
fit.run(target_accept=0.95)

mass=fit.result.posterior.M.values.flatten()
spin=fit.result.posterior.chi.values.flatten()
A1=fit.result.posterior.A[:,:,0].values.flatten()

np.savetxt('220_1/mass'+str(tinit),mass)
np.savetxt('220_1/spin'+str(tinit),spin)
np.savetxt('220_1/A1'+str(tinit),A1)
