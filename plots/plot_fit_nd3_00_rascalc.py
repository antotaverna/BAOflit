import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
import matplotlib.transforms as mtransforms



fig=plt.figure(figsize=(8,5))
#fig.subplots_adjust(top = 0.9)
gs = gridspec.GridSpec(1,1, wspace=0.3, hspace=0.2)#, width_ratios=[1, 1])
ax1 = plt.subplot(gs[0,0])
#ax2 = plt.subplot(gs[1,0])
#ax3 = plt.subplot(gs[2,0])

def f1(a1,a2,a3,x):
    return a1/(x**2) + a2/x + a3


file = np.genfromtxt('../../RascalC/output/nd3_00/pycorr_z0_nd3_00_xi_jkn_8_3_r_25_150.dat')
cov = np.genfromtxt('../../RascalC/output/nd3_00/cov_matrix_RascalC_z0_nd3_00_r_25_150.dat')
fit = np.genfromtxt('../output/2pcf_auto_z0/nd3_00/rascalc_r_25_150/r_25_150_rascalc_model_mean.dat')

r = file[:,0]
xi = file[:, 1]
err = np.diag(cov)**0.5*r**2

# r vs xi
ax1.plot(r, r**2 * xi, 'o', color='grey') #all
ax1.errorbar(r, r**2*xi, yerr=err, color='grey',linestyle='None',label='error RascalC')
y=r**2*xi
#ax1.fill_between(r, y-err2, y+err2, alpha=0.1, color='C'+str(idx))    
#ax1.legend()
ax1.set_xlim(20,180)
ax1.set_ylim(-100,100)
ax1.set_title("Auto-corr [nd = 0.001]")
#ax1.set_xticklabels([''])
ax1.set_ylabel(r'$r^2 \xi(r)$')
ax1.set_xlabel(r'$r \, [h^{-1}Mpc]$')
ax1.text(140,80,"r [25-150] Mpc/h", bbox=dict(facecolor='none', edgecolor='black', boxstyle='round,pad=1'))

ii=np.where((fit[:,0]<180) & (fit[:,0]>20)) 
rfit = fit[:,0][ii]
xifit = fit[:,1][ii]
ax1.plot(rfit,rfit**2 * xifit, label='fit model_mean', marker='.')
fit = np.genfromtxt('../output/2pcf_auto_z0/nd3_00/rascalc_r_25_150/r_25_150_rascalc_model_maxlike.dat')
ii=np.where((fit[:,0]<180) & (fit[:,0]>20)) 
rfit = fit[:,0][ii]
xifit = fit[:,1][ii]
ax1.plot(rfit,rfit**2 * xifit, label='fit maxlike_',  marker='.',color='g',linestyle='dotted')

ax1.legend(loc = 'lower left')

#plt.savefig('../../plots_nden/test_fit_model_nd3_00.png')

plt.show()

