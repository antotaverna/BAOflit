import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
import matplotlib.transforms as mtransforms
import pandas as pd



fig=plt.figure(figsize=(9,12))
#fig.subplots_adjust(top = 0.9)
gs = gridspec.GridSpec(3,1, wspace=0.3, hspace=0.2)#, width_ratios=[1, 1])
ax1 = plt.subplot(gs[0,0])
ax2 = plt.subplot(gs[1,0])
ax3 = plt.subplot(gs[2,0])

##############################################################################

file2 = np.genfromtxt('../../RascalC/output/first/xi_to_multipoles.dat')
file = np.genfromtxt('../../get_env/fcorr/xi_files/2pcf_first_z0_r_60_160_full_fixefAmp_002.dat')
cov_rascalc = np.genfromtxt('../../RascalC/output/first/cov_matrix_RascalC_z0_first_r_60_160_2x.dat')
fit_maxl = np.genfromtxt('../output/2pcf_auto_z0/first/r_60_160_rascalc_2x_model_maxlike.dat')

r = file[:,0]
xi = file[:, 1]
err = np.sqrt(np.diag(cov_rascalc))
err2=r**2*(err)

# r vs xi
ax1.errorbar(r, r**2*xi, yerr=err2, marker='o', color='k',linestyle='None',label='first-sample')
#ax1.plot(r, r**2 * xi, marker='o', color='gray') #all
y=r**2*xi
ax1.set_xlim(30,160)
ax1.set_ylim(-20,30)
ax1.set_ylabel(r'$r^2 \xi(r)$')
ax1.set_xlabel(r'$r \, [h^{-1}Mpc]$')
#mutipolos
ax1.plot(file2[:,0], file2[:,0]**2*file2[:,1], marker='.', color='g',linestyle='None')


rfit = fit_maxl[:,0]
xifit = fit_maxl[:,1]
ax1.plot(rfit,rfit**2 * xifit, label='fit maxlike_',color='r')
ax1.axvline(105)
ax1.legend(loc = 'lower left')



##############################################################################


file = np.genfromtxt('../../RascalC/output/nd3_00/pycorr_z0_nd3_00_xi_jkn_8_3_r_60_160.dat')
cov_rascalc = np.genfromtxt('../../RascalC/output/nd3_00/cov_matrix_RascalC_z0_nd3_00_r_60_160_10x.dat')
fit_maxl = np.genfromtxt('../output/2pcf_auto_z0/nd3_00/rascalc_r_60_160/r_60_160_rascalc_10x_model_maxlike.dat')

r = file[:,0]
xi = file[:, 1]
err = np.sqrt(np.diag(cov_rascalc))
err2=r**2*(err)

# r vs xi
ax2.errorbar(r, r**2*xi, yerr=err2, marker='o', color='C1',linestyle='None',label='nd=0.0001')
#ax2.plot(r, r**2 * xi, marker='o', color='gray') #all
y=r**2*xi
ax2.set_xlim(30,160)
ax2.set_ylim(-100,100)
ax2.set_ylabel(r'$r^2 \xi(r)$')
ax2.set_xlabel(r'$r \, [h^{-1}Mpc]$')
ax2.axvline(105)


rfit = fit_maxl[:,0]
xifit = fit_maxl[:,1]
ax2.plot(rfit,rfit**2 * xifit, label='fit maxlike_',color='C1')

ax2.legend(loc = 'lower left')


def Convert(string):
    li = list(string.split("   "))
    return li



#plt.savefig('../../plots_nden/xi_fit_rascalc.png')

plt.show()

