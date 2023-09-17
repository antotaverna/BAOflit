import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
import matplotlib.transforms as mtransforms



fig=plt.figure(figsize=(8,7))
#fig.subplots_adjust(top = 0.9)
gs = gridspec.GridSpec(3,1, wspace=0.3, hspace=0.5)#, width_ratios=[1, 1])
ax1 = plt.subplot(gs[0,0])
ax2 = plt.subplot(gs[1,0])
ax3 = plt.subplot(gs[2,0])

fig.suptitle('Auto-corr [nd = 0.001]')

def f1(a1,a2,a3,x):
    return a1/(x**2) + a2/x + a3


file = np.genfromtxt('../../RascalC/output/nd3_00/pycorr_z0_nd3_00_xi_jkn_8_3_r_25_150.dat')
cov = np.genfromtxt('../../RascalC/output/nd3_00/cov_matrix_RascalC_z0_nd3_00_r_25_150_10x.dat')
fit = np.genfromtxt('../output/2pcf_auto_z0/nd3_00/rascalc_r_25_150/r_25_150_rascalc_10x_model_mean.dat')
fit_maxl = np.genfromtxt('../output/2pcf_auto_z0/nd3_00/rascalc_r_25_150/r_25_150_rascalc_10x_model_maxlike.dat')

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
ax1.set_title("r [25-150] Mpc/h")
ax1.set_ylabel(r'$r^2 \xi(r)$')
#ax1.set_xlabel(r'$r \, [h^{-1}Mpc]$')
#ax1.text(140,80,"r [25-150] Mpc/h", bbox=dict(facecolor='none', edgecolor='black', boxstyle='round,pad=1'))

ii=np.where((fit[:,0]<180) & (fit[:,0]>20)) 
rfit = fit[:,0][ii]
xifit = fit[:,1][ii]
ax1.plot(rfit,rfit**2 * xifit, label='fit model_mean', marker='.',color='b')

ii=np.where((fit_maxl[:,0]<180) & (fit_maxl[:,0]>20)) 
rfit_maxl = fit_maxl[:,0][ii]
xifit_maxl = fit_maxl[:,1][ii]
ax1.plot(rfit_maxl,rfit_maxl**2 * xifit_maxl, label='fit maxlike_10x',  marker='.',color='g',linestyle='dotted')

fit_maxl = np.genfromtxt('../output/2pcf_auto_z0/nd3_00/rascalc_r_25_150/r_25_150_rascalc_5x_model_maxlike.dat')
rfit_maxl = fit_maxl[:,0][ii]
xifit_maxl = fit_maxl[:,1][ii]
ax1.plot(rfit_maxl,rfit_maxl**2 * xifit_maxl, label='fit maxlike_5x',  marker='.',linestyle='dashed',color='y')

ax1.legend(loc = 'lower left')

######################################
######################################
######################################


file = np.genfromtxt('../../RascalC/output/nd3_00/pycorr_z0_nd3_00_xi_jkn_8_3_r_60_160.dat')
cov = np.genfromtxt('../../RascalC/output/nd3_00/cov_matrix_RascalC_z0_nd3_00_r_60_160_10x.dat')
fit = np.genfromtxt('../output/2pcf_auto_z0/nd3_00/rascalc_r_60_160/r_60_160_rascalc_10x_model_mean.dat')
fit_maxl = np.genfromtxt('../output/2pcf_auto_z0/nd3_00/rascalc_r_60_160/r_60_160_rascalc_10x_model_maxlike.dat')

r = file[:,0]
xi = file[:, 1]
err = np.diag(cov)**0.5*r**2

# r vs xi
ax2.plot(r, r**2 * xi, 'o', color='grey') #all
ax2.errorbar(r, r**2*xi, yerr=err, color='grey',linestyle='None',label='error RascalC')
y=r**2*xi
#ax1.fill_between(r, y-err2, y+err2, alpha=0.1, color='C'+str(idx))    
#ax1.legend()
ax2.set_xlim(20,180)
ax2.set_ylim(-100,100)
ax2.set_title("r [60-160] Mpc/h")
#ax1.set_xticklabels([''])
ax2.set_ylabel(r'$r^2 \xi(r)$')
#ax2.set_xlabel(r'$r \, [h^{-1}Mpc]$')
#ax2.text(140,80,"r [60-160] Mpc/h", bbox=dict(facecolor='none', edgecolor='black', boxstyle='round,pad=1'))

ii=np.where((fit[:,0]<180) & (fit[:,0]>20)) 
rfit = fit[:,0][ii]
xifit = fit[:,1][ii]
ax2.plot(rfit,rfit**2 * xifit, label='fit model_mean', marker='.',color='r')

ii=np.where((fit_maxl[:,0]<180) & (fit_maxl[:,0]>20)) 
rfit_maxl = fit_maxl[:,0][ii]
xifit_maxl = fit_maxl[:,1][ii]
ax2.plot(rfit_maxl,rfit_maxl**2 * xifit_maxl, label='fit maxlike_',  marker='.',color='m',linestyle='dotted')

ax2.legend(loc = 'lower left')


######################################
######################################
######################################

file = np.genfromtxt('../../RascalC/output/nd3_00/pycorr_z0_nd3_00_xi_jkn_8_3_r_60_160.dat')
fit_maxl_25_150 = np.genfromtxt('../output/2pcf_auto_z0/nd3_00/rascalc_r_25_150/r_25_150_rascalc_10x_model_maxlike.dat')
fit_maxl_60_160 = np.genfromtxt('../output/2pcf_auto_z0/nd3_00/rascalc_r_60_160/r_60_160_rascalc_10x_model_maxlike.dat')

r = file[:,0]
xi = file[:, 1]
err = np.diag(cov)**0.5*r**2

# r vs xi
ax3.plot(r, r**2 * xi, 'o', color='grey') #all
ax3.set_xlim(20,180)
ax3.set_ylim(-100,100)
ax3.set_title("Both fitrange")
ax3.set_ylabel(r'$r^2 \xi(r)$')
ax3.set_xlabel(r'$r \, [h^{-1}Mpc]$')
#ax2.text(140,80,"r [60-160] Mpc/h", bbox=dict(facecolor='none', edgecolor='black', boxstyle='round,pad=1'))

ii=np.where((fit_maxl_25_150[:,0]<180) & (fit_maxl_25_150[:,0]>20)) 
rfit = fit_maxl_25_150[:,0][ii]
xifit = fit_maxl_25_150[:,1][ii]
ax3.plot(rfit,rfit**2 * xifit, label='fit maxlike_25_150', marker='.',color='g',linestyle='dashed')

ii=np.where((fit_maxl_60_160[:,0]<180) & (fit_maxl_60_160[:,0]>20)) 
rfit_maxl = fit_maxl_60_160[:,0][ii]
xifit_maxl = fit_maxl_60_160[:,1][ii]
ax3.plot(rfit_maxl,rfit_maxl**2 * xifit_maxl, label='fit maxlike_60_160',  marker='.',color='m',linestyle='dashed')

ax3.legend(loc = 'lower left')



#plt.savefig('../../plots_nden/test_fit_model_nd3_00.png')

plt.show()

