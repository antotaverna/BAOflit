import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
import matplotlib.transforms as mtransforms


#nsv_jack = [5,8,10,16,20,32,40,64,128]

fig=plt.figure(figsize=(8,6)) #6,14
#fig.subplots_adjust(top = 0.9)
gs = gridspec.GridSpec(1,1, wspace=0.3, hspace=0.2)#, width_ratios=[1, 1])
ax1 = plt.subplot(gs[0,0])
#ax2 = plt.subplot(gs[1,0])
#ax3 = plt.subplot(gs[2,0])

def f1(a1,a2,a3,x):
    return a1/(x**2) + a2/x + a3

nsv_jack = [5]
for idx, nsv in enumerate(nsv_jack):

    file = np.genfromtxt('../../get_env/n_density/test_jackknife/files/pycorr_z0_nd0_00_xi_jkn_'+str(nsv)+'_3.dat')
    fit = np.genfromtxt('../output/2pcf_auto_z0/nd0_00/r_70_150_njack_125_model_mean.dat')

    r = file[:,0]
    xi = file[:, 1]
    err = file[:, 2]
    err2=r**2*(err)
    err_max=max(err)

    # r vs xi
    ax1.plot(r, r**2 * xi, '.',label='2pcf njack=125') #all
    ax1.errorbar(r, r**2*xi, linestyle='none', yerr=err2, color='C'+str(idx))
    y=r**2*xi
    #ax1.fill_between(r, y-err2, y+err2, alpha=0.1, color='C'+str(idx))    
    #ax1.legend()
    #ax1.set_xlim(70,150)
    ax1.set_ylim(-60,70)
    ax1.set_title("Auto-corr [nd = Tot]")
    #ax1.set_xticklabels([''])
    ax1.set_ylabel(r'$r^2 \xi(r)$')
    ax1.set_xlabel(r'$r \, [h^{-1}Mpc]$')
    ax1.text(60,45,"r [50-170] Mpc/h", bbox=dict(facecolor='none', edgecolor='black', boxstyle='round,pad=1'))

    ii=np.where((fit[:,0]<160) & (fit[:,0]>60)) 
    rfit = fit[:,0][ii]
    xifit = fit[:,1][ii]
    #ax1.plot(rfit,rfit**2 * xifit, label='fit model', color='green')
    fit = np.genfromtxt('../output/2pcf_auto_z0/nd0_00/r_70_150_njack_125_fortran_model_mean.dat')
    rfit = fit[:,0][ii]
    xifit = fit[:,1][ii]
    ax1.plot(rfit,rfit**2 * xifit, label='fit model', color='r')

    #nd0_00 r in [70-150]
    #a1=61.12254301
    #a2=-0.1013020098
    #a3=-0.004608987533

    #ax1.text(50, -20, r'$\alpha$: 1.007795114', fontsize=12)
    #ax1.text(50, -30, r'bias: 0.2234969351', fontsize=12)
    #ax1.text(50, -40, r'$\Sigma_{nl}$: 13.47428231', fontsize=12)
    #ax1.text(50, -50, r'$a_1,a_2,a_3$: 61.12254301; -0.1013020098; -0.004608987533', fontsize=12)

    #x=np.arange(50,170,1)
    #y=f1(a1,a2,a3,x)
    #ax1.plot(x,x**2*y, label='A(s)', color='red')#, marker='.')
    ax1.legend()
#plt.savefig('../../plots_nden/test_fit_model_nd0_00.png')


plt.show()
##############################################################################
stop
nsv_jack = [8]
for idx, nsv in enumerate(nsv_jack):
    file = np.genfromtxt('../../get_env/n_density/test_jackknife/files/pycorr_z0_nd0_00_xi_jkn_'+str(nsv)+'_3.dat')
    fit = np.genfromtxt('../output/2pcf_auto_z0/nd0_00/r_70_150_njack_512_model_mean.dat')

    r = file[:,0]
    xi = file[:, 1]
    err = file[:, 2]
    err2=r**2*(err)
    err_max=max(err)

    # r vs xi
    ax2.plot(r, r**2 * xi, '.',label='2pcf njack=512') #all
    ax2.errorbar(r, r**2*xi, linestyle='none', yerr=err2, color='C'+str(idx))
    y=r**2*xi
    #ax1.set_xlim(70,150)
    ax2.set_ylim(-60,70)
    ax2.set_ylabel(r'$r^2 \xi(r)$')
    ax2.set_xlabel(r'$r \, [h^{-1}Mpc]$')
    ax2.text(60,45,"r [50-170] Mpc/h", bbox=dict(facecolor='none', edgecolor='black', boxstyle='round,pad=1'))

    ii=np.where((fit[:,0]<160) & (fit[:,0]>60)) 
    rfit = fit[:,0][ii]
    xifit = fit[:,1][ii]
    ax2.plot(rfit,rfit**2 * xifit, label='fit model', color='green')

    ax2.legend()

##############################################################################

nsv_jack = [5]
for idx, nsv in enumerate(nsv_jack):
    file = np.genfromtxt('../../get_env/n_density/test_jackknife/files/pycorr_z0_nd0_00_xi_jkn_'+str(nsv)+'_3_r_20_200.dat')
    fit = np.genfromtxt('../output/2pcf_auto_z0/nd0_00/r_20_200_njack_125_model_mean.dat')

    r = file[:,0]
    xi = file[:, 1]
    err = file[:, 2]
    err2=r**2*(err)
    err_max=max(err)

    # r vs xi
    ax3.plot(r, r**2 * xi, '.',label='2pcf njack=512') #all
    ax3.errorbar(r, r**2*xi, linestyle='none', yerr=err2, color='C'+str(idx))
    y=r**2*xi
    #ax1.set_xlim(70,150)
    ax3.set_ylim(-60,70)
    ax3.set_ylabel(r'$r^2 \xi(r)$')
    ax3.set_xlabel(r'$r \, [h^{-1}Mpc]$')
    ax3.text(20,45,"r [20-200] Mpc/h", bbox=dict(facecolor='none', edgecolor='black', boxstyle='round,pad=1'))

    ii=np.where((fit[:,0]<150) & (fit[:,0]>60)) 
    rfit = fit[:,0][ii]
    xifit = fit[:,1][ii]
    ax3.plot(rfit,rfit**2 * xifit, label='fit model', color='green')

    ax3.legend()

#plt.savefig('../../../plots_nden/test_fit_model_nd3_00.png')

plt.show()

