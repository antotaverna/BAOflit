import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
import matplotlib.transforms as mtransforms


#nsv_jack = [5,8,10,16,20,32,40,64,128]
nsv_jack = [5,8]

fig=plt.figure(figsize=(9,12))
#fig.subplots_adjust(top = 0.9)
gs = gridspec.GridSpec(3,1, wspace=0.3, hspace=0.2)#, width_ratios=[1, 1])
ax1 = plt.subplot(gs[0,0])
ax2 = plt.subplot(gs[1,0])
ax3 = plt.subplot(gs[2,0])

def f1(a1,a2,a3,x):
    return a1/(x**2) + a2/x + a3

for idx, nsv in enumerate(nsv_jack):
    if (nsv==5):njack=125
    if (nsv==8):njack=512

    file = np.genfromtxt('../../get_env/n_density/test_jackknife/files/pycorr_z0_nd3_00_xi_jkn_'+str(nsv)+'_3_r_20_200.dat')
    fit = np.genfromtxt('../output/2pcf_auto_z0/nd3_00/r_20_200_njack_'+str(njack)+'_model_mean.dat')

    r = file[:,0]
    xi = file[:, 1]
    err = file[:, 2]
    err2=r**2*(err)
    err_max=max(err)

    # r vs xi
    ax1.plot(r, r**2 * xi, 'o', color='grey') #all
    ax1.errorbar(r, r**2*xi, yerr=err2, color='C'+str(idx),linestyle='None',label='')
    y=r**2*xi
    #ax1.fill_between(r, y-err2, y+err2, alpha=0.1, color='C'+str(idx))    
    #ax1.legend()
    ax1.set_xlim(20,200)
    ax1.set_ylim(-100,100)
    ax1.set_title("Auto-corr [nd = 0.001]")
    #ax1.set_xticklabels([''])
    ax1.set_ylabel(r'$r^2 \xi(r)$')
    ax1.set_xlabel(r'$r \, [h^{-1}Mpc]$')
    ax1.text(40,70,"r [20-200] Mpc/h", bbox=dict(facecolor='none', edgecolor='black', boxstyle='round,pad=1'))

    ii=np.where((fit[:,0]<190) & (fit[:,0]>20)) 
    rfit = fit[:,0][ii]
    xifit = fit[:,1][ii]
    ax1.plot(rfit,rfit**2 * xifit, label='fit model_'+str(njack), marker='.')
    #fit = np.genfromtxt('../output/2pcf_auto_z0/nd3_00/r_70_150_njack_'+str(njack)+'_model_maxlike.dat')
    #ii=np.where((fit[:,0]<160) & (fit[:,0]>60)) 
    #rfit = fit[:,0][ii]
    #xifit = fit[:,1][ii]
    #ax1.plot(rfit,rfit**2 * xifit, label='fit maxlike_'+str(njack),  marker='.',color='g')
    if(nsv==5):
        fit = np.genfromtxt('../output/2pcf_auto_z0/nd3_00/r_70_150_njack_'+str(njack)+'_fortran_model_mean.dat')
        ii=np.where((fit[:,0]<160) & (fit[:,0]>60)) 
        rfit = fit[:,0][ii]
        xifit = fit[:,1][ii]
        #ax1.plot(rfit,rfit**2 * xifit, label='fit model_'+str(njack)+'fortran',  marker='.',color='r')

    ax1.legend(loc = 'lower left')

##############################################################################

nsv_jack = [5,8]
for idx, nsv in enumerate(nsv_jack):
    if (nsv==5):njack=125
    if (nsv==8):njack=512

    file = np.genfromtxt('../../get_env/n_density/test_jackknife/files/pycorr_z0_nd3_00_xi_jkn_'+str(nsv)+'_3_r_60_160.dat')

    r = file[:,0]
    xi = file[:, 1]
    err = file[:, 2]
    err2=r**2*(err)
    err_max=max(err)

    # r vs xi
    ax2.errorbar(r, r**2*xi, yerr=err2, color='C'+str(idx),linestyle='None',label='2pcf njack='+str(njack))
    ax2.plot(r, r**2 * xi, marker='o', color='gray') #all
    y=r**2*xi
    ax2.set_xlim(30,160)
    ax2.set_ylim(-100,100)
    ax2.set_ylabel(r'$r^2 \xi(r)$')
    ax2.set_xlabel(r'$r \, [h^{-1}Mpc]$')
    ax2.text(40,70,"r [60-160] Mpc/h", bbox=dict(facecolor='none', edgecolor='black', boxstyle='round,pad=1'))
    #ax2.plot(file2[:,0], file2[:,0]**2 * file2[:,1], marker='o',label='2pcf njack=512',color='gray') #all

    fit = np.genfromtxt('../output/2pcf_auto_z0/nd3_00/r_60_160_njack_'+str(njack)+'_model_mean.dat')
    ii=np.where((fit[:,0]<160) & (fit[:,0]>30)) 
    rfit = fit[:,0][ii]
    xifit = fit[:,1][ii]
    ax2.plot(rfit,rfit**2 * xifit, label='fit model_'+str(njack),  marker='.')


    ax2.legend(loc = 'lower left')

##############################################################################

nsv_jack = [5,8]
for idx, nsv in enumerate(nsv_jack):
    if (nsv==5):njack=125
    if (nsv==8):njack=512

    file = np.genfromtxt('../../get_env/n_density/test_jackknife/files/pycorr_z0_nd3_00_xi_jkn_'+str(nsv)+'_3_r_30_150.dat')

    r = file[:,0]
    xi = file[:, 1]
    err = file[:, 2]
    err2=r**2*(err)
    err_max=max(err)

    # r vs xi
    ax3.errorbar(r, r**2*xi, yerr=err2, color='C'+str(idx),linestyle='None',label='2pcf njack='+str(njack))
    ax3.plot(r, r**2 * xi, marker='o', color='gray') #all
    y=r**2*xi
    ax3.set_xlim(30,160)
    ax3.set_ylim(-100,100)
    ax3.set_ylabel(r'$r^2 \xi(r)$')
    ax3.set_xlabel(r'$r \, [h^{-1}Mpc]$')
    ax3.text(40,70,"r [30-150] Mpc/h", bbox=dict(facecolor='none', edgecolor='black', boxstyle='round,pad=1'))
    #ax2.plot(file2[:,0], file2[:,0]**2 * file2[:,1], marker='o',label='2pcf njack=512',color='gray') #all

    fit = np.genfromtxt('../output/2pcf_auto_z0/nd3_00/r_30_150_njack_'+str(njack)+'_model_mean.dat')
    ii=np.where((fit[:,0]<160) & (fit[:,0]>30)) 
    rfit = fit[:,0][ii]
    xifit = fit[:,1][ii]
    ax3.plot(rfit,rfit**2 * xifit, label='fit model_'+str(njack),  marker='.')


    ax3.legend(loc = 'lower left')



#plt.savefig('../../plots_nden/test_fit_model_nd3_00.png')

plt.show()

