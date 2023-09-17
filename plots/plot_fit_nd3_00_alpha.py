import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
import matplotlib.transforms as mtransforms
import pandas as pd


#nsv_jack = [5,8,10,16,20,32,40,64,128]

fig=plt.figure(figsize=(9,12))
#fig.subplots_adjust(top = 0.9)
gs = gridspec.GridSpec(3,1, wspace=0.3, hspace=0.2)#, width_ratios=[1, 1])
ax1 = plt.subplot(gs[0,0])
ax2 = plt.subplot(gs[1,0])
ax3 = plt.subplot(gs[2,0])

def f1(a1,a2,a3,x):
    return a1/(x**2) + a2/x + a3

nsv_jack = [5,8]
for idx, nsv in enumerate(nsv_jack):
    if (nsv==5):njack=125
    if (nsv==8):njack=512

    file = np.genfromtxt('../../get_env/n_density/test_jackknife/files/pycorr_z0_nd3_00_xi_jkn_'+str(nsv)+'_3_r_70_150.dat')
    fit = np.genfromtxt('../output/2pcf_auto_z0/nd3_00/r_70_150_njack_'+str(njack)+'_model_mean.dat')

    r = file[:,0]
    xi = file[:, 1]
    err = file[:, 2]
    err2=r**2*(err)
    err_max=max(err)

    # r vs xi
    #ax1.plot(r, r**2 * xi, 'o', color='grey') #all
    ax1.errorbar(r, r**2*xi, yerr=err2, color='C'+str(idx),linestyle='None',label='')
    ax1.plot(r, r**2 * xi, marker='o', color='gray') #all
    y=r**2*xi
    #ax1.fill_between(r, y-err2, y+err2, alpha=0.1, color='C'+str(idx))    
    #ax1.legend()
    ax1.set_xlim(20,200)
    ax1.set_ylim(-100,100)
    ax1.set_title("Auto-corr [nd = 0.001]")
    #ax1.set_xticklabels([''])
    ax1.set_ylabel(r'$r^2 \xi(r)$')
    ax1.set_xlabel(r'$r \, [h^{-1}Mpc]$')

    ii=np.where((fit[:,0]<190) & (fit[:,0]>20)) 
    rfit = fit[:,0][ii]
    xifit = fit[:,1][ii]
    #ax1.plot(rfit,rfit**2 * xifit, label='fit mean_'+str(njack), marker='.')
    fit = np.genfromtxt('../output/2pcf_auto_z0/nd3_00/r_70_150_njack_'+str(njack)+'_model_maxlike.dat')
    ii=np.where((fit[:,0]<160) & (fit[:,0]>60)) 
    rfit = fit[:,0][ii]
    xifit = fit[:,1][ii]
    ax1.plot(rfit,rfit**2 * xifit, label='fit maxlike_'+str(njack),color='C'+str(idx))

    ax1.legend(loc = 'lower left')


##############################################################################

fit_range = [1]#,2]
for idx, fr in enumerate(fit_range):
    if (fr==1):fitrange='25_150'
    if (fr==2):fitrange='60_160'

    file = np.genfromtxt('../../RascalC/output/nd3_00/pycorr_z0_nd3_00_xi_jkn_8_3_r_'+str(fitrange)+'.dat')
    cov_rascalc = np.genfromtxt('../../RascalC/output/nd3_00/cov_matrix_RascalC_z0_nd3_00_r_'+str(fitrange)+'_10x.dat')
    fit_maxl = np.genfromtxt('../output/2pcf_auto_z0/nd3_00/rascalc_r_'+str(fitrange)+'/r_'+str(fitrange)+'_rascalc_10x_model_maxlike.dat')

    r = file[:,0]
    xi = file[:, 1]
    err = np.sqrt(np.diag(cov_rascalc))
    err2=r**2*(err)

    # r vs xi
    ax2.errorbar(r, r**2*xi, yerr=err2, color='C'+str(idx),linestyle='None',label='2pcf range='+str(fitrange))
    ax2.plot(r, r**2 * xi, marker='o', color='gray') #all
    y=r**2*xi
    ax2.set_xlim(30,160)
    ax2.set_ylim(-100,100)
    ax2.set_ylabel(r'$r^2 \xi(r)$')
    ax2.set_xlabel(r'$r \, [h^{-1}Mpc]$')


    rfit = fit_maxl[:,0]
    xifit = fit_maxl[:,1]
    ax2.plot(rfit,rfit**2 * xifit, label='fit maxlike_'+str(fitrange),color='C'+str(idx))

    ax2.legend(loc = 'lower left')

##############################################################################
########## Plot alpha ################

####  nd 3_00 pycorr ##################
#r [60,160]
#nsv = [5^3,8^3]
al = [0.989174560697012484E+00,0.986222170079396609E+00] 
err_al = [0.170585367196523369E-01, 0.242586733709745095E-01]
xx= [1,2]
ax3.errorbar(xx[0], al[0], yerr=err_al[0], label='pycorr 5^3: [60,160]', fmt="o", color='C1')
ax3.errorbar(xx[1], al[1], yerr=err_al[1], label='pycorr 8^3: [60,160]', fmt="o", color='C2')



################  nd 2_00 #####################################
#r [60,160]
#nsv = [5^3,8^3]
al = [0.996468238219410996E+00, 0.993100703781965910E+00]
err_al = [0.152699988503018653E-01, 0.194684800204054931E-01]
xx= [6,7]
ax3.errorbar(xx[0], al[0], yerr=err_al[0], fmt="o", color='C1')
ax3.errorbar(xx[1], al[1], yerr=err_al[1], fmt="o", color='C2')


##############################################################





############ read alpha + error BAOflit

def Convert(string):
    li = list(string.split("   "))
    return li


#/r_60_160
#10x
stats = pd.read_csv('../output/2pcf_auto_z0/nd3_00/rascalc_r_60_160/r_60_160_rascalc_10x_stats.dat')
str1 = stats.iloc[2,0]
a, b, alpha, error_alpha = Convert(str1)
al_rc=float(alpha)
err_al_rc=float(error_alpha)
ax3.errorbar(3, al_rc, yerr=err_al_rc, label='rascal: nd3_00 [60,160]', fmt="o", color='k')
#5x
stats = pd.read_csv('../output/2pcf_auto_z0/nd3_00/rascalc_r_60_160/r_60_160_rascalc_5x_stats.dat')
str1 = stats.iloc[2,0]
a, b, alpha, error_alpha = Convert(str1)
al_rc=float(alpha)
err_al_rc=float(error_alpha)
ax3.errorbar(3.1, al_rc, yerr=err_al_rc, fmt="o", color='gray')
#2x
stats = pd.read_csv('../output/2pcf_auto_z0/nd3_00/rascalc_r_60_160/r_60_160_rascalc_2x_stats.dat')
str1 = stats.iloc[2,0]
a, b, alpha, error_alpha = Convert(str1)
al_rc=float(alpha)
err_al_rc=float(error_alpha)
ax3.errorbar(3.2, al_rc, yerr=err_al_rc, fmt="o", color='gray')
#1x
stats = pd.read_csv('../output/2pcf_auto_z0/nd3_00/rascalc_r_60_160/r_60_160_rascalc_1x_stats.dat')
str1 = stats.iloc[2,0]
a, b, alpha, error_alpha = Convert(str1)
al_rc=float(alpha)
err_al_rc=float(error_alpha)
ax3.errorbar(3.3, al_rc, yerr=err_al_rc, fmt="o", color='gray')


#/r_25_150
stats = pd.read_csv('../output/2pcf_auto_z0/nd3_00/rascalc_r_25_150/r_25_150_rascalc_10x_stats.dat')
str1 = stats.iloc[2,0]
a, b, alpha, error_alpha = Convert(str1)
al_rc=float(alpha)
err_al_rc=float(error_alpha)
ax3.errorbar(4, al_rc, yerr=err_al_rc, label='rascal: nd3_00 [25,150]', fmt="o", color='y')
stats = pd.read_csv('../output/2pcf_auto_z0/nd3_00/rascalc_r_25_150/r_25_150_rascalc_5x_stats.dat')
str1 = stats.iloc[2,0]
a, b, alpha, error_alpha = Convert(str1)
al_rc=float(alpha)
err_al_rc=float(error_alpha)
ax3.errorbar(4.1, al_rc, yerr=err_al_rc, fmt="o", color='gray')
stats = pd.read_csv('../output/2pcf_auto_z0/nd3_00/rascalc_r_25_150/r_25_150_rascalc_2x_stats.dat')
str1 = stats.iloc[2,0]
a, b, alpha, error_alpha = Convert(str1)
al_rc=float(alpha)
err_al_rc=float(error_alpha)
ax3.errorbar(4.2, al_rc, yerr=err_al_rc, fmt="o", color='gray')
stats = pd.read_csv('../output/2pcf_auto_z0/nd3_00/rascalc_r_25_150/r_25_150_rascalc_1x_stats.dat')
str1 = stats.iloc[2,0]
a, b, alpha, error_alpha = Convert(str1)
al_rc=float(alpha)
err_al_rc=float(error_alpha)
ax3.errorbar(4.3, al_rc, yerr=err_al_rc, fmt="o", color='gray')




ax3.set_ylim(0.90,1.10)
ax3.set_xlim(0,10)
ax3.axhline(1.0 ,linestyle='dotted')
ax3.axvline(5.0 ,linestyle='dotted')
ax3.legend(loc='lower right', fontsize=8)
ax3.text(0.3,1.05,"nd3_00", bbox=dict(facecolor='none', edgecolor='black', boxstyle='round,pad=1'))
ax3.text(5.3,1.05,"nd2_00", bbox=dict(facecolor='none', edgecolor='black', boxstyle='round,pad=1'))

#plt.savefig('../../plots_nden/test_fit_model_nd3_00.png')

plt.show()

