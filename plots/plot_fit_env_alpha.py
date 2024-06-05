import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
import matplotlib.transforms as mtransforms
import pandas as pd


fig=plt.figure(figsize=(9,12))
#fig.subplots_adjust(top = 0.9)
fig.suptitle('Auto-corr')
gs = gridspec.GridSpec(3,2, wspace=0.3, hspace=0.2)#, width_ratios=[1, 1])
ax1 = plt.subplot(gs[0,0])
ax2 = plt.subplot(gs[0,1])
ax4 = plt.subplot(gs[1,0])
ax5 = plt.subplot(gs[1,1])

ax3 = plt.subplot(gs[2,:])

axs = [ax1,ax2,ax4,ax5]

for indax,ax in enumerate(axs):

    if(indax==0):
        env='voids'
        tit='Voids'
    if(indax==1):
        env='sheets'
        tit='Sheets'
    if(indax==2):
        env='filaments'
        tit='Filaments'
    if(indax==3):
        env='knots'
        tit='Knots'

    # VWEB
    file = np.genfromtxt('../../get_env/fcorr/xi_files/2pcf_auto_vweb_'+env+'_z0_r_60_160_full_fixefAmp_002.dat')
    err_file = np.genfromtxt('../../2pcf_cov_fork/output/envs/cov_vweb_'+env+'_xi0_60_160_4Mpc.txt')
    fit = np.genfromtxt('../output/2pcf_auto_z0/env_vweb/'+env+'_r_60_160/r_60_160_gauss_model_mean.dat')

    r = file[:,0]
    xi = file[:, 1]
    err = r**2*np.diag(err_file)**0.5

    # r vs xi
    #ax.plot(r, r**2 * xi, 'o', color='grey') #all
    #ax.errorbar(r, r**2*xi, yerr=err, color='C1',label='')
    ax.plot(r, r**2 * xi, marker='.', color='C1',linestyle='None', label='Vweb') #all
    y=r**2*xi
    ax.fill_between(r, y-err, y+err, alpha=0.2, color='C1')    
    #ax.legend()
    ax.set_xlim(50,160)
    ax.set_ylim(-70,250)
    if(env=='knots'):
        ax.set_ylim(-150,650)
    if(env=='sheets'):
        ax.set_ylim(-20,50)
    #ax.set_xticklabels([''])
    ax.set_ylabel(r'$r^2 \xi(r)$')
    ax.set_xlabel(r'$r \, [h^{-1}Mpc]$')

    #ii=np.where((fit[:,0]<190) & (fit[:,0]>20)) 
    #rfit = fit[:,0][ii]
    #xifit = fit[:,1][ii]
    #ax.plot(rfit,rfit**2 * xifit, label='fit mean_', marker='.')
    fit = np.genfromtxt('../output/2pcf_auto_z0/env_vweb/'+env+'_r_60_160/r_60_160_gauss_model_maxlike.dat')
    ii=np.where((fit[:,0]<160) & (fit[:,0]>60)) 
    rfit = fit[:,0][ii]
    xifit = fit[:,1][ii]
    ax.plot(rfit,rfit**2 * xifit, color='C1') #, label='fit maxlike')


    # PWEB
    file = np.genfromtxt('../../get_env/fcorr/xi_files/2pcf_auto_pweb_'+env+'_z0_r_60_160_full_fixefAmp_002.dat')
    err_file = np.genfromtxt('../../2pcf_cov_fork/output/envs/cov_pweb_'+env+'_xi0_60_160_4Mpc.txt')
    fit = np.genfromtxt('../output/2pcf_auto_z0/env_pweb/'+env+'_r_60_160/r_60_160_gauss_model_mean.dat')

    r = file[:,0]
    xi = file[:, 1]
    err = r**2*np.diag(err_file)**0.5
    #ax.errorbar(r, r**2*xi, yerr=err, color='C2',linestyle='None',label='')
    ax.plot(r, r**2 * xi, marker='.', color='C2',linestyle='None', label='Pweb') #all
    y=r**2*xi
    ax.fill_between(r, y-err, y+err, alpha=0.2, color='C2')    
    fit = np.genfromtxt('../output/2pcf_auto_z0/env_pweb/'+env+'_r_60_160/r_60_160_gauss_model_maxlike.dat')
    ii=np.where((fit[:,0]<160) & (fit[:,0]>60)) 
    rfit = fit[:,0][ii]
    xifit = fit[:,1][ii]
    ax.plot(rfit,rfit**2 * xifit,color='C2')#, label='fit maxlike')

    ax.legend(loc = 'upper right', title=tit, fontsize=8)

##############################################################################
########## Plot alpha ################

############ read alpha + error BAOflit
#nd
x1=1.
x2=2.
x3=3.
x4=4.
#envs
x6=6.
x7=7.
x8=8.
x9=9.

def Convert(string):
    li = list(string.split("   "))
    return li


#################### nd0 --> Total ########################
#/r_60_160----------------------------------
#rascalC c/first
stats = pd.read_csv('../output/2pcf_auto_z0/first/r_60_160_rascalc_2x_stats.dat')
str1 = stats.iloc[2,0]
a, b, alpha, error_alpha = Convert(str1)
al_rc=float(alpha)
err_al_rc=float(error_alpha)
ax3.errorbar(x1+0.1, al_rc, yerr=err_al_rc, fmt="o", color='g', label='RascalC')
#rascalC c/nd3
stats = pd.read_csv('../output/2pcf_auto_z0/nd3_00/rascalc_r_60_160/r_60_160_rascalc_10x_stats.dat')
str1 = stats.iloc[2,0]
a, b, alpha, error_alpha = Convert(str1)
al_rc=float(alpha)
err_al_rc=float(error_alpha)
ax3.errorbar(x4+0.1, al_rc, yerr=err_al_rc, fmt="o", color='g')#, label='first RascalC')


#Gauss
#------- nd0--------
stats = pd.read_csv('../output/2pcf_auto_z0/nd0_00/gauss_r_60_160/r_60_160_gauss_stats.dat')
str1 = stats.iloc[2,0]
a, b, alpha, error_alpha = Convert(str1)
al_rc=float(alpha)
err_al_rc=float(error_alpha)
ax3.errorbar(x1-0.1, al_rc, yerr=err_al_rc, fmt="o", color='m', label='Gauss')
#------- nd1--------
stats = pd.read_csv('../output/2pcf_auto_z0/nd1_00/gauss_r_60_160/r_60_160_gauss_stats.dat')
str1 = stats.iloc[2,0]
a, b, alpha, error_alpha = Convert(str1)
al_rc=float(alpha)
err_al_rc=float(error_alpha)
ax3.errorbar(x2, al_rc, yerr=err_al_rc, fmt="o", color='m')#, label='Gauss')
#Gauss
#------- nd2--------
stats = pd.read_csv('../output/2pcf_auto_z0/nd2_00/gauss_r_60_160/r_60_160_gauss_stats.dat')
str1 = stats.iloc[2,0]
a, b, alpha, error_alpha = Convert(str1)
al_rc=float(alpha)
err_al_rc=float(error_alpha)
ax3.errorbar(x3, al_rc, yerr=err_al_rc, fmt="o", color='m')#, label='Gauss')
#------- nd3--------
stats = pd.read_csv('../output/2pcf_auto_z0/nd3_00/gauss_r_60_160/r_60_160_gauss_stats.dat')
str1 = stats.iloc[2,0]
a, b, alpha, error_alpha = Convert(str1)
al_rc=float(alpha)
err_al_rc=float(error_alpha)
ax3.errorbar(x4, al_rc, yerr=err_al_rc, fmt="o", color='m')#, label='Gauss')

#--------------------
#------- ENVS--------
#--------------------
#voids
stats = pd.read_csv('../output/2pcf_auto_z0/env_pweb/voids_r_60_160/r_60_160_gauss_stats.dat')
str1 = stats.iloc[2,0]
a, b, alpha, error_alpha = Convert(str1)
al_rc=float(alpha)
err_al_rc=float(error_alpha)
ax3.errorbar(x6-0.1, al_rc, yerr=err_al_rc, fmt="o", color='m')#, label='Gauss')
stats = pd.read_csv('../output/2pcf_auto_z0/env_vweb/voids_r_60_160/r_60_160_gauss_stats.dat')
str1 = stats.iloc[2,0]
a, b, alpha, error_alpha = Convert(str1)
al_rc=float(alpha)
err_al_rc=float(error_alpha)
ax3.errorbar(x6+0.1, al_rc, yerr=err_al_rc, fmt='^', color='m')#, label='Gauss')
#sheets
stats = pd.read_csv('../output/2pcf_auto_z0/env_pweb/sheets_r_60_160/r_60_160_gauss_stats.dat')
str1 = stats.iloc[2,0]
a, b, alpha, error_alpha = Convert(str1)
al_rc=float(alpha)
err_al_rc=float(error_alpha)
ax3.errorbar(x7-0.1, al_rc, yerr=err_al_rc, fmt="o", color='m')#, label='Gauss')
stats = pd.read_csv('../output/2pcf_auto_z0/env_vweb/sheets_r_60_160/r_60_160_gauss_stats.dat')
str1 = stats.iloc[2,0]
a, b, alpha, error_alpha = Convert(str1)
al_rc=float(alpha)
err_al_rc=float(error_alpha)
ax3.errorbar(x7+0.1, al_rc, yerr=err_al_rc, fmt="^", color='m')#, label='Gauss')
#filaments
stats = pd.read_csv('../output/2pcf_auto_z0/env_pweb/filaments_r_60_160/r_60_160_gauss_stats.dat')
str1 = stats.iloc[2,0]
a, b, alpha, error_alpha = Convert(str1)
al_rc=float(alpha)
err_al_rc=float(error_alpha)
ax3.errorbar(x8-0.1, al_rc, yerr=err_al_rc, fmt="o", color='m')#, label='Gauss')
stats = pd.read_csv('../output/2pcf_auto_z0/env_vweb/filaments_r_60_160/r_60_160_gauss_stats.dat')
str1 = stats.iloc[2,0]
a, b, alpha, error_alpha = Convert(str1)
al_rc=float(alpha)
err_al_rc=float(error_alpha)
ax3.errorbar(x8+0.1, al_rc, yerr=err_al_rc, fmt="^", color='m')#, label='Gauss')
#knots
stats = pd.read_csv('../output/2pcf_auto_z0/env_pweb/knots_r_60_160/r_60_160_gauss_stats.dat')
str1 = stats.iloc[2,0]
a, b, alpha, error_alpha = Convert(str1)
al_rc=float(alpha)
err_al_rc=float(error_alpha)
ax3.errorbar(x9-0.1, al_rc, yerr=err_al_rc, fmt="o", color='m')#, label='Gauss')
stats = pd.read_csv('../output/2pcf_auto_z0/env_vweb/knots_r_60_160/r_60_160_gauss_stats.dat')
str1 = stats.iloc[2,0]
a, b, alpha, error_alpha = Convert(str1)
al_rc=float(alpha)
err_al_rc=float(error_alpha)
ax3.errorbar(x9+0.1, al_rc, yerr=err_al_rc, fmt="^", color='m')#, label='Gauss')


ax3.set_ylim(0.90,1.10)
ax3.axhline(1.0 ,linestyle='dotted')
ax3.axvline(5.0 ,linestyle='dotted')
#ax3.axvline(8.0 ,linestyle='dotted')
ax3.axhline(1.0 ,linestyle='dotted')
ax3.legend(loc='upper right', fontsize=10)

#ax3.text(3.3,1.09,"nden", bbox=dict(facecolor='none', edgecolor='black', boxstyle='round,pad=1'), fontsize=8)
#ax3.text(6.3,1.09,"vweb", bbox=dict(facecolor='none', edgecolor='black', boxstyle='round,pad=1'), fontsize=8)
#ax3.text(8.3,1.05,"nd0_00", bbox=dict(facecolor='none', edgecolor='black', boxstyle='round,pad=1'), fontsize=8)


ax3.set_xlim(0,12)
start, end = ax3.get_xlim()
ax3.xaxis.set_ticks(np.arange(start, end, 1.))
labels = [item.get_text() for item in ax3.get_xticklabels()]
labels[1] = '$total$'
labels[2] = '$nd1$'
labels[3] = '$nd2$'
labels[4] = '$nd3$'
labels[6] = '$Voids$'
labels[7] = '$Sheets$'
labels[8] = '$Filaments$'
labels[9] = '$Knots$'

ax3.set_xticklabels(labels, rotation=0, fontsize=8)
#plt.savefig('../../plots_nden/test_fit_model_nd3_00.png')

plt.show()

