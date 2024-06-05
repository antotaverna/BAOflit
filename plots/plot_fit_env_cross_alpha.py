import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
import matplotlib.transforms as mtransforms
import pandas as pd

ranger = '60_160'
#ranger = '10_130'

fig=plt.figure(figsize=(10,8))
fig.subplots_adjust(hspace=0.9)

fig.suptitle('Cross-corr')


gs = gridspec.GridSpec(9, 2, height_ratios=[2, 1, 2, 1, 2, 1, 2, 1, 2], hspace=0, wspace=0.4)
ax1 = plt.subplot(gs[0:2, 0])
ax1b = plt.subplot(gs[2:3, 0], sharex=ax1)
ax2 = plt.subplot(gs[4:6, 0], sharex=ax1)
ax2b = plt.subplot(gs[6:7, 0], sharex=ax2)

ax4 = plt.subplot(gs[0:2, 1])
ax4b = plt.subplot(gs[2:3, 1], sharex=ax4)
ax5 = plt.subplot(gs[4:6, 1], sharex=ax4)
ax5b = plt.subplot(gs[6:7, 1], sharex=ax5)

ax3 = plt.subplot(gs[8,:])


tot = np.genfromtxt('../../get_env/fcorr/xi_files/2pcf_cross_vweb_tot_z0_r_'+ranger+'_full_fixefAmp_002.dat')


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
    file = np.genfromtxt('../../get_env/fcorr/xi_files/2pcf_cross_vweb_'+env+'_z0_r_'+ranger+'_full_fixefAmp_002.dat')
    err_file = np.genfromtxt('../../2pcf_cov_fork/output/envs/cov_cross_vweb_'+env+'_xi0_'+ranger+'_4Mpc.txt')
    fit = np.genfromtxt('../output/2pcf_cross_z0/env_vweb/'+env+'_r_'+ranger+'/r_'+ranger+'_gauss_model_mean.dat')

    r = file[:,0]
    xi = file[:, 1]
    err = r**2*np.diag(err_file)**0.5

    # r vs xi
    #ax.plot(r, r**2 * xi, 'o', color='grey') #all
    #ax.errorbar(r, r**2*xi, yerr=err, color='C1',label='')
    ax.plot(r, r**2 * xi, marker='.', color='C1',linestyle='None', label='Vweb') #all
    ax.plot(r, r**2 * tot[:,1], marker='.', color='black',linestyle='None', label='Tot') #all
    y=r**2*xi
    ax.fill_between(r, y-err, y+err, alpha=0.2, color='C1')    
    #ax.legend()
    ax.set_xlim(0,160)
    #ax.set_ylim(-50,150)
    #if(env=='knots'):
    #    ax.set_ylim(-50,70)
    #if(env=='sheets'):
    #    ax.set_ylim(-20,20)
    #ax.set_xticklabels([''])
    ax.set_ylabel(r'$r^2 \xi(r)$')
    ax.set_xlabel(r'$r \, [h^{-1}Mpc]$')

    #ii=np.where((fit[:,0]<190) & (fit[:,0]>20)) 
    #rfit = fit[:,0][ii]
    #xifit = fit[:,1][ii]
    #ax.plot(rfit,rfit**2 * xifit, label='fit mean_', marker='.')
    fit = np.genfromtxt('../output/2pcf_cross_z0/env_vweb/'+env+'_r_'+ranger+'/r_'+ranger+'_gauss_model_maxlike.dat')
    ii=np.where((fit[:,0]<160) & (fit[:,0]>10)) 
    rfit = fit[:,0][ii]
    xifit = fit[:,1][ii]
    ax.plot(rfit,rfit**2 * xifit, color='C1') #, label='fit maxlike')


    # PWEB
    file = np.genfromtxt('../../get_env/fcorr/xi_files/2pcf_cross_pweb_'+env+'_z0_r_'+ranger+'_full_fixefAmp_002.dat')
    err_file = np.genfromtxt('../../2pcf_cov_fork/output/envs/cov_cross_pweb_'+env+'_xi0_'+ranger+'_4Mpc.txt')
    fit = np.genfromtxt('../output/2pcf_cross_z0/env_pweb/'+env+'_r_'+ranger+'/r_'+ranger+'_gauss_model_mean.dat')

    r = file[:,0]
    xi = file[:, 1]
    err = r**2*np.diag(err_file)**0.5
    #ax.errorbar(r, r**2*xi, yerr=err, color='C2',linestyle='None',label='')
    ax.plot(r, r**2 * xi, marker='.', color='C2',linestyle='None', label='Pweb') #all
    y=r**2*xi
    ax.fill_between(r, y-err, y+err, alpha=0.2, color='C2')    
    fit = np.genfromtxt('../output/2pcf_cross_z0/env_pweb/'+env+'_r_'+ranger+'/r_'+ranger+'_gauss_model_maxlike.dat')
    ii=np.where((fit[:,0]<160) & (fit[:,0]>10)) 
    rfit = fit[:,0][ii]
    xifit = fit[:,1][ii]
    ax.plot(rfit,rfit**2 * xifit,color='C2')#, label='fit maxlike')

    if(ranger=='10_130'):
        ax.legend(loc = 'upper right', title=tit, fontsize=8)
    else:
        ax.legend(loc = 'upper left', title=tit, fontsize=8)
    ax.axes.get_xaxis().set_visible(False)
    ax.set_xlabel(r'$r \, [h^{-1}Mpc]$')

#### BIAS########
axsb = [ax1b,ax2b,ax4b,ax5b]

for indax,ax in enumerate(axsb):

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
    file = np.genfromtxt('../../get_env/fcorr/xi_files/2pcf_cross_vweb_'+env+'_z0_r_'+ranger+'_full_fixefAmp_002.dat')
    err_file = np.genfromtxt('../../2pcf_cov_fork/output/envs/cov_cross_vweb_'+env+'_xi0_'+ranger+'_4Mpc.txt')

    r = file[:,0]
    xi = file[:, 1]
    err = r**2*np.diag(err_file)**0.5
    ratio=xi/tot[:,1]

    # r vs xi
    #ax.plot(r, r**2 * xi, 'o', color='grey') #all
    #ax.errorbar(r, r**2*xi, yerr=err, color='C1',label='')
    ax.plot(r, ratio, marker='.', color='C1',linestyle='None', label='Vweb') #all
    y=r**2*xi
    #ax.fill_between(r, y-err, y+err, alpha=0.2, color='C1')    
    ax.set_xlim(0,160)
    ax.set_ylim(-10,10)
    ax.set_ylabel(r'$\xi(r)/xi_{tot}$')
    #ax.set_xlabel(r'$r \, [h^{-1}Mpc]$')


    # PWEB
    file = np.genfromtxt('../../get_env/fcorr/xi_files/2pcf_cross_pweb_'+env+'_z0_r_'+ranger+'_full_fixefAmp_002.dat')
    err_file = np.genfromtxt('../../2pcf_cov_fork/output/envs/cov_cross_pweb_'+env+'_xi0_'+ranger+'_4Mpc.txt')
    fit = np.genfromtxt('../output/2pcf_cross_z0/env_pweb/'+env+'_r_'+ranger+'/r_'+ranger+'_gauss_model_mean.dat')

    r = file[:,0]
    xi = file[:, 1]
    err = r**2*np.diag(err_file)**0.5
    ratio=xi/tot[:,1]

    #ax.errorbar(r, r**2*xi, yerr=err, color='C2',linestyle='None',label='')
    ax.plot(r, ratio, marker='.', color='C2',linestyle='None', label='Pweb') #all
    ax.axhline(1,0,160)
 
#plt.subplots_adjust(hspace=.0)
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
#envs cross
x11=11.
x12=12.
x13=13.
x14=14.

def Convert(string):
    li = list(string.split("   "))
    return li






#--------------------------
#------- NDEN - gauss------
#--------------------------
def plot_alpha_nden(nden,range,cov,x_point):
    #stats = pd.read_csv('../output/2pcf_auto_z0/env_'+env+'/'+struct+'_r_'+range+'/r_'+range+'_'+cov+'_stats.dat')
    stats = pd.read_csv('../output/2pcf_auto_z0/'+nden+'/'+cov+'_r_'+range+'/r_'+range+'_'+cov+'_stats.dat')
    str1 = stats.iloc[2,0]
    a, b, alpha, error_alpha = Convert(str1)
    al_rc=float(alpha)
    err_al_rc=float(error_alpha)
    ax3.errorbar(x_point, al_rc, yerr=err_al_rc, fmt="o", color='k')#, label='Gauss')


plot_alpha_nden('nd0_00','60_160','gauss',x1)
plot_alpha_nden('nd1_00','60_160','gauss',x2)
plot_alpha_nden('nd2_00','60_160','gauss',x3)
plot_alpha_nden('nd3_00','60_160','gauss',x4)

#################### nd0 --> Total ########################
#/r_60_160----------------------------------
#rascalC c/first
stats = pd.read_csv('../output/2pcf_auto_z0/first/r_60_160_rascalc_2x_stats.dat')
str1 = stats.iloc[2,0]
a, b, alpha, error_alpha = Convert(str1)
al_rc=float(alpha)
err_al_rc=float(error_alpha)
#ax3.errorbar(x1+0.1, al_rc, yerr=err_al_rc, fmt="o", color='y', label='RascalC')
#rascalC c/nd3
stats = pd.read_csv('../output/2pcf_auto_z0/nd3_00/rascalc_r_60_160/r_60_160_rascalc_10x_stats.dat')
str1 = stats.iloc[2,0]
a, b, alpha, error_alpha = Convert(str1)
al_rc=float(alpha)
err_al_rc=float(error_alpha)
#ax3.errorbar(x4+0.1, al_rc, yerr=err_al_rc, fmt="o", color='y')#, label='first RascalC')





#--------------------------
#------- ENVS CROSS --------
#--------------------------

def plot_alpha_cross(env,struct,range,cov,x_point):
    stats = pd.read_csv('../output/2pcf_cross_z0/env_'+env+'/'+struct+'_r_'+range+'/r_'+range+'_'+cov+'_stats.dat')
    str1 = stats.iloc[2,0]
    a, b, alpha, error_alpha = Convert(str1)
    al_rc=float(alpha)
    err_al_rc=float(error_alpha)
    if(env=='pweb'):
        ax3.errorbar(x_point-0.1, al_rc, yerr=err_al_rc, fmt="o", color='C2')#, label='Gauss')
    if(env=='vweb'):
        ax3.errorbar(x_point+0.1, al_rc, yerr=err_al_rc, fmt="^", color='C1')#, label='Gauss')

#voids
plot_alpha_cross('pweb','voids','60_160','gauss',x6)
plot_alpha_cross('vweb','voids','60_160','gauss',x6)  

#sheets
plot_alpha_cross('pweb','sheets','60_160','gauss',x7)
plot_alpha_cross('vweb','sheets','60_160','gauss',x7) 

#filaments
plot_alpha_cross('pweb','filaments','60_160','gauss',x8)
plot_alpha_cross('vweb','filaments','60_160','gauss',x8) 

#knots
plot_alpha_cross('pweb','knots','60_160','gauss',x9)
plot_alpha_cross('vweb','knots','60_160','gauss',x9) 




#--------------------------
#------- ENVS AUTO --------
#--------------------------
def plot_alpha_auto(env,struct,range,cov,x_point):
    stats = pd.read_csv('../output/2pcf_auto_z0/env_'+env+'/'+struct+'_r_'+range+'/r_'+range+'_'+cov+'_stats.dat')
    str1 = stats.iloc[2,0]
    a, b, alpha, error_alpha = Convert(str1)
    al_rc=float(alpha)
    err_al_rc=float(error_alpha)
    if(env=='pweb'):
        ax3.errorbar(x_point-0.1, al_rc, yerr=err_al_rc, fmt="o", color='C2')#, label='Gauss')
    if(env=='vweb'):
        ax3.errorbar(x_point+0.1, al_rc, yerr=err_al_rc, fmt="^", color='C1')#, label='Gauss')

#voids
plot_alpha_auto('pweb','voids','60_160','gauss',x11)
plot_alpha_auto('vweb','voids','60_160','gauss',x11)  

#sheets
plot_alpha_auto('pweb','sheets','60_160','gauss',x12)
plot_alpha_auto('vweb','sheets','60_160','gauss',x12) 

#filaments
plot_alpha_auto('pweb','filaments','60_160','gauss',x13)
plot_alpha_auto('vweb','filaments','60_160','gauss',x13) 

#knots
plot_alpha_auto('pweb','knots','60_160','gauss',x14)
plot_alpha_auto('vweb','knots','60_160','gauss',x14) 

#---------------------------------------------------------------------------


ax3.set_ylim(0.90,1.10)
ax3.axhline(1.0 ,linestyle='dotted')
ax3.axvline(5.0 ,linestyle='dotted')
ax3.axvline(10.0 ,linestyle='dotted')
ax3.axhline(1.0 ,linestyle='dotted')
#ax3.legend(loc='lower left', fontsize=8)

ax3.text(3.9,0.93,"nden", bbox=dict(facecolor='none', edgecolor='black', boxstyle='round,pad=1'), fontsize=6)
ax3.text(8.9,0.93,"Cross", bbox=dict(facecolor='none', edgecolor='black', boxstyle='round,pad=1'), fontsize=6)
ax3.text(13.9,.93,"Auto", bbox=dict(facecolor='none', edgecolor='black', boxstyle='round,pad=1'), fontsize=6)


ax3.set_xlim(0,15)
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
labels[11] = '$Voids$'
labels[12] = '$Sheets$'
labels[13] = '$Filaments$'
labels[14] = '$Knots$'

ax3.set_xticklabels(labels, rotation=0, fontsize=8)
plt.savefig('../../plots_fcorr/fit_2pcf_cross_alpha.png')

plt.show()

