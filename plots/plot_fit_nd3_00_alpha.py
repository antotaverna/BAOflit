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

fit_range = [2]#,1]
for idx, fr in enumerate(fit_range):
    if (fr==1):fitrange='25_150'
    if (fr==2):fitrange='60_160'

    file = np.genfromtxt('../../RascalC/output/nd3_00/pycorr_z0_nd3_00_xi_jkn_8_3_r_'+str(fitrange)+'.dat')
    cov_rascalc = np.genfromtxt('../../RascalC/output/nd3_00/cov_matrix_RascalC_z0_nd3_00_r_'+str(fitrange)+'_10x.dat')
    fit_maxl = np.genfromtxt('../output/2pcf_auto_z0/nd3_00/rascalc_r_'+str(fitrange)+'/r_'+str(fitrange)+'_rascalc_10x_model_maxlike.dat')
    fit_maxl2 = np.genfromtxt('../output/2pcf_auto_z0/nd3_00/gauss_r_'+str(fitrange)+'/r_'+str(fitrange)+'_gauss_model_maxlike.dat')

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
    rfit = fit_maxl2[:,0]
    xifit = fit_maxl2[:,1]
    ax2.plot(rfit,rfit**2 * xifit, label='Gauss fit maxlike_'+str(fitrange),color='m')

    ax2.legend(loc = 'lower left')

##############################################################################
########## Plot alpha ################

############ read alpha + error BAOflit
#nd3
x1=0.9
x2=1.1
x3=2.
x4=3.
#nd2
x5=4.9
x6=5.1

#nd0
x8=8.9
x9=9.1
x10=10.
x11=10.1

def Convert(string):
    li = list(string.split("   "))
    return li

#################### nd3 --> nd=0.001 ########################
#/r_60_160----------------------------------
#10x
stats = pd.read_csv('../output/2pcf_auto_z0/nd3_00/rascalc_r_60_160/r_60_160_rascalc_10x_stats.dat')
str1 = stats.iloc[2,0]
a, b, alpha, error_alpha = Convert(str1)
al_rc=float(alpha)
err_al_rc=float(error_alpha)
ax3.errorbar(x3, al_rc, yerr=err_al_rc, label='rascal: nd3_00 [60,160]', fmt="o", color='k')
#5x
stats = pd.read_csv('../output/2pcf_auto_z0/nd3_00/rascalc_r_60_160/r_60_160_rascalc_5x_stats.dat')
str1 = stats.iloc[2,0]
a, b, alpha, error_alpha = Convert(str1)
al_rc=float(alpha)
err_al_rc=float(error_alpha)
ax3.errorbar(x3+0.1, al_rc, yerr=err_al_rc, fmt="o", color='gray')
#2x
stats = pd.read_csv('../output/2pcf_auto_z0/nd3_00/rascalc_r_60_160/r_60_160_rascalc_2x_stats.dat')
str1 = stats.iloc[2,0]
a, b, alpha, error_alpha = Convert(str1)
al_rc=float(alpha)
err_al_rc=float(error_alpha)
ax3.errorbar(x3+0.2, al_rc, yerr=err_al_rc, fmt="o", color='gray')
#1x
stats = pd.read_csv('../output/2pcf_auto_z0/nd3_00/rascalc_r_60_160/r_60_160_rascalc_1x_stats.dat')
str1 = stats.iloc[2,0]
a, b, alpha, error_alpha = Convert(str1)
al_rc=float(alpha)
err_al_rc=float(error_alpha)
ax3.errorbar(x3+0.3, al_rc, yerr=err_al_rc, fmt="o", color='gray')

#gauss
stats = pd.read_csv('../output/2pcf_auto_z0/nd3_00/gauss_r_60_160/r_60_160_gauss_stats.dat')
str1 = stats.iloc[2,0]
a, b, alpha, error_alpha = Convert(str1)
al_rc=float(alpha)
err_al_rc=float(error_alpha)
ax3.errorbar(x3+0.4, al_rc, yerr=err_al_rc, fmt="o", color='magenta')


#pycorr
stats = pd.read_csv('../output/2pcf_auto_z0/nd3_00/r_60_160_njack_125_stats.dat')
str1 = stats.iloc[2,0]
a, b, alpha, error_alpha = Convert(str1)
al_rc=float(alpha)
err_al_rc=float(error_alpha)
ax3.errorbar(x1, al_rc, yerr=err_al_rc, fmt="o", color='C1',label='pycorr 5^3: [60,160]')
stats = pd.read_csv('../output/2pcf_auto_z0/nd3_00/r_60_160_njack_512_stats.dat')
str1 = stats.iloc[2,0]
a, b, alpha, error_alpha = Convert(str1)
al_rc=float(alpha)
err_al_rc=float(error_alpha)
ax3.errorbar(x2, al_rc, yerr=err_al_rc, fmt="o", color='g',label='pycorr 8^3: [60,160]')

#------------------------------------------------------------
#/r_25_150---------------------------------------------------
stats = pd.read_csv('../output/2pcf_auto_z0/nd3_00/rascalc_r_25_150/r_25_150_rascalc_10x_stats.dat')
str1 = stats.iloc[2,0]
a, b, alpha, error_alpha = Convert(str1)
al_rc=float(alpha)
err_al_rc=float(error_alpha)
ax3.errorbar(x4, al_rc, yerr=err_al_rc, label='rascal: nd3_00 [25,150]', fmt="o", color='y')
stats = pd.read_csv('../output/2pcf_auto_z0/nd3_00/rascalc_r_25_150/r_25_150_rascalc_5x_stats.dat')
str1 = stats.iloc[2,0]
a, b, alpha, error_alpha = Convert(str1)
al_rc=float(alpha)
err_al_rc=float(error_alpha)
ax3.errorbar(x4+0.1, al_rc, yerr=err_al_rc, fmt="o", color='gray')
stats = pd.read_csv('../output/2pcf_auto_z0/nd3_00/rascalc_r_25_150/r_25_150_rascalc_2x_stats.dat')
str1 = stats.iloc[2,0]
a, b, alpha, error_alpha = Convert(str1)
al_rc=float(alpha)
err_al_rc=float(error_alpha)
ax3.errorbar(x4+0.2, al_rc, yerr=err_al_rc, fmt="o", color='gray')
stats = pd.read_csv('../output/2pcf_auto_z0/nd3_00/rascalc_r_25_150/r_25_150_rascalc_1x_stats.dat')
str1 = stats.iloc[2,0]
a, b, alpha, error_alpha = Convert(str1)
al_rc=float(alpha)
err_al_rc=float(error_alpha)
ax3.errorbar(x4+0.3, al_rc, yerr=err_al_rc, fmt="o", color='gray')


#################### nd2 --> nd=0.01 ########################
#/r_60_160----------------------------------
#pycorr
stats = pd.read_csv('../output/2pcf_auto_z0/nd2_00/r_60_160_njack_125_stats.dat')
str1 = stats.iloc[2,0]
a, b, alpha, error_alpha = Convert(str1)
al_rc=float(alpha)
err_al_rc=float(error_alpha)
ax3.errorbar(x5, al_rc, yerr=err_al_rc, fmt="o", color='C1')
stats = pd.read_csv('../output/2pcf_auto_z0/nd2_00/r_60_160_njack_512_stats.dat')
str1 = stats.iloc[2,0]
a, b, alpha, error_alpha = Convert(str1)
al_rc=float(alpha)
err_al_rc=float(error_alpha)
ax3.errorbar(x6, al_rc, yerr=err_al_rc, fmt="o", color='g')


#################### nd0 --> nd=0.1 ########################
#/r_60_160----------------------------------



#################### nd0 --> Total ########################
#/r_60_160----------------------------------
#pycorr
stats = pd.read_csv('../output/2pcf_auto_z0/nd0_00/r_60_160_njack_125_stats.dat')
str1 = stats.iloc[2,0]
a, b, alpha, error_alpha = Convert(str1)
al_rc=float(alpha)
err_al_rc=float(error_alpha)
ax3.errorbar(x8, al_rc, yerr=err_al_rc, fmt="o", color='C1')
stats = pd.read_csv('../output/2pcf_auto_z0/nd0_00/r_60_160_njack_512_stats.dat')
str1 = stats.iloc[2,0]
a, b, alpha, error_alpha = Convert(str1)
al_rc=float(alpha)
err_al_rc=float(error_alpha)
ax3.errorbar(x9, al_rc, yerr=err_al_rc, fmt="o", color='g')
stats = pd.read_csv('../output/2pcf_auto_z0/first/r_60_160_rascalc_2x_stats.dat')
str1 = stats.iloc[2,0]
a, b, alpha, error_alpha = Convert(str1)
al_rc=float(alpha)
err_al_rc=float(error_alpha)
ax3.errorbar(x10, al_rc, yerr=err_al_rc, fmt="o", color='g')
#Gauss
stats = pd.read_csv('../output/2pcf_auto_z0/total/total_r_60_160_stats.dat')
str1 = stats.iloc[2,0]
a, b, alpha, error_alpha = Convert(str1)
al_rc=float(alpha)
err_al_rc=float(error_alpha)
ax3.errorbar(x11, al_rc, yerr=err_al_rc, fmt="o", color='m')
stats = pd.read_csv('../output/2pcf_auto_z0/total/total_r_60_1600_tapp_stats.dat')
str1 = stats.iloc[2,0]
a, b, alpha, error_alpha = Convert(str1)
al_rc=float(alpha)
err_al_rc=float(error_alpha)
ax3.errorbar(x11+0.1, al_rc, yerr=err_al_rc, fmt="o", color='y')
stats = pd.read_csv('../output/2pcf_auto_z0/total/total_r_60_160_tapp_c_stats.dat')
str1 = stats.iloc[2,0]
a, b, alpha, error_alpha = Convert(str1)
al_rc=float(alpha)
err_al_rc=float(error_alpha)
ax3.errorbar(x11+0.2, al_rc, yerr=err_al_rc, fmt="o", color='r')

ax3.set_ylim(0.90,1.10)
ax3.axhline(1.0 ,linestyle='dotted')
ax3.axvline(4.0 ,linestyle='dotted')
ax3.axvline(8.0 ,linestyle='dotted')
ax3.legend(loc='center', fontsize=8)

ax3.text(0.3,1.05,"nd3_00", bbox=dict(facecolor='none', edgecolor='black', boxstyle='round,pad=1'), fontsize=8)
ax3.text(4.3,1.05,"nd2_00", bbox=dict(facecolor='none', edgecolor='black', boxstyle='round,pad=1'), fontsize=8)
ax3.text(8.3,1.05,"nd0_00", bbox=dict(facecolor='none', edgecolor='black', boxstyle='round,pad=1'), fontsize=8)


ax3.set_xlim(0,12)
start, end = ax3.get_xlim()
ax3.xaxis.set_ticks(np.arange(start, end, 1.))
labels = [item.get_text() for item in ax3.get_xticklabels()]
labels[1] = '$Pycorr$'
labels[2] = '$[60-160]$'
labels[3] = '$[25-150]$'
#labels[4] = 'Testing'
labels[5] = '$Pycorr$'
labels[9] = '$Pycorr$'

ax3.set_xticklabels(labels, rotation=0, fontsize=8)
#plt.savefig('../../plots_nden/test_fit_model_nd3_00.png')

plt.show()

