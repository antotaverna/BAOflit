import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
import matplotlib.transforms as mtransforms
import pandas as pd
import seaborn as sns



#-------------------------------------------------------------
fig=plt.figure(figsize=(15,5))
#fig.subplots_adjust(top = 0.9)
gs = gridspec.GridSpec(2,4, wspace=0.5, hspace=0.3)#, width_ratios=[2, 1])
ax1 = plt.subplot(gs[0,0:2])
ax2 = plt.subplot(gs[1,0:2])
ax3 = plt.subplot(gs[0,2])
ax4 = plt.subplot(gs[1,2])
ax5 = plt.subplot(gs[0,3])
ax6 = plt.subplot(gs[1,3])

def f1(a1,a2,a3,x):
    return a1/(x**2) + a2/x + a3

nsv_jack = [5]
for idx, nsv in enumerate(nsv_jack):
    if (nsv==5):njack=125
    if (nsv==8):njack=512

    file = np.genfromtxt('../../get_env/n_density/test_jackknife/files/pycorr_z0_nd3_00_xi_jkn_'+str(nsv)+'_3_r_30_150.dat')
    fit = np.genfromtxt('../output/2pcf_auto_z0/nd3_00/r_30_150_njack_'+str(njack)+'_model_mean.dat')

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
    #ax1.set_xlim(70,150)
    ax1.set_ylim(-100,100)
    ax1.set_title("Auto-corr [nd = 0.001]")
    #ax1.set_xticklabels([''])
    ax1.set_ylabel(r'$r^2 \xi(r)$')
    ax1.set_xlabel(r'$r \, [h^{-1}Mpc]$')
    ax1.text(40,70,"r [30-150] Mpc/h nsv=125", bbox=dict(facecolor='none', edgecolor='black', boxstyle='round,pad=1'))

    ii=np.where((fit[:,0]<160) & (fit[:,0]>30)) 
    rfit = fit[:,0][ii]
    xifit = fit[:,1][ii]
    ax1.plot(rfit,rfit**2 * xifit, label='fit mean_'+str(njack), marker='.')
    fit = np.genfromtxt('../output/2pcf_auto_z0/nd3_00/r_30_150_njack_'+str(njack)+'_model_maxlike.dat')
    ii=np.where((fit[:,0]<160) & (fit[:,0]>30)) 
    rfit = fit[:,0][ii]
    xifit = fit[:,1][ii]
    ax1.plot(rfit,rfit**2 * xifit, 'g--', label='fit maxlike_'+str(njack))#,  marker='.',color='g')
    fit = np.genfromtxt('../output/2pcf_auto_z0/nd3_00/r_30_150_njack_'+str(njack)+'_model_map.dat')
    ii=np.where((fit[:,0]<160) & (fit[:,0]>30)) 
    rfit = fit[:,0][ii]
    xifit = fit[:,1][ii]
    #ax1.plot(rfit,rfit**2 * xifit, label='fit map_'+str(njack),  marker='.',color='m')

    fit = np.genfromtxt('../output/2pcf_auto_z0/nd3_00/r_30_150_njack_'+str(njack)+'_taper_model_mean.dat')
    ii=np.where((fit[:,0]<160) & (fit[:,0]>30)) 
    rfit = fit[:,0][ii]
    xifit = fit[:,1][ii]
    ax1.plot(rfit,rfit**2 * xifit, label='fit mean_'+str(njack)+'_taper',  marker='.',color='y')

    #nuisanse parameters
    a1=-7.285999093
    a2=0.3180680197
    a3=-0.005191553816
    x=np.arange(30,170,1)
    y=f1(a1,a2,a3,x)
    #ax1.plot(x,x**2*y, label='A(s)', color='red')#, marker='.')

    a1=-13.05108044
    a2=0.4805136648
    a3=-0.005052943607
    x=np.arange(30,170,1)
    y=f1(a1,a2,a3,x)
    #ax1.plot(x,x**2*y, label='A(s) taper', color='y')#, marker='.')


    #fortran--------------
    #if(nsv==5):
        #fit = np.genfromtxt('../output/2pcf_auto_z0/nd3_00/r_30_150_njack_'+str(njack)+'_fortran_model_mean.dat')
        #ii=np.where((fit[:,0]<160) & (fit[:,0]>60)) 
        #rfit = fit[:,0][ii]
        #xifit = fit[:,1][ii]
        #ax1.plot(rfit,rfit**2 * xifit, label='fit mean_'+str(njack)+'fortran',  marker='.',color='r')
        #fit = np.genfromtxt('../output/2pcf_auto_z0/nd3_00/r_30_150_njack_'+str(njack)+'_fortran_model_maxlike.dat')
        #ii=np.where((fit[:,0]<160) & (fit[:,0]>60)) 
        #rfit = fit[:,0][ii]
        #xifit = fit[:,1][ii]
        #ax1.plot(rfit,rfit**2 * xifit,'r--', label='fit maxlike_'+str(njack)+'fortran')
        #fit = np.genfromtxt('../output/2pcf_auto_z0/nd3_00/r_30_150_njack_'+str(njack)+'_fortran_model_map.dat')
        #ii=np.where((fit[:,0]<160) & (fit[:,0]>60)) 
        #rfit = fit[:,0][ii]
        #xifit = fit[:,1][ii]
        #ax1.plot(rfit,rfit**2 * xifit,'r-.', label='fit map_'+str(njack)+'fortran')

    ax1.legend(loc = 'lower left',fontsize=7)

##############################################################################

nsv_jack = [8]
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
    ax2.errorbar(r, r**2*xi, yerr=err2, color='C'+str(idx),linestyle='None',label='2pcf njack='+str(njack))
    ax2.plot(r, r**2 * xi, marker='o', color='gray') #all
    y=r**2*xi
    #ax1.set_xlim(70,150)
    ax2.set_ylim(-100,100)
    ax2.set_ylabel(r'$r^2 \xi(r)$')
    ax2.set_xlabel(r'$r \, [h^{-1}Mpc]$')
    ax2.text(40,70,"r [30-150] Mpc/h nsv=512", bbox=dict(facecolor='none', edgecolor='black', boxstyle='round,pad=1'))
    #ax2.plot(file2[:,0], file2[:,0]**2 * file2[:,1], marker='o',label='2pcf njack=512',color='gray') #all

    fit = np.genfromtxt('../output/2pcf_auto_z0/nd3_00/r_30_150_njack_'+str(njack)+'_model_mean.dat')
    ii=np.where((fit[:,0]<160) & (fit[:,0]>30)) 
    rfit = fit[:,0][ii]
    xifit = fit[:,1][ii]
    ax2.plot(rfit,rfit**2 * xifit, label='fit mean_'+str(njack),  marker='.')
    fit = np.genfromtxt('../output/2pcf_auto_z0/nd3_00/r_30_150_njack_'+str(njack)+'_model_maxlike.dat')
    ii=np.where((fit[:,0]<160) & (fit[:,0]>30)) 
    rfit = fit[:,0][ii]
    xifit = fit[:,1][ii]
    ax2.plot(rfit,rfit**2 * xifit, 'g--', label='fit maxlike_'+str(njack))#,  marker='.',color='g')
    fit = np.genfromtxt('../output/2pcf_auto_z0/nd3_00/r_30_150_njack_'+str(njack)+'_model_map.dat')
    ii=np.where((fit[:,0]<160) & (fit[:,0]>30)) 
    rfit = fit[:,0][ii]
    xifit = fit[:,1][ii]
    #ax2.plot(rfit,rfit**2 * xifit, label='fit map_'+str(njack),  marker='.',color='m')

    fit = np.genfromtxt('../output/2pcf_auto_z0/nd3_00/r_30_150_njack_'+str(njack)+'_taper_model_mean.dat')
    ii=np.where((fit[:,0]<160) & (fit[:,0]>30)) 
    rfit = fit[:,0][ii]
    xifit = fit[:,1][ii]
    ax2.plot(rfit,rfit**2 * xifit, label='fit mean_'+str(njack)+'_taper',  marker='.',color='y')

    #nuisanse parameters
    a1=-10.98588407
    a2=0.3490658031
    a3=-0.002316500647
    x=np.arange(30,170,1)
    y=f1(a1,a2,a3,x)
    #ax2.plot(x,x**2*y, label='A(s)', color='red')#, marker='.')

    a1=-12.27708696
    a2=0.4311536555
    a3=-0.002433069441
    x=np.arange(30,170,1)
    y=f1(a1,a2,a3,x)
    #ax2.plot(x,x**2*y, label='A(s) taper', color='y')#, marker='.')


    ax2.legend(loc = 'lower left',fontsize=7)

###################################### cov matrix
#cov matrix of autocorr nd=0.001
cov = np.genfromtxt("../../get_env/n_density/test_jackknife/files/pycorr_z0_nd3_00_cov_jkn_5_3_r_30_150.dat")
cov2 = np.genfromtxt("../../get_env/n_density/test_jackknife/files/pycorr_z0_nd3_00_cov_jkn_8_3_r_30_150.dat")

M = cov
M2 = cov2
print(np.shape(cov))
#--------------tapering
def tapering(dist,Tp):
    if(dist < Tp):
        K = (1. - dist/Tp)**4 *(4. * dist/Tp + 1.)
    elif(dist >= Tp):
        K = 0. 
    return K

tt=[]
ct=[]
def cov_taper(cc,Tp):
    tt=cc*0.
    ct=cc*0.
    nbin=len(cc[0,:])
    for i in range(nbin):
        for j in range(nbin):
            d=np.abs(i-j)
            ct[i,j]=cc[i,j]*tapering(d,Tp)
            tt[i,j]=tapering(d,Tp)
    return(ct)

Tp=70.

#profucto elemento a elemento
Mt=cov_taper(M,Tp)
M2t=cov_taper(M2,Tp)
#print(cov_t)

#convert_cov('files/nd0_00_cov_8.bin',cov)
#x, v = np.linalg.eig(data)

nbin=24

D = np.zeros((nbin, nbin))
N = np.zeros((nbin, nbin))
diagM = np.diag(M)
D2 = np.zeros((nbin, nbin))
N2 = np.zeros((nbin, nbin))
diagM2 = np.diag(M2)
Dt = np.zeros((nbin, nbin))
Nt = np.zeros((nbin, nbin))
diagMt = np.diag(Mt)
D2t = np.zeros((nbin, nbin))
N2t = np.zeros((nbin, nbin))
diagM2t = np.diag(M2t)


D3 = np.zeros((nbin, nbin))
Dt3 = np.zeros((nbin, nbin))
N3 = np.zeros((nbin, nbin))


for row in range(len(M)): 
    # iterate through columns 
    for column in range(len(M[0])): 

        D[row][column] = (M[row][row] * M[column][column])**0.5
        D2[row][column] = (M2[row][row] * M2[column][column])**0.5
        Dt[row][column] = (Mt[row][row] * Mt[column][column])**0.5
        D2t[row][column] = (M2t[row][row] * M2t[column][column])**0.5
        #print('Dividendo', D[row][column])

        N[row][column] = M[row][column]/D[row][column]
        N2[row][column] = M2[row][column]/D2[row][column]
        Nt[row][column] = Mt[row][column]/Dt[row][column]
        N2t[row][column] = M2t[row][column]/D2t[row][column]
        #print('Nomralization', N[row][column])

vmin = 0.
vmax = 1.
center = 0.5

rango = np.linspace(30,150,25) + 2.5
rango = rango[0:24]

df = pd.DataFrame(N, columns=rango, index=rango)
df2 = pd.DataFrame(N2, columns=rango, index=rango)
dft = pd.DataFrame(Nt, columns=rango, index=rango)
df2t = pd.DataFrame(N2t, columns=rango, index=rango)

# plotting the heatmap
hm = sns.heatmap(data=df,
                vmin=vmin,
                vmax=vmax,
                center=center,
                cmap='coolwarm',
                ax=ax3)

hm = sns.heatmap(data=df2,
                vmin=vmin,
                vmax=vmax,
                center=center,
                cmap='coolwarm',
                ax=ax4) 

# plotting the heatmap
hm = sns.heatmap(data=dft,
                vmin=vmin,
                vmax=vmax,
                center=center,
                cmap='coolwarm',
                ax=ax5)

hm = sns.heatmap(data=df2t,
                vmin=vmin,
                vmax=vmax,
                center=center,
                cmap='coolwarm',
                ax=ax6)
#plt.savefig('../../plots_nden/test_fit_model_nd3_00_fortran_b.png')

plt.show()

