import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
import matplotlib.transforms as mtransforms


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

nsv_jack = [16,32]
for idx, nsv in enumerate(nsv_jack):
    if (nsv==16):njack='16_3'
    if (nsv==32):njack='32_3'

    file = np.genfromtxt('../../get_env/n_density/test_jackknife/files/pycorr_z0_nd3_00_xi_jkn_'+str(nsv)+'_3_r_70_150.dat')

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

    fit = np.genfromtxt('../output/2pcf_auto_z0/nd3_00/r_70_150_njack_'+str(njack)+'_model_mean.dat')
    ii=np.where((fit[:,0]<160) & (fit[:,0]>30)) 
    rfit = fit[:,0][ii]
    xifit = fit[:,1][ii]
   # ax2.plot(rfit,rfit**2 * xifit, label='fit mean_'+str(njack),  marker='.')
    fit = np.genfromtxt('../output/2pcf_auto_z0/nd3_00/r_70_150_njack_'+str(njack)+'_model_maxlike.dat')
    ii=np.where((fit[:,0]<160) & (fit[:,0]>60)) 
    rfit = fit[:,0][ii]
    xifit = fit[:,1][ii]
    ax2.plot(rfit,rfit**2 * xifit, color='C'+str(idx))


    ax2.legend(loc = 'lower left')

##############################################################################

################  nd 3_00 #####################################
#r [70,150]
nsv = [5,8]#,16,32]
alpha = [0.989143850825359605E+00, 0.992579667578962876E+00]#, 0.989571947419984332E+00, 0.992715844972345218E+00]
err_alpha=[0.202165423187788892E-01, 0.267253423780814681E-01]#, 0.279428107113747597E-01, 0.290501070335324532E-01]
#alpha_b = [0.992408728702855347E+00, 0.993032250003943573E+00, 0.990075834142475908E+00, 0.989737408453017831E+00]
#err_alpha_b=[0.556673755356241898E-03, 0.392508390765457769E-03, 0.580331000206931041E-01, 0.626432407655688195E-01]
ax3.errorbar(nsv, alpha, yerr=err_alpha, label='[70,150]', fmt="o", color='C0')
#ax3.errorbar(nsv, alpha_b, yerr=err_alpha_b)

#r [60,160]
nsv = [5.1,8.1]
al = [0.989174560697012484E+00,0.986222170079396609E+00]
err_al = [0.170585367196523369E-01, 0.242586733709745095E-01]
ax3.errorbar(nsv, al, yerr=err_al, label='[60,160]', fmt="o", color='C1')

#r [30,170]
nsv = [5.2,8.2]
alpha = [0.100318535020550192E+01, 0.969332891858893042E+00]
err_alpha=[0.133753250647854118E-01, 0.278153174915375875E-01]
ax3.errorbar(nsv, al, yerr=err_al, label='[30,170]', fmt="o", color='C2')

#r [20,200]
#nsv = [5.3,8.3]
#al = [0.990635309405448305E+00, 0.982193544173814215E+00]
#err_al = [0.144496747707993211E-01, 0.180450521405499976E-01]
#ax3.errorbar(nsv, al, yerr=err_al, label='[20,200]', fmt="o")
##############################################################

################  nd 2_00 #####################################
#r [50,170]
#nsv = [6,9]
#al = [, ]
#err_al = [, ]
#ax3.errorbar(nsv, al, yerr=err_al, label='[20,200]', fmt="o", color='C0')

#r [60,160]
nsv = [6.1,9.1]
al = [0.996468238219410996E+00, 0.993100703781965910E+00]
err_al = [0.152699988503018653E-01, 0.194684800204054931E-01]
ax3.errorbar(nsv, al, yerr=err_al, fmt="o", color='C1')

#r [30,170]
nsv = [6.2,9.2]
al = [0.100042660535782346E+01, 0.995615692767555127E+00]
err_al = [0.151209974711211340E-01, 0.196172723467254034E-01]
ax3.errorbar(nsv, al, yerr=err_al, fmt="o", color='C2')

#r [20,200]
#nsv = [6.3,9.3]
#al = [, ]
#err_al = [, ]
#ax3.errorbar(nsv, al, yerr=err_al, label='[20,200]', fmt="o")
##############################################################


################  nd 0_00 #####################################
#r [50,170]
nsv = [7,10]
al = [0.992304286801488078E+00, 0.992111505908539026E+00]
err_al = [0.816717341227586125E-03, 0.494559243681163998E-03]
#ax3.errorbar(nsv, al, yerr=err_al, fmt="o", color='C0')

#create chart 
#ax3.bar(x = np.arange (len (al)), height = al, yerr = err_al, width = 0.5)
#ax3.set_xticks([r for r in range(len(al))], ['5^3', '8^3'])

ax3.legend(loc='center')

ax3.set_ylim(0.95,1.03)
#plt.savefig('../../plots_nden/test_fit_model_nd3_00.png')

plt.show()

