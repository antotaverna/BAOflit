import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns 
import pandas as pd 
 
tpcf='cross'
env='pweb'
str='sheets'
ranger='60_160'
#ranger='10_130'

if tpcf=='cross':
    #data = np.genfromtxt("../output/2pcf_auto_z0/nd3_00/r_70_150_njack_125_b_ev.dat") 
    data = np.genfromtxt("../output/2pcf_"+tpcf+"_z0/env_"+env+"/"+str+"_r_"+ranger+"/r_"+ranger+"_gauss_ev.dat")

    df=pd.DataFrame(data[:, 0:4], columns=["alpha", "B", "B2", "Sigma_nl"])

if tpcf=='auto':
    data = np.genfromtxt("../output/2pcf_"+tpcf+"_z0/env_"+env+"/"+str+"_r_"+ranger+"/r_"+ranger+"_gauss_ev.dat")
    df=pd.DataFrame(data[:, 0:3], columns=["alpha", "B", "Sigma_nl"])



g = sns.PairGrid(df,corner=True)
#g.map_lower(plt.scatter, s=10)
g.map_diag(sns.histplot)
g.map_lower(sns.kdeplot, cmap="Blues_d", levels=3)
plt.show()



