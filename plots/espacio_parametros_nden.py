import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns 
import pandas as pd 
 
env='nd3_00'
ranger='60_160'
#ranger='10_130'
#data = np.genfromtxt("../output/2pcf_auto_z0/nd3_00/r_70_150_njack_125_b_ev.dat") 
data = np.genfromtxt("../output/2pcf_auto_z0/"+env+"/gauss_r_"+ranger+"/r_"+ranger+"_gauss_ev.dat")

df=pd.DataFrame(data[:, 0:3], columns=["alpha", "B", "Sigma_nl"])



g = sns.PairGrid(df,corner=True)
#g.map_upper(plt.scatter, s=10)
g.map_diag(sns.histplot)
g.map_lower(sns.kdeplot, cmap="Blues_d", levels=3)
g.map_lower(sns.scatterplot, cmap="Blues_d", marker='.')
plt.show()



