import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns 
import pandas as pd 
 
data = np.genfromtxt("../output/2pcf_auto_z0/nd3_00/r_70_150_njack_125_b_ev.dat") 

df=pd.DataFrame(data[:, 0:3], columns=["alpha", "B", "Sigma_nl"])



g = sns.PairGrid(df, palette=["red"],corner=True)
#g.map_upper(plt.scatter, s=10)
g.map_diag(sns.histplot)
g.map_lower(sns.kdeplot, cmap="Blues_d", levels=5)
plt.show()



