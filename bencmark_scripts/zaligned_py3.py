import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import matplotlib
import sys
import matplotlib.pyplot as plt

##### VIOLIN PLOT #########
# python3 zaligned.py


######GLOBALS########
plt.rcParams.update({'font.size': 28})
path1 = 'packed_forest_0_1.csv'
path2 = 'compiled_trees_0_1.csv'
path3 = 'sklean_naive_0_1.csv'
scale = 1.85


b0 = np.genfromtxt(path1, delimiter='\n')
b1 = np.genfromtxt(path2, delimiter='\n')
b2 = np.genfromtxt(path3, delimiter='\n')

b0 = b0[~np.isnan(b0)]
b1 = b1[~np.isnan(b1)]
b2 = b2[~np.isnan(b2)]

box_plot_data = [b0, b1, b2]

data = pd.DataFrame({'PF':box_plot_data[0],
                     'CT':box_plot_data[1],
                     'SK-n':box_plot_data[2]
                     })

sns.set(font_scale=scale)
sns.set(style="whitegrid")

ax = sns.violinplot(data=data , split=True, inner = "box" )
    
ax.set(xlabel='Layout', ylabel='Inference Time in (ms)')
plt.tight_layout()
plt.savefig('Layout_vs_ioblocks_'+'.png', dpi=900);
plt.show()
