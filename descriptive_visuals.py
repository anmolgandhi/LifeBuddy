


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
class descriptive_visuals:
    
    def __init__(self,data):
        self.data = data
        
    def numerical_graphs(self):
        for i in self.data.columns:
            if((self.data[i].dtype == "int64") or (self.data[i].dtype == "float64")):
                plt.figure(i)
                plt.subplot(2,1,1)
                self.temp = sns.distplot(self.data[i])
                plt.subplot(2,1,2)
                self.temp = sns.boxplot(self.data[i])
                plt.savefig(os.path.join("/Users/anmolgandhi/Desktop/Future/graphs/Numerical_graphs", str(i)+"_numplot"))
                plt.close()

    

        
    


