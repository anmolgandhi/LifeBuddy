#%%
import pandas as pd

import Exploratory
import descriptive_visuals
import Predictive_modelling as pm
import pandas as pd
data1 = pd.read_csv('~/Desktop/Future/falldeteciton.csv', encoding='latin-1')
data1["ACTIVITY"] = data1["ACTIVITY"].astype("category")

a = Exploratory.Exploratory(data1)
a.all_stats()
#%%
b = descriptive_visuals.descriptive_visuals(data1)
b.numerical_graphs()
#%%
c = pm.predictive_modelling(data1,"classification")
c.prepare_dataframe() 
c.prepare_algo()  

#%%

from sklearn.ensemble import RandomForestClassifier
d = pm.predictive_modelling(data1,"classification")
d.prepare_dataframe() 
model = RandomForestClassifier(random_state = 0)
d.final_model(model)

#%%
from random import randint
d.notification(0)
print("\n")
for i in range(0,10):
    d.notification(randint(0, len(d.X_test)))
    print("\n")