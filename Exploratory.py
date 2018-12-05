
# 
import pandas as pd

class Exploratory:
    
    def __init__(self, data):
        self.data = data
        self.writer = pd.ExcelWriter('/Users/anmolgandhi/Desktop/Future/Analysis/Exploratory.xlsx')
    
    def unique_vals(self):
        self.name = []
        self.val = []
        for i in self.data.columns:
            uniq = list(set(self.data[i]))
            if(len(uniq) > 2):
                uniq = uniq[0:10]
            self.name.append(i)
            self.val.append(uniq)
        self.temp = pd.DataFrame({"Name": self.name, "Vals": self.val})
        self.temp.to_excel(self.writer,'unique values')
        

    def stats(self):
        self.temp = self.data.describe().T
        self.temp.to_excel(self.writer,'Numerical stats')
        if(self.temp.shape[0] != len(self.data.columns)-1):
            self.temp = self.data.describe(include=["O"]).T
            self.temp.to_excel(self.writer,'categorical stats')
      

    def basic_info(self):
        self._null = []
        self._total = []
        self._non_null = []
        self._distinct = []
        self._category = []
        self.__b = 0
        for i in self.data.columns:
            self._category.append(str(self.data[i].dtype))
            self._total.append(len(self.data[i]))
            self._non_null.append(len(self.data[i].dropna())/len(self.data[i])*100)
            self._null.append(self.data[i].isnull().sum()/len(self.data[i])*100)
            self._distinct.append(len(self.data[i].unique()))
        self.temp = pd.DataFrame({"Name": self.name,"Total rows": self._total, "Non_Null(%)": self._non_null, 
                                      "Null(%)": self._null,"Unique":self._distinct,"Dtype":self._category})
        self.temp.to_excel(self.writer,'Basic Info')
       
        
    def all_stats(self):
        self.unique_vals()
        self.stats()
        self.basic_info()
        self.writer.save()
        
