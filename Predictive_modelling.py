import pandas as pd
# machine learning
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV,StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_curve, precision_recall_curve, auc, make_scorer, recall_score, accuracy_score, precision_score, confusion_matrix
from sklearn import svm

class predictive_modelling:
    
    def __init__(self,data,mode):
        self.data = data
        self.mode = mode
        self.writer = pd.ExcelWriter('/Users/anmolgandhi/Desktop/Future/Analysis/model_results.xlsx')
        
    def prepare_dataframe(self):
        target = self.data.loc[:, self.data.dtypes == "category"].columns[0]
        self.X = self.data[target]
        self.Y = self.data.drop([target],1)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.Y, self.X, train_size=0.8, random_state=42)
        
    def prepare_algo(self):
        if(self.mode == "classification"):
            self.algo = [ExtraTreesClassifier(),GaussianNB(),RandomForestClassifier(random_state = 0),
                    LogisticRegression(),GradientBoostingClassifier(),DecisionTreeClassifier(),
                    KNeighborsClassifier()]

        for model in self.algo:
            self.model = model
            self.model.fit(self.X_train,self.y_train)
            self.y_pred = self.model.predict(self.X_test)
            self.report = classification_report(self.y_test, self.y_pred)
            self.classification_report_csv()
        self.writer.save()
    
    def classification_report_csv(self):
        self.report_data = []
        self.lines = self.report.split('\n')
        for self.line in self.lines[2:-3]:
            self.row = {}
            self.row_data = self.line.split(' ')
            self.row_data = list(filter(None, self.row_data))
            self.row['class'] = self.row_data[0]
            self.row['precision'] = float(self.row_data[1])
            self.row['recall'] = float(self.row_data[2])
            self.row['f1_score'] = float(self.row_data[3])
            self.row['support'] = float(self.row_data[4])
            self.report_data.append(self.row)
        self.report = pd.DataFrame.from_dict(self.report_data)
        self.report.to_excel(self.writer,str(self.model).split("(")[0])
        
    def final_model(self,algo):
        self.model = algo
        self.model.fit(self.X_train,self.y_train)
    
    def notification(self,loc):
        y_pred = []
        df = pd.DataFrame(self.X_test.iloc[loc]).T
        columns = df.columns
        y_pred.append(self.model.predict(df))
        print(y_pred[-1][0])
        if(y_pred[-1] == 3):
            print("SOS")
            print("---------------------------------------------")
            print("The person has fallen down")
            print("---------------------------------------------")
            print("Following are the measurements")
            print("---------------------------------------------")
            for i in columns:
                print(i,"=",df[i].values)
            print("---------------------------------------------")
            print("Sending Details and SOS to Emergency contact")
            print("---------------------------------------------")

