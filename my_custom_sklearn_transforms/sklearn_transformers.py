from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import StandardScaler



# All sklearn Transforms must have the `transform` and `fit` methods
class DropColumns(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Primeiro realizamos a cópia do dataframe 'X' de entrada
        data = X.copy()
        # Retornamos um novo dataframe sem as colunas indesejadas
        return data.drop(labels=self.columns, axis='columns')
    



class ImputerFrequent(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns
        self.imputers = {}
        for datacol in self.columns:
            self.imputers[datacol]=SimpleImputer(missing_values=np.nan,strategy='most_frequent',verbose=0,copy=True)

    def fit(self, X, y=None):
        for datacol in self.columns:
            self.imputers[datacol].fit(X.drop(X.drop(datacol ,axis=1).columns ,axis=1))
        return self
    
    def transform(self, X):
        data = X.copy()
        for datacol in self.columns:
            data["imputedFrequent"+datacol]=0
            data.loc[data[datacol].isnull(),"imputedFrequent"+datacol]=1
            data[datacol]=self.imputers[datacol].transform(data.drop(data.drop(datacol ,axis=1).columns ,axis=1))
        return data
    
class ImputerMean(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns
        self.imputers = {}
        for datacol in self.columns:
            self.imputers[datacol]=SimpleImputer(missing_values=np.nan,strategy='mean',verbose=0,copy=True)

    def fit(self, X, y=None):
        for datacol in self.columns:
            self.imputers[datacol].fit(X.drop(X.drop(datacol ,axis=1).columns ,axis=1))
        return self
    
    def transform(self, X):
        data = X.copy()
        for datacol in self.columns:
            data["imputedMean"+datacol]=0
            data.loc[data[datacol].isnull(),"imputedMean"+datacol]=1
            data[datacol]=self.imputers[datacol].transform(data.drop(data.drop(datacol ,axis=1).columns ,axis=1))
        return data
    
class ImputerCategorical(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns
        self.imputers = {}
        for datacol in self.columns:
            self.imputers[datacol]=SimpleImputer(missing_values=np.nan,strategy='constant',verbose=0,copy=True,fill_value="Desconocido")

    def fit(self, X, y=None):
        for datacol in self.columns:
            self.imputers[datacol].fit(X.drop(X.drop(datacol ,axis=1).columns ,axis=1))
        return self
    
    def transform(self, X):
        data = X.copy()
        for datacol in self.columns:
            data["imputedCategorical"+datacol]=0
            data.loc[data[datacol].isnull(),"imputedCategorical"+datacol]=1
            data[datacol]=self.imputers[datacol].transform(data.drop(data.drop(datacol ,axis=1).columns ,axis=1))
        return data
    
    
    
class ImputerDummies(BaseEstimator, TransformerMixin):
    def __init__(self, columns,createColumns=None):
        self.columns = columns
        self.imputers = {}
        for datacol in self.columns:
            self.imputers[datacol]=LabelBinarizer()

    def fit(self, X, y=None):
        for datacol in self.columns:
            self.imputers[datacol].fit(X.drop(X.drop(datacol ,axis=1).columns ,axis=1))
        return self
    
    def transform(self, X):
        data = X.copy()
        for datacol in self.columns:
            if(len(self.imputers[datacol].classes_)==2):
                #print(datacol)
                #print(data)
                imputed=pd.DataFrame(self.imputers[datacol].transform(data.drop(data.drop(datacol ,axis=1).columns ,axis=1)),columns=[datacol+"Dummies"],index=data.index)
            else:
                imputed=pd.DataFrame(self.imputers[datacol].transform(data.drop(data.drop(datacol ,axis=1).columns ,axis=1)),columns=self.imputers[datacol].classes_,index=data.index)
            data.drop(datacol,axis=1,inplace=True)
            data=pd.merge(
                data, imputed, how='inner',
                on=None, left_index=True, right_index=True, sort=False,
                suffixes=('_x', '_y'), copy=True, indicator=False,
                validate=None
            )
        return data
    
    
class ScalerStandard(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.imputers=None
        self.columns=None
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        data = X.copy()
        if(self.imputers==None):
            self.columns=[col for col in data.columns if len(set(data[col]))>2]
            self.imputers={}
            for col in self.columns:
                self.imputers[col] = StandardScaler()        
                self.imputers[col].fit(np.array(data[col]).reshape((-1,1)))
        
        for col in self.columns:
            data[col]=self.imputers[col].transform(np.array(data[col]).reshape((-1,1)))
        return data
    
class capOutliers(BaseEstimator, TransformerMixin):
    """
    cap the rows  whose value does not lies in the specified deviation
    :param columns:
    :param factor:
    :param method: IQR or STD
    :return:
    """
    def __init__(self, columns, factor=1.5, method="IQR"):
        self.columns = columns
        self.method = method
        self.factor = factor
        self.params = {}
        for datacol in self.columns:
            self.params[datacol]={}

    def fit(self, X, y=None):
        for datacol in self.columns:
            if self.method == 'STD':
                permissable_std = self.factor * X[datacol].std()
                col_mean = X[datacol].mean()
                floor, ceil = col_mean - permissable_std, col_mean + permissable_std
            elif self.method == 'IQR':
                Q1 = X[datacol].quantile(0.25)
                Q3 = X[datacol].quantile(0.75)
                IQR = Q3 - Q1
                floor, ceil = Q1 - self.factor * IQR, Q3 + self.factor * IQR
            self.params[datacol]["floor"]=floor
            self.params[datacol]["ceil"]=ceil
        return self
    
    def transform(self, X):
        data = X.copy()
        for datacol in self.columns:
            data[datacol] = data[datacol].clip(self.params[datacol]["floor"], self.params[datacol]["ceil"])
        return data
