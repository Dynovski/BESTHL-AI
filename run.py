import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Lasso, Ridge, SGDRegressor
from sklearn.svm import SVR, LinearSVR
#from sklearn.metrics import mean_absolute_error
from typing import List
import matplotlib.pyplot as plt
import numpy as np
import joblib
from sklearn.decomposition import PCA
import standarizing

def mean_absolute_error(x, y):
    return np.mean(np.abs(x - y))

class DataLoader:
    def __init__(self, data_path: str, attributes: List[str], class_column_name: str):
        self.class_column_name: str = class_column_name
        self.attributes: List[str] = attributes
        self.data: pd.DataFrame = pd.read_csv(data_path, error_bad_lines=False)
        self.data.fillna(0, inplace=True)

    def get_data(self) -> pd.DataFrame:
        #return self.data[self.attributes]
        return self.data.drop(columns=self.class_column_name)

    def get_targets(self) -> pd.Series:
        return self.data[self.class_column_name]


class ModelsEvaluator:
    def __init__(self, x_train, y_train, x_test, y_test):
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test

        self.linear = LinearRegression()
        self.forest = RandomForestRegressor(min_samples_split=4, min_samples_leaf=2, n_estimators=1000)

        self.lasso = Lasso()
        self.ridge = Ridge()
        self.regressor = SGDRegressor()
        self.svr = SVR()
        self.linear_svr = LinearSVR()
        
    def reverse_transform_y(self, y):
        y = standarizing.reverse_transform(y, 'standard_scaler_y.joblib')
        y = standarizing.reverse_log(y)
        return y

    def evaluate(self):
        self.linear.fit(self.x_train, self.y_train)
        models = {
            'Linear': self.linear,
            'Lasso': self.lasso,
            'Ridge': self.ridge,
            'RF': self.forest,
            'Linear SVR': self.linear_svr,
            'SVR': self.svr,
            'SGDRegressor': self.regressor
            }
        
        preds_train = {}
        preds_test = {}
        results_train = {}
        results_test = {}
        
        y_train = self.reverse_transform_y(self.y_train)
        y_test = self.reverse_transform_y(self.y_test)
        
        for key in models:
            model = models[key]
            model.fit(self.x_train, self.y_train)
            y_pred_train = model.predict(self.x_train)
            y_pred_test = model.predict(self.x_test)
            
            y_pred_train = standarizing.reverse_transform(y_pred_train, 'standard_scaler_y.joblib')
            y_pred_test = standarizing.reverse_transform(y_pred_test, 'standard_scaler_y.joblib')
            y_pred_train = standarizing.reverse_log(y_pred_train)
            y_pred_test = standarizing.reverse_log(y_pred_test)
            
            preds_train[key] = y_pred_train
            preds_test[key] = y_pred_test
            
            error_train = mean_absolute_error(y_train, y_pred_train)
            results_train[key] = error_train
            error_test = mean_absolute_error(y_test, y_pred_test)
            results_test[key] = error_test
            print(key, 'mean absolute error on traininig data:', error_train)
            print(key, 'mean absolute error on test data:', error_test)
        
        return preds_train, preds_test, results_train, results_test


def run_models(path_to_data: str):
    dataloader = DataLoader(
        path_to_data,
        ['BORO', 'BLOCK', 'LOT', 'LTFRONT', 'LTDEPTH', 'STORIES', 'AVLAND',
         'AVTOT', 'EXLAND', 'EXTOT',
         'BLDFRONT', 'BLDDEPTH', 'AVLAND2', 'AVTOT2', 'EXLAND2', 'EXTOT2',
         'Latitude',
         'Longitude', 'Council District'],
        'FULLVAL'
    )

    x_train, x_test, y_train, y_test = train_test_split(
        dataloader.get_data(),
        dataloader.get_targets(),
        test_size=0.3,
    )

    manager = ModelsEvaluator(x_train, y_train, x_test, y_test)
    preds_train, preds_test, res_train, res_test = manager.evaluate()
    return preds_train, preds_test, res_train, res_test, manager


if __name__ == '__main__':
    #run_models('./train_data.csv')
    preds_train, preds_test, res_train, res_test, manager = run_models('./processed.csv')
    res_train = pd.DataFrame(res_train, index=range(len(res_train)))
    res_test = pd.DataFrame(res_test, index=range(len(res_test)))
    
    importances = manager.forest.feature_importances_
    imp_sorted = list(reversed(sorted(importances)))
    dl = DataLoader('./processed.csv', [], 'FULLVAL')
    df_imp_sorted = dl.get_data().loc[:, reversed([dl.get_data().columns[idx] for idx in importances.argsort()])]
    
    errors = np.abs(preds_test['RF'] - manager.y_test)
    plt.scatter(manager.y_test, errors)
    
    y_test = manager.y_test

    #pca = PCA(n_components=100)
    #df_filtered = pd.DataFrame(pca.fit_transform(dl.get_data()))
    #df_filtered['FULLVAL'] = dl.get_targets()
    #df_filtered.to_csv('./filtered.csv', index=0)