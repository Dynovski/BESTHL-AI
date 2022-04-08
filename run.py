import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Lasso, Ridge, SGDRegressor
from sklearn.svm import SVR, LinearSVR
from sklearn.metrics import mean_absolute_error
from typing import List


class DataLoader:
    def __init__(self, data_path: str, attributes: List[str], class_column_name: str):
        self.class_column_name: str = class_column_name
        self.attributes: List[str] = attributes
        self.data: pd.DataFrame = pd.read_csv(data_path, on_bad_lines='skip')
        self.data.fillna(0, inplace=True)

    def get_data(self) -> pd.DataFrame:
        return self.data[self.attributes]

    def get_targets(self) -> pd.Series:
        return self.data[self.class_column_name]


class ModelsEvaluator:
    def __init__(self, x_train, y_train, x_test, y_test):
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test

        self.linear = LinearRegression()
        self.forest = RandomForestRegressor()
        self.lasso = Lasso()
        self.ridge = Ridge()
        self.regressor = SGDRegressor()
        self.svr = SVR()
        self.linear_svr = LinearSVR()

    def evaluate(self):
        self.linear.fit(self.x_train, self.y_train)
        linear_y_pred_train = self.linear.predict(self.x_train)
        linear_y_pred_test = self.linear.predict(self.x_test)
        print('Linear mean squared error on training data:', mean_absolute_error(self.y_train, linear_y_pred_train))
        print('Linear mean squared error on test data:', mean_absolute_error(self.y_test, linear_y_pred_test))

        self.lasso.fit(self.x_train, self.y_train)
        lasso_y_pred_train = self.lasso.predict(self.x_train)
        lasso_y_pred_test = self.lasso.predict(self.x_test)
        print('Lasso mean squared error on training data:', mean_absolute_error(self.y_train, lasso_y_pred_train))
        print('Lasso mean squared error on test data:', mean_absolute_error(self.y_test, lasso_y_pred_test))

        self.ridge.fit(self.x_train, self.y_train)
        ridge_y_pred_train = self.ridge.predict(self.x_train)
        ridge_y_pred_test = self.ridge.predict(self.x_test)
        print('Ridge mean squared error on training data:', mean_absolute_error(self.y_train, ridge_y_pred_train))
        print('Ridge mean squared error on test data:', mean_absolute_error(self.y_test, ridge_y_pred_test))

        self.forest.fit(self.x_train, self.y_train)
        forest_y_pred_train = self.forest.predict(self.x_train)
        forest_y_pred_test = self.forest.predict(self.x_test)
        print('Forest mean squared error on training data:', mean_absolute_error(self.y_train, forest_y_pred_train))
        print('Forest mean squared error on test data:', mean_absolute_error(self.y_test, forest_y_pred_test))

        self.linear_svr.fit(self.x_train, self.y_train)
        linear_svr_y_pred_train = self.linear_svr.predict(self.x_train)
        linear_svr_y_pred_test = self.linear_svr.predict(self.x_test)
        print('Linear SVR mean squared error on training data:', mean_absolute_error(self.y_train, linear_svr_y_pred_train))
        print('Linear SVR mean squared error on test data:', mean_absolute_error(self.y_test, linear_svr_y_pred_test))

        self.svr.fit(self.x_train, self.y_train)
        svr_y_pred_train = self.svr.predict(self.x_train)
        svr_y_pred_test = self.svr.predict(self.x_test)
        print('SVR mean squared error on training data:', mean_absolute_error(self.y_train, svr_y_pred_train))
        print('SVR mean squared error on test data:', mean_absolute_error(self.y_test, svr_y_pred_test))

        self.regressor.fit(self.x_train, self.y_train)
        regressor_y_pred_train = self.regressor.predict(self.x_train)
        regressor_y_pred_test = self.regressor.predict(self.x_test)
        print('Regressor mean squared error on training data:', mean_absolute_error(self.y_train, regressor_y_pred_train))
        print('Regressor mean squared error on test data:', mean_absolute_error(self.y_test, regressor_y_pred_test))


def run_models(path_to_data: str):
    loader = dataloader = DataLoader(
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
    manager.evaluate()


if __name__ == '__main__':
    run_models('./train_data.csv')
