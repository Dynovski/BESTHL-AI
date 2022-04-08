from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
# from sklearn.linear_model import GammaRegressor
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_squared_error


class ModelsManager:
    def __init__(self, x_train, y_train, x_test, y_test):
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test

        self.linear = LinearRegression()
        self.lasso = Lasso()
        self.ridge = Ridge()
        # self.gamma = GammaRegressor()
        self.elastic = ElasticNet()
        self.forest = RandomForestRegressor()

    def run_models(self):
        self.linear.fit(self.x_train, self.y_train)
        linear_y_pred_train = self.linear.predict(self.x_train)
        linear_y_pred_test = self.linear.predict(self.x_test)
        print('Linear mean squared error on training data:', mean_squared_error(self.y_train, linear_y_pred_train))
        print('Linear mean squared error on test data:', mean_squared_error(self.y_test, linear_y_pred_test))

        self.lasso.fit(self.x_train, self.y_train)
        lasso_y_pred_train = self.lasso.predict(self.x_train)
        lasso_y_pred_test = self.lasso.predict(self.x_test)
        print('Lasso mean squared error on training data:', mean_squared_error(self.y_train, lasso_y_pred_train))
        print('Lasso mean squared error on test data:', mean_squared_error(self.y_test, lasso_y_pred_test))

        self.ridge.fit(self.x_train, self.y_train)
        ridge_y_pred_train = self.ridge.predict(self.x_train)
        ridge_y_pred_test = self.ridge.predict(self.x_test)
        print('Ridge mean squared error on training data:', mean_squared_error(self.y_train, ridge_y_pred_train))
        print('Ridge mean squared error on test data:', mean_squared_error(self.y_test, ridge_y_pred_test))

        # self.gamma.fit(self.x_train, self.y_train)
        # gamma_y_pred_train = self.gamma.predict(self.x_train)
        # gamma_y_pred_test = self.gamma.predict(self.x_test)
        # print('Gamma mean squared error on training data:', mean_squared_error(self.y_train, gamma_y_pred_train))
        # print('Gamma mean squared error on test data:', mean_squared_error(self.y_test, gamma_y_pred_test))

        self.elastic.fit(self.x_train, self.y_train)
        elastic_y_pred_train = self.elastic.predict(self.x_train)
        elastic_y_pred_test = self.elastic.predict(self.x_test)
        print('Elastic mean squared error on training data:', mean_squared_error(self.y_train, elastic_y_pred_train))
        print('Elastic mean squared error on test data:', mean_squared_error(self.y_test, elastic_y_pred_test))

        self.forest.fit(self.x_train, self.y_train)
        forest_y_pred_train = self.forest.predict(self.x_train)
        forest_y_pred_test = self.forest.predict(self.x_test)
        print('Forest mean squared error on training data:', mean_squared_error(self.y_train, forest_y_pred_train))
        print('Forest mean squared error on test data:', mean_squared_error(self.y_test, forest_y_pred_test))

