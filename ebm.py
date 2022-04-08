import pandas as pd
from interpret import set_visualize_provider
from interpret.provider import InlineProvider
from interpret.glassbox import ExplainableBoostingClassifier
set_visualize_provider(InlineProvider())
from interpret import show
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_curve, auc
import matplotlib.pyplot as plt
from datetime import datetime
import pytz
from geopy.geocoders import Nominatim
import pycountry
from sklearn import preprocessing
from interpret.glassbox import LogisticRegression
from interpret import show


if __name__ == '__main__':
    data = pd.read_csv("train_data.csv", on_bad_lines="skip")
    data=data[['BORO', 'BLOCK', 'LOT',
       'TAXCLASS', 'LTFRONT', 'LTDEPTH', 'STORIES', 'FULLVAL', 'AVLAND',
       'AVTOT', 'EXLAND', 'EXTOT', 'EXCD1', 'POSTCODE',
       'BLDFRONT', 'BLDDEPTH', 'AVLAND2', 'AVTOT2',
         'YEAR', 'Latitude',
       'Longitude', 'Community Board', 'Census Tract'
       ]].fillna(0)
    feature_columns = data.columns.copy()
    X = data[['BORO', 'BLOCK', 'LOT',
        'LTFRONT', 'LTDEPTH', 'STORIES', 'AVLAND',
       'AVTOT', 'EXLAND', 'EXTOT', 'EXCD1', 'POSTCODE',
       'BLDFRONT', 'BLDDEPTH', 'AVLAND2', 'AVTOT2',
          'Latitude',
       'Longitude', 'Community Board', 'Census Tract'
       ]].fillna(0)
    y = data['FULLVAL']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

    #label_encoders = []
    #for c in train_df.columns:
    #    pass
        # Label encoding here

    # EBM logistic regression
    lr = LogisticRegression(random_state=420)
    lr.fit(X_train, y_train)

    y_pred = lr.predict(X_test)
    print(accuracy_score(y_pred=y_pred, y_true=y_test))