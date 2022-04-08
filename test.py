import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from typing import List, Dict, Any
from models.regression import ModelsManager


class DataLoader:
    def __init__(self, data_path: str, attributes: List[str], class_column_name: str):
        self.class_column_name: str = class_column_name
        self.attributes: List[str] = attributes
        self.data: pd.DataFrame = pd.read_csv(data_path, on_bad_lines='skip')
        self.data.dropna(how='all', axis=1, inplace=True)
        non_number = []
        for col in self.data:
            if self.data[col].dtypes != "float64" and self.data[col].dtypes != "int64":
                non_number.append(col)
        self.data = self.data.drop(columns=non_number)
        self.attributes = [attribute for attribute in self.attributes if attribute not in non_number]
        self.data.fillna(0, inplace=True)

    def get_data(self) -> pd.DataFrame:
        return self.data[self.attributes]

    def get_labels(self) -> pd.Series:
        return self.data[self.class_column_name]


def run_models(path_to_data: str, normalize: bool = False):
    loader = dataloader = DataLoader(
        path_to_data,
        ['BBLE', 'BORO', 'BLOCK', 'LOT', 'EASEMENT', 'OWNER', 'BLDGCL',
         'TAXCLASS', 'LTFRONT', 'LTDEPTH', 'EXT', 'STORIES', 'AVLAND',
         'AVTOT', 'EXLAND', 'EXTOT', 'EXCD1', 'STADDR', 'POSTCODE', 'EXMPTCL',
         'BLDFRONT', 'BLDDEPTH', 'AVLAND2', 'AVTOT2', 'EXLAND2', 'EXTOT2',
         'EXCD2', 'PERIOD', 'YEAR', 'VALTYPE', 'Borough', 'Latitude',
         'Longitude', 'Community Board', 'Council District', 'Census Tract',
         'BIN', 'NTA', 'New Georeferenced Column'],
        'FULLVAL'
    )

    x_train, x_test, y_train, y_test = train_test_split(
        dataloader.get_data(),
        dataloader.get_labels(),
        test_size=0.3,
        random_state=42
    )

    if normalize:
        scaler = StandardScaler()
        scaler.fit(x_train)
        x_train = scaler.transform(x_train)
        x_test = scaler.fit(x_test)

    manager = ModelsManager(x_train, y_train, x_test, y_test)
    manager.run_models()


if __name__ == '__main__':
    run_models('./train_data.csv', False)
