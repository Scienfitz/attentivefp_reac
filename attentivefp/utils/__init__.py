from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.impute import KNNImputer
from copy import copy


class TabularPP(object):
    def __init__(self, scaler='Robust', imputer='KNN', nNeighbors=5):
        if scaler == 'Standard':
            self.scaler_impute = StandardScaler()
            self.scaler_scale = StandardScaler()
        elif scaler == 'Robust':
            self.scaler_impute = RobustScaler()
            self.scaler_scale = RobustScaler()
        else:
            assert scaler in ['Standard', 'Robust'], 'The scaler selected must be either "Standard" oder "Robust"'

        if imputer == 'KNN':
            self.imputer = KNNImputer(n_neighbors=nNeighbors)
        else:
            assert imputer in ['KNN'], 'The imputer selected can only be "KNN"'

    def fit(self, data):
        # Columns that only have 0 and 1s (like Fingerprints) should not be scaled
        cols = data.columns[
            ~data.apply(lambda x: set(np.unique(x)) in [set([0.0, 1.0]), set([0.0, 1.0, np.nan])], axis=0)]
        self.cols = cols

        data = copy(data)
        x = data[cols].values

        self.scaler_impute.fit(x)
        x = self.scaler_impute.transform(x)
        x = self.imputer.fit_transform(x)
        x = self.scaler_impute.inverse_transform(x)
        x = self.scaler_scale.fit_transform(x)

    def transform(self, data):
        cols = self.cols
        data = copy(data)

        x = data[cols].values
        x = self.scaler_impute.transform(x)
        x = self.imputer.transform(x)
        x = self.scaler_impute.inverse_transform(x)
        x = self.scaler_scale.transform(x)
        data[cols] = x

        return data