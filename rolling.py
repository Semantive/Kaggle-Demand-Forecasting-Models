import numpy as np
import pandas as pd
import datetime

# from https://stackoverflow.com/a/21230438
def running_view(arr, window, axis=-1):
    """
    return a running view of length 'window' over 'axis'
    the returned array has an extra last dimension, which spans the window
    """
    shape = list(arr.shape)
    shape[axis] -= (window-1)
    assert(shape[axis]>0)
    return np.lib.index_tricks.as_strided(
        arr,
        shape + [window],
        arr.strides + (arr.strides[axis],),
        writeable=False
    )

def add_dates_to_array(array, start_date):
    """
    Converts a raw array to a Series with DatetimeIndex.
    
    Parameters:
        array: array-like (np.array, pd.Series, etc.)
        start_date : date
            Date corresponding to the first value of the series.
    
    Return value:
        pd.Series containing the same values as the array with a daily DatetimeIndex starting at start_date.
    """
    s = pd.Series(array)
    s.index = pd.DatetimeIndex(start=start_date, freq="D", periods=len(array))
    return s

class Rolling(object):
    """
    A helper for training and forecasting single-variable time series.
    window : int
        Specifies how many days from the past to make available for feature computation.
        Warning: big window may slow-down computation and increase the amount of data for the utility to work - periods shorter than window are not currently supported.
        Default: 365 days
    extract_features : (np.array, date, object) -> np.array
        The function that extracts features from previous values.
        It's arguments are an array of previous values (of length `window`),
        date corresponding to the first value and some arbitrary object that can be passed in other methods.
        Default behaviour is to just take all the values and add no other features.
    pretransform : (np.array, date) -> np.array
        A function called on input data, used to convert it to a supported format.
        It's a good place to call scaler.transform or remove_seasonal_component here.
        Default behavior is to take the data as-is.
    posttransform : (np.array, date) -> np.array
        A function called on generated data, used to undo any conversions for pretransform.
        It's a good place to call scaler.inverse_transform or add_seasonal_component here.
        Default behavior is to pass the input without changes.
    """
    def __init__(self,
                 window=365,
                 extract_features=lambda prev_values, current_date, metadata: prev_values,
                 pretransform=lambda values, start_date: values,
                 posttransform=lambda values, start_date: values):
        self.window = window
        self.extract_features = extract_features
        self.vextract_features = np.vectorize(extract_features)
        self.pretransform = pretransform
        self.posttransform = posttransform

    def make_training_data(self, value_series, metadata=None, start_date=None):
        """
        Generates training data from a raw series of values.
        
        Parameters:
            value_series : np.array or pd.Series
                Values of some time series for forecasting.
            metadata : object
                Any data that you want to be passed to extract_features.
            start_date : date or None
                The date corresponding to the first value in value_series.
                If not set, value_series has to be a pd.Series with a DatetimeIndex with 1 day interval.
        
        Return value: (np.array, np.array)
            Tuple of arrays X and y which are training data in the format supported by libraries like sklearn.
        """
        if start_date is None:
            start_date = value_series.index.min()
        value_series = self.pretransform(value_series, start_date)
        X_base = value_series[:-1] # remove last value
        X = []
        for i, row in enumerate(running_view(X_base, self.window)):
            X.append(self.extract_features(row, start_date + datetime.timedelta(days=i), metadata))
        X = np.array(X)
        y = value_series[self.window:]
        assert len(X) == len(y)
        return X, y

    def predict(self,
                predictor,
                previous_values,
                metadata=None,
                prev_start_date=None,
                fcast_len=90):
        """
        Makes future predictions based on previous data.
        
        Parameters:
            predictor : ([X]) -> [y]
                A function that takes an array containing a single X vector and returning an array containing a single y scalar.
            previous_values: np.array or pd.Series
                Values that are used to kickoff the forecasting. Forecasted values will start on the day following the last value from this series.
            metadata : object
                Any data that you want to be passed to extract_features.
            prev_start_date : date or None
                The date corresponding to the first value in previous_values.
                If not set, previous_values has to be a pd.Series with a DatetimeIndex with 1 day interval.
            fcast_len : int
                The amount of days that will be forecasted (length of output)
        
        Return value:
            np.array of length fcast_len - forecasted values
        """
        if prev_start_date is None:
            prev_start_date = previous_values.index.min()

        fcast_start = prev_start_date + datetime.timedelta(days=len(previous_values))

        # TODO if we cut the window without pretransform we may gain some speed, but be cautious about not breaking prev_start_date
        # pretransform
        previous_values = self.pretransform(previous_values, prev_start_date)

        # we only need one window
        previous_values = previous_values[-self.window:]

        fcast_date = fcast_start
        fcast = np.zeros(fcast_len)
        for i in range(fcast_len):
            X = self.extract_features(previous_values, fcast_date, metadata)
            y = predictor([X])[0]
            fcast[i] = y
            fcast_date += datetime.timedelta(days=1)

            previous_values = np.roll(previous_values, -1)
            previous_values[-1] = y

        return self.posttransform(fcast, fcast_start)


