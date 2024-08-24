from __future__ import print_function, division
from datetime import datetime
from nilmtk.timeframe import merge_timeframes, TimeFrame


class Disaggregator(object):
    """Provides a common interface to all disaggregation classes.

    See https://github.com/nilmtk/nilmtk/issues/755 for discussion

    Attributes
    ----------
    model :
        Each subclass should internally store models learned from training.

    MODEL_NAME : string
        A short name for this type of model.
        e.g. 'CO' for combinatorial optimisation.
    """

    def partial_fit(self, train_mains, train_appliances, **load_kwargs):
        """ Trains the model given a metergroup containing appliance meters
        (supervised) or a site meter (unsupervised).  Will have a
        default implementation in super class.
        train_main: list of pd.DataFrames with pd.DatetimeIndex as index and 1
                    or more power columns
        train_appliances: list of (appliance_name,list of pd.DataFrames) with
                        the same pd.DatetimeIndex as index as train_main and
                        the same 1 or more power columns as train_main
        """
        raise NotImplementedError()

    def disaggregate_chunk(self, test_mains):
        """Passes each chunk from mains generator to disaggregate_chunk()
        Parameters
        ----------
        test_mains : list of pd.DataFrames
        """
        raise NotImplementedError()

    def call_preprocessing(self, train_mains, train_appliances):
        """Calls the preprocessing functions of this algorithm and returns the
           preprocessed data in the same format
        Parameters
        ----------
        train_main: list of pd.DataFrames with pd.DatetimeIndex as index and 1
                    or more power columns
        train_appliances: list of (appliance_name,list of pd.DataFrames) with
                    the same pd.DatetimeIndex as index as train_main and the
                    same 1 or more power columns as train_main
        """
        return train_mains, train_appliances

    def save_model(self, folder_name):
        """Passes each chunk from mains generator to disaggregate_chunk()
        Parameters
        ----------
        test_mains : list of pd.DataFrames
        """
        raise NotImplementedError()

    def load_model(self, folder_name):
        """Passes each chunk from mains generator to disaggregate_chunk()
        Parameters
        ----------
        test_mains : list of pd.DataFrames
        """
        raise NotImplementedError()

