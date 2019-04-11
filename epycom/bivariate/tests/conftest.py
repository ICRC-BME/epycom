# -*- coding: utf-8 -*-
# Copyright (c) St. Anne's University Hospital in Brno. International Clinical
# Research Center, Biomedical Engineering. All Rights Reserved.
# Distributed under the (new) BSD License. See LICENSE.txt for more info.

# Std imports
import pickle

# Third pary imports
import pytest

# Local imports


@pytest.fixture(scope="module")
def get_ieeg_data():
    """
    Reads testing data
    """
    data_dict = pickle.load(open("tests/data/ieeg_data.pkl", "rb"))
    return data_dict
