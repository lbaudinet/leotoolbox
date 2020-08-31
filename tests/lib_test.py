# -*- coding: UTF-8 -*-

# Import from standard library
import os
import leotoolbox
import pandas as pd
# Import from our lib
from leotoolbox.lib import get_data, data_projection, data_reconstruction
import pytest


def test_get_data():
    faces = get_data()
    assert faces.images.shape == (1288, 50, 37)
    assert faces.data.shape == (1288, 1850)

def test_data_projection():
    faces = get_data()
    assert data_projection(faces)[0].shape == (1288, 150)

def test_data_reconstruction():
    faces = get_data()
    a,b = data_projection(faces)
    assert data_reconstruction(a,b).shape == (1288, 1850)
