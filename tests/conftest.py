import os, sys
import pytest

# Modify sys.path to include the parent directory
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)

from pySubstructures.ms2lda.common import download_example_data

# Restore sys.path to its original state if needed
sys.path.remove(parent_dir)

@pytest.fixture(scope="session")
def beer_data():
    download_example_data()
    data_path = 'example_data/Beer_multibeers_1_T10_POS.mzML'
    return data_path
