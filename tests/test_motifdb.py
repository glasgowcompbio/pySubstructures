import os
import sys

from importlib import resources as impresources
import pytest

# Modify sys.path to include the parent directory for relative imports
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)

# Import pytest and the constants from your module
from pySubstructures.motifdb.main import generate_motif_selections
from pySubstructures.motifdb.main import acquire_motifsets, load_db
from pySubstructures import resources
from pySubstructures.motifdb.constants import GNPS_LIBRARY_DERIVED_MASS2MOTIFS

# Restore sys.path to its original state if needed
sys.path.remove(parent_dir)


# Fixture for loading data from MS2LDA
@pytest.fixture
def ms2lda_data():
    class Args:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    args = Args(
        gnps_motif_include="yes",
        massbank_motif_include="no",
        urine_motif_include="no",
        euphorbia_motif_include="no",
        rhamnaceae_motif_include="no",
        strep_salin_motif_include="no",
        photorhabdus_motif_include="no",
        user_motif_sets=None,
    )

    return acquire_motifsets(args)


# Fixture for loading data from local dump
@pytest.fixture
def loaded_data():
    RESOURCE_DIR = impresources.files(resources)
    db_path = os.path.abspath(RESOURCE_DIR / "MOTIFDB")
    db_list = [GNPS_LIBRARY_DERIVED_MASS2MOTIFS]

    return load_db(db_list, db_path)


# Pytest to check if the generated motif selections match the expected results
def test_motif_selections():
    motif_selections = generate_motif_selections()
    # Expected results for testing
    expected_results = {
        "gnps_motif_include": 2,
        "massbank_motif_include": 4,
        "urine_motif_include": 1,
        "euphorbia_motif_include": 3,
        "rhamnaceae_motif_include": 5,
        "strep_salin_motif_include": 6,
        "photorhabdus_motif_include": 16,
    }
    assert (
        motif_selections == expected_results
    ), "Motif selections do not match expected results"


# Test to compare lengths of loaded and MS2LDA data
def test_data_lengths(ms2lda_data, loaded_data):
    assert len(ms2lda_data[0]) == len(loaded_data[0]), "Spectra lengths differ"
    assert len(ms2lda_data[1]) == len(loaded_data[1]), "Metadata lengths differ"
    assert len(ms2lda_data[2]) == len(loaded_data[2]), "Features lengths differ"


# Test to compare content, ignoring specific keys in metadata
def test_data_content(ms2lda_data, loaded_data):
    ignore_keys = ["motifdb_id", "motifdb_url", "merged"]

    # Spectra
    for key, value in ms2lda_data[0].items():
        assert key in loaded_data[0], f"Key {key} not found in loaded_spectra."
        assert (
            value == loaded_data[0][key]
        ), f"Value mismatch for key {key} between motifdb_spectra and loaded_spectra."

    # Metadata
    for key, value in ms2lda_data[1].items():
        assert key in loaded_data[1], f"Key {key} not found in loaded_metadata."
        for sub_key in value:
            if sub_key not in ignore_keys:
                assert (
                    sub_key in loaded_data[1][key]
                ), f"Sub-key {sub_key} not found in loaded_metadata[{key}]."
                assert (
                    value[sub_key] == loaded_data[1][key][sub_key]
                ), f"Mismatch for sub-key {sub_key} in key {key}."

    # Features
    assert (
        ms2lda_data[2] == loaded_data[2]
    ), "Mismatch in motifdb_features and loaded_features."


if __name__ == "__main__":
    pytest.main()
