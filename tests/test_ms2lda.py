import os
import sys

import numpy as np
import random
from loguru import logger

# Modify sys.path to include the parent directory for relative imports
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)

# Importing necessary modules from the local package
from pySubstructures.ms2lda.main import msfile_to_corpus
from pySubstructures.ms2lda.lda_variational import VariationalLDA
from pySubstructures.motifdb.main import acquire_motifdb

# Restore sys.path to its original state if needed
sys.path.remove(parent_dir)


def test_feature_extraction(beer_data):
    """
    Test feature extraction process to ensure all necessary data elements
    are present in the lda_dict.
    """
    ms2_format = 'mzML'
    corpus_json = 'test_data/beer1pos_test.json'
    lda_dict = msfile_to_corpus(beer_data, ms2_format, min_ms1_intensity=0, min_ms2_intensity=50, mz_tol=5.0,
                                rt_tol=10.0, feature_set_name='binned_005', K=300, corpus_json=corpus_json)
    # Assertions to validate the extraction results
    necessary_keys = ['corpus', 'word_index', 'doc_index', 'doc_metadata', 'topic_index', 'topic_metadata', 'features']
    for key in necessary_keys:
        assert key in lda_dict


def test_lda_run(beer_data):
    """
    Test the LDA (Latent Dirichlet Allocation) model run with a given dataset.
    This function ensures that the model initializes properly and produces expected outputs,
    verifying the integrity and functionality of the LDA modeling process.
    """
    logger.info('Running feature extraction')
    ms2_format = 'mzML'
    corpus_json = 'test_data/beer1pos_test.json'
    lda_dict = msfile_to_corpus(beer_data, ms2_format, min_ms1_intensity=0, min_ms2_intensity=50, mz_tol=5.0,
                                rt_tol=10.0, feature_set_name='binned_005', K=300, corpus_json=corpus_json)

    vlda = VariationalLDA(corpus=lda_dict['corpus'], K=300, normalise=1000.0)
    vlda.run_vb(n_its=3, initialise=True)

    # Validate the basic structural properties of the LDA model.
    assert vlda.n_docs == len(lda_dict['corpus']), "The number of documents should match the corpus size."
    assert vlda.K == 300, "The number of topics (K) should match the predefined value."
    assert vlda.n_words == len(lda_dict['word_index']), "The number of unique words should match the word index size."

    # Ensure all initial topic-document distribution values (alpha) are positive.
    assert all(vlda.alpha > 0), "All alpha values must be positive to represent a valid distribution."

    # Confirm the shape of the document-topic distribution matrix.
    assert vlda.gamma_matrix.shape == (
        vlda.n_docs, vlda.K), "The gamma matrix must match the number of documents and topics."
    assert np.all(vlda.gamma_matrix >= 0), "All values in the gamma matrix should be non-negative."

    # Validate the shape and normalization of the topic-word distribution matrix.
    assert vlda.beta_matrix.shape == (
        vlda.K, vlda.n_words), "The beta matrix must match the number of topics and unique words."
    # Check that each topic's word distribution sums to 1 for a valid probability distribution.
    assert all(np.isclose(vlda.beta_matrix.sum(axis=1), 1)), "Each topic's word distribution should sum to 1."

    assert vlda.n_fixed_topics == 0, "No fixed topics should be present for this basic LDA test."


def test_motifdb_interaction():
    """
    Test interaction with the motif database, ensuring that data is successfully
    acquired and valid.
    """
    # Example IDs for motifs, adjust as necessary
    motifset_ids = [1, 2, 3, 4, 5, 6, 16]  # Example motif set IDs
    motifdb_spectra, motifdb_metadata, _ = acquire_motifdb(motifset_ids)
    assert len(motifdb_spectra) > 0 and len(motifdb_metadata) > 0


def test_lda_run_with_fixed_motifs(beer_data):
    """
    Test the LDA model run incorporating fixed motifs, verifying the correct integration
    of fixed topics into the model and ensuring the model adapts to include these predefined
    topics alongside the dynamically generated ones.
    """
    logger.info('Running feature extraction with fixed motifs')
    # Prepare motif data to be integrated as fixed topics in the LDA model.
    # This is crucial for incorporating domain-specific knowledge or predefined topics.
    motifset_ids = [1, 2, 3, 4, 5, 6, 16]  # Example motif set IDs
    motif_metadata = {}
    motif_features = set()
    motifdb_spectra_dict = {}

    # Acquire motif data for each motif set ID
    for motif_id in motifset_ids:
        spectra, metadata, features = acquire_motifdb([motif_id])
        motifdb_spectra_dict.update(spectra)
        motif_metadata.update(metadata)
        motif_features.update(features)

    ms2_format = 'mzML'
    corpus_json = 'test_data/beer1pos_test.json'
    lda_dict = msfile_to_corpus(beer_data, ms2_format, min_ms1_intensity=0, min_ms2_intensity=50, mz_tol=5.0,
                                rt_tol=10.0, feature_set_name='binned_005', K=300, corpus_json=corpus_json)

    vlda = VariationalLDA(corpus=lda_dict['corpus'], K=300, normalise=1000.0,
                          fixed_topics=motifdb_spectra_dict, fixed_topics_metadata=motif_metadata)
    vlda.run_vb(n_its=3, initialise=True)

    # Assertions to confirm LDA model's adaptation to fixed motifs
    assert vlda.n_docs == len(lda_dict['corpus']), "The number of documents must be consistent with the corpus size."
    assert vlda.K == 300 + len(
        motifdb_spectra_dict), "The total number of topics should include both LDA-generated and fixed motifs."
    assert all(
        vlda.alpha > 0), "Alpha values, representing topic-document distributions, must be positive for a valid model."

    assert vlda


def set_random_seed(seed_value=42):
    """
    Sets the random seed for numpy and random to ensure reproducibility.
    """
    np.random.seed(seed_value)
    random.seed(seed_value)


def test_lda_reproducibility(beer_data):
    """
    Test the reproducibility of the LDA model by running it twice with the same
    settings and data, and comparing the results.
    """
    set_random_seed()  # Ensure a consistent seed for reproducibility

    # First run
    lda_dict_first_run = msfile_to_corpus(beer_data, 'mzML', min_ms1_intensity=0, min_ms2_intensity=50, mz_tol=5.0,
                                          rt_tol=10.0, feature_set_name='binned_005', K=300,
                                          corpus_json='test_data/beer1pos_test.json')
    vlda_first_run = VariationalLDA(corpus=lda_dict_first_run['corpus'], K=300, normalise=1000.0)
    vlda_first_run.run_vb(n_its=3, initialise=True)

    set_random_seed()  # Reset the seed before the second run to ensure identical initial conditions

    # Second run
    lda_dict_second_run = msfile_to_corpus(beer_data, 'mzML', min_ms1_intensity=0, min_ms2_intensity=50, mz_tol=5.0,
                                           rt_tol=10.0, feature_set_name='binned_005', K=300,
                                           corpus_json='test_data/beer1pos_test.json')
    vlda_second_run = VariationalLDA(corpus=lda_dict_second_run['corpus'], K=300, normalise=1000.0)
    vlda_second_run.run_vb(n_its=3, initialise=True)

    # Enhanced assertions for thorough reproducibility checks
    assert np.allclose(vlda_first_run.gamma_matrix, vlda_second_run.gamma_matrix, atol=1e-5), \
        "Document-topic distributions (gamma_matrix) must be identical across runs, within floating-point tolerance."

    assert np.allclose(vlda_first_run.beta_matrix, vlda_second_run.beta_matrix, atol=1e-5), \
        "Topic-word distributions (beta_matrix) must be identical across runs, within floating-point tolerance."

    assert vlda_first_run.n_docs == vlda_second_run.n_docs, \
        "The number of documents (n_docs) should be identical across runs."

    assert vlda_first_run.K == vlda_second_run.K, \
        "The number of topics (K) should be identical across runs."

    assert vlda_first_run.n_words == vlda_second_run.n_words, \
        "The number of unique words (n_words) should be identical across runs."

    assert np.allclose(vlda_first_run.alpha, vlda_second_run.alpha, atol=1e-5), \
        "Topic concentration parameters (alpha) must be identical across runs, within floating-point tolerance."

    assert np.allclose(vlda_first_run.eta, vlda_second_run.eta, atol=1e-5), \
        "Word concentration parameters (eta) must be identical across runs, within floating-point tolerance."


def test_lda_reproducibility_with_fixed_motifs(beer_data):
    """
    Test the reproducibility of the LDA model with fixed motifs by running it twice with the same
    settings and data, and comparing the results.
    """
    set_random_seed()  # Ensure a consistent seed for reproducibility

    # Hardcoded list of motifset ids for fixed motifs
    motifset_ids = [1, 2, 3, 4, 5, 6, 16]  # adjust as per your project's needs
    motif_metadata = {}
    motif_features = set()
    motifdb_spectra_dict = {}
    for motif_id in motifset_ids:
        motifdb_spectra, motifdb_metadata, motifdb_features = acquire_motifdb([motif_id])
        motifdb_spectra_dict.update(motifdb_spectra)
        motif_metadata.update(motifdb_metadata)
        motif_features.update(motifdb_features)

    # First run
    lda_dict_first_run = msfile_to_corpus(beer_data, 'mzML', min_ms1_intensity=0, min_ms2_intensity=50, mz_tol=5.0,
                                          rt_tol=10.0, feature_set_name='binned_005', K=300,
                                          corpus_json='test_data/beer1pos_test.json')
    vlda_first_run = VariationalLDA(corpus=lda_dict_first_run['corpus'], K=300 + len(motifdb_spectra_dict),
                                    normalise=1000.0, fixed_topics=motifdb_spectra_dict,
                                    fixed_topics_metadata=motif_metadata)
    vlda_first_run.run_vb(n_its=3, initialise=True)

    set_random_seed()  # Reset the seed before the second run to ensure identical initial conditions

    # Second run
    lda_dict_second_run = msfile_to_corpus(beer_data, 'mzML', min_ms1_intensity=0, min_ms2_intensity=50, mz_tol=5.0,
                                           rt_tol=10.0, feature_set_name='binned_005', K=300,
                                           corpus_json='test_data/beer1pos_test.json')
    vlda_second_run = VariationalLDA(corpus=lda_dict_second_run['corpus'], K=300 + len(motifdb_spectra_dict),
                                     normalise=1000.0, fixed_topics=motifdb_spectra_dict,
                                     fixed_topics_metadata=motif_metadata)
    vlda_second_run.run_vb(n_its=3, initialise=True)

    # Enhanced assertions for thorough reproducibility checks
    assert np.allclose(vlda_first_run.gamma_matrix, vlda_second_run.gamma_matrix, atol=1e-5), \
        "Document-topic distributions (gamma_matrix) must be identical across runs, within floating-point tolerance."

    assert np.allclose(vlda_first_run.beta_matrix, vlda_second_run.beta_matrix, atol=1e-5), \
        "Topic-word distributions (beta_matrix) must be identical across runs, within floating-point tolerance."

    assert vlda_first_run.n_docs == vlda_second_run.n_docs, \
        "The number of documents (n_docs) should be identical across runs."

    assert vlda_first_run.K == vlda_second_run.K, \
        "The number of topics (K) should be identical across runs."

    assert vlda_first_run.n_words == vlda_second_run.n_words, \
        "The number of unique words (n_words) should be identical across runs."

    assert np.allclose(vlda_first_run.alpha, vlda_second_run.alpha, atol=1e-5), \
        "Topic concentration parameters (alpha) must be identical across runs, within floating-point tolerance."

    assert np.allclose(vlda_first_run.eta, vlda_second_run.eta, atol=1e-5), \
        "Word concentration parameters (eta) must be identical across runs, within floating-point tolerance."

    # Additional assertions
    assert vlda_first_run.n_fixed_topics == vlda_second_run.n_fixed_topics, \
        "The number of fixed topics (n_fixed_topics) should be identical across runs."
