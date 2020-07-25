import json

from loguru import logger

from .preprocess import LoadMZML, LoadMSP, LoadMGF, MakeBinnedFeatures
from .topic_modelling import VariationalLDA


class MS2LDAFeatureExtractor(object):
    """
    Convenience class to perform data loading and feature extraction for MS2LDA analysis
    """

    def __init__(self, input_set, loader, feature_maker):
        self.input_set = input_set
        self.loader = loader
        print(self.loader)
        self.feature_maker = feature_maker
        print(self.feature_maker)
        print("Loading spectra")
        self.ms1, self.ms2, self.metadata = self.loader.load_spectra(self.input_set)
        print("Creating corpus")
        self.corpus, self.word_mz_range = self.feature_maker.make_features(self.ms2)

    def get_first_corpus(self):
        first_file_name = self.corpus.keys()[0]
        return self.corpus[first_file_name]


def msfile_to_corpus(ms2_file, ms2_format, min_ms1_intensity, min_ms2_intensity, mz_tol, rt_tol, feature_set_name, K,
                     corpus_json=None):
    if ms2_format == 'mzxml':
        loader = LoadMZML(mz_tol=mz_tol,
                          rt_tol=rt_tol, peaklist=None,
                          min_ms1_intensity=min_ms1_intensity,
                          min_ms2_intensity=min_ms2_intensity)
    elif ms2_format == 'msp':
        loader = LoadMSP(min_ms1_intensity=min_ms1_intensity,
                         min_ms2_intensity=min_ms2_intensity,
                         mz_tol=mz_tol,
                         rt_tol=rt_tol,
                         peaklist=None,
                         name_field="")
    elif ms2_format == 'mgf':
        loader = LoadMGF(min_ms1_intensity=min_ms1_intensity,
                         min_ms2_intensity=min_ms2_intensity,
                         mz_tol=mz_tol,
                         rt_tol=rt_tol,
                         peaklist=None,
                         name_field="")
    else:
        raise NotImplementedError('Unknown ms2 format')

    logger.info('Loading %s using %s' % (ms2_file, loader))
    ms1, ms2, metadata = loader.load_spectra([ms2_file])

    bin_widths = {'binned_005': 0.005,
                  'binned_01': 0.01,
                  'binned_05': 0.05,
                  'binned_1': 0.1,
                  'binned_5': 0.5}
    bin_width = bin_widths[feature_set_name]
    logger.info('bin_width = %f' % bin_width)

    fm = MakeBinnedFeatures(bin_width=bin_width)
    logger.info('Using %s to make features' % fm)
    corpus, features = fm.make_features(ms2)
    first_key = list(corpus.keys())[0]
    corpus = corpus[first_key]

    # To insert in db some additional data is generated inVariationalLDA
    vlda = VariationalLDA(corpus=corpus, K=K)
    lda_dict = {
        'corpus': corpus,
        'word_index': vlda.word_index,
        'doc_index': vlda.doc_index,
        'doc_metadata': metadata,
        'topic_index': vlda.topic_index,
        'topic_metadata': vlda.topic_metadata,
        'features': features
    }

    if corpus_json is not None:
        logger.info('Saving lda_dict to %s' % corpus_json)
        with open(corpus_json, 'w') as f:
            json.dump(lda_dict, f)

    return lda_dict
