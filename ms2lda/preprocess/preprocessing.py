from loguru import logger

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