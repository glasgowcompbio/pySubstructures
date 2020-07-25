import bisect

from loguru import logger


class MakeFeatures(object):
    """
    Abstract feature making class
    """

    def make_features(self, ms2):
        raise NotImplementedError("make features method must be implemented")


class MakeRawFeatures(MakeFeatures):
    """
    Class to make non-processed features
    i.e. no feature matching, just the raw value
    """

    def __str__(self):
        return 'Raw feature maker'

    def make_features(self, ms2):
        self.word_mz_range = {}
        self.corpus = {}

        for peak in ms2:
            frag_mz = peak[0]
            frag_intensity = peak[2]
            parent = peak[3]
            doc_name = parent.name
            file_name = peak[4]

            if not file_name in self.corpus:
                self.corpus[file_name] = {}
            if not doc_name in self.corpus[file_name]:
                self.corpus[file_name][doc_name] = {}

            feature_name = 'fragment_{}'.format(frag_mz)
            self.corpus[file_name][doc_name][feature_name] = frag_intensity
            self.word_mz_range[feature_name] = (frag_mz, frag_mz)

        return self.corpus, self.word_mz_range


class MakeBinnedFeatures(MakeFeatures):
    """
    Class to make features by binning with width bin_width
    """

    def __str__(self):
        return "Binning feature creator with bin_width = {}".format(self.bin_width)

    def __init__(self, min_frag=0.0, max_frag=1000.0,
                 min_loss=10.0, max_loss=200.0,
                 min_intensity=0.0, min_intensity_perc=0.0, bin_width=0.005):
        self.min_frag = min_frag
        self.max_frag = max_frag
        self.min_loss = min_loss
        self.max_loss = max_loss
        self.min_intensity = min_intensity
        self.bin_width = bin_width
        self.min_intensity_perc = min_intensity_perc

    def make_features(self, ms2):
        self.word_mz_range = {}
        self.word_counts = {}
        self.fragment_words = []
        self.loss_words = []
        self.corpus = {}

        self._make_words(self.min_loss, self.max_loss, self.loss_words, 'loss')
        self._make_words(self.min_frag, self.max_frag, self.fragment_words, 'fragment')

        for word in self.fragment_words:
            self.word_counts[word] = 0
        for word in self.loss_words:
            self.word_counts[word] = 0

        # make a list of the lower edges
        frag_lower = [self.word_mz_range[word][0] for word in self.fragment_words]
        loss_lower = [self.word_mz_range[word][0] for word in self.loss_words]

        for peak in ms2:

            # MS2 objects are ((mz,rt,intensity,parent,file_name,id))
            # TODO: make the search more efficients
            mz = peak[0]
            if peak[3].mz == None:
                # There isnt a precursor mz so we cant do losses
                do_losses = False
                loss_mz = 0.0
            else:
                do_losses = True
                if peak[3].single_charge_precursor_mass:
                    loss_mz = peak[3].single_charge_precursor_mass - mz
                else:
                    loss_mz = peak[3].mz - mz
            intensity = peak[2]
            if intensity >= self.min_intensity:
                doc_name = peak[3].name
                file_name = peak[4]
                if mz > self.min_frag and mz < self.max_frag:
                    pos = bisect.bisect_right(frag_lower, mz)
                    word = self.fragment_words[pos - 1]
                    if not file_name in self.corpus:
                        self.corpus[file_name] = {}
                    if not doc_name in self.corpus[file_name]:
                        self.corpus[file_name][doc_name] = {}
                    if not word in self.corpus[file_name][doc_name]:
                        self.corpus[file_name][doc_name][word] = 0.0
                        self.word_counts[word] += 1
                    self.corpus[file_name][doc_name][word] += intensity

                if do_losses and loss_mz > self.min_loss and loss_mz < self.max_loss:
                    pos = bisect.bisect_right(loss_lower, loss_mz)
                    word = self.loss_words[pos - 1]
                    if not file_name in self.corpus:
                        self.corpus[file_name] = {}
                    if not doc_name in self.corpus[file_name]:
                        self.corpus[file_name][doc_name] = {}
                    if not word in self.corpus[file_name][doc_name]:
                        self.corpus[file_name][doc_name][word] = 0.0
                        self.word_counts[word] += 1
                    self.corpus[file_name][doc_name][word] += intensity

        # TODO: Test code to remove blank words!!!!!
        to_remove = []
        for word in self.word_mz_range:
            if self.word_counts[word] == 0:
                to_remove.append(word)

        for word in to_remove:
            del self.word_mz_range[word]

        n_docs = 0
        for c in self.corpus:
            n_docs += len(self.corpus[c])

        logger.info("{} documents".format(n_docs))
        logger.info("After removing empty words, {} words left".format(len(self.word_mz_range)))

        if self.min_intensity_perc > 0:
            # Remove words that are smaller than a certain percentage of the highest feature
            for c in self.corpus:
                for doc in self.corpus[c]:
                    max_intensity = 0.0
                    for word in self.corpus[c][doc]:
                        intensity = self.corpus[c][doc][word]
                        if intensity > max_intensity:
                            max_intensity = intensity
                    to_remove = []
                    for word in self.corpus[c][doc]:
                        intensity = self.corpus[c][doc][word]
                        if intensity < max_intensity * self.min_intensity_perc:
                            to_remove.append(word)
                    for word in to_remove:
                        del self.corpus[c][doc][word]
                        self.word_counts[word] -= 1
            # Remove any words that are now empty
            to_remove = []
            for word in self.word_mz_range:
                if self.word_counts[word] == 0:
                    to_remove.append(word)
            for word in to_remove:
                del self.word_mz_range[word]
            logger.info("After applying min_intensity_perc filter, {} words left".format(len(self.word_mz_range)))

        return self.corpus, self.word_mz_range

    def _make_words(self, min_mz, max_mz, word_list, prefix):
        # Bin the ranges to make the words
        min_word = float(min_mz)
        max_word = min_word + self.bin_width
        while min_word < max_mz:
            up_edge = min(max_mz, max_word)
            word_mean = 0.5 * (min_word + up_edge)
            new_word = '{}_{:.4f}'.format(prefix, word_mean)  # 4dp
            word_list.append(new_word)
            self.word_mz_range[new_word] = (min_word, up_edge)
            min_word += self.bin_width
            max_word += self.bin_width


class MakeNominalFeatures(MakeFeatures):
    """
    Class to make nominal (i.e. integer features)
    """

    def __str__(self):
        return "Nominal feature extractor, bin_width = {}".format(self.bin_width)

    def __init__(self, min_frag=0.0, max_frag=10000.0,
                 min_loss=10.0, max_loss=200.0,
                 min_intensity=0.0, bin_width=0.3):
        self.min_frag = min_frag
        self.max_frag = max_frag
        self.min_loss = min_loss
        self.max_loss = max_loss
        self.min_intensity = min_intensity
        self.bin_width = bin_width

    def make_features(self, ms2):
        # Just make integers between min and max values and assign to corpus
        word_names = []
        word_mz_range = {}
        self.corpus = {}
        for frag in ms2:
            parentmass = frag[3].mz
            frag_mass = frag[0]
            loss_mass = parentmass - frag_mass
            intensity = frag[2]
            doc_name = frag[3].name
            file_name = frag[4]

            if intensity >= self.min_intensity:
                if frag_mass >= self.min_frag and frag_mass <= self.max_frag:
                    frag_word = round(frag_mass)
                    err = abs(frag_mass - frag_word)
                    if err <= self.bin_width:
                        # Keep it
                        word_name = 'fragment_' + str(frag_word)
                        if not word_name in word_names:
                            word_names.append(word_name)
                            word_mz_range[word_name] = (frag_word - self.bin_width, frag_word + self.bin_width)
                        self._add_word_to_corpus(word_name, file_name, doc_name, intensity)

                if loss_mass >= self.min_loss and loss_mass <= self.max_loss:
                    loss_word = round(loss_mass)
                    err = abs(loss_mass - loss_word)
                    if err <= self.bin_width:
                        word_name = 'loss_' + str(loss_word)
                        if not word_name in word_names:
                            word_names.append(word_name)
                            word_mz_range[word_name] = (loss_word - self.bin_width, loss_word + self.bin_width)
                        self._add_word_to_corpus(word_name, file_name, doc_name, intensity)

        return self.corpus, word_mz_range

    def _add_word_to_corpus(self, word_name, file_name, doc_name, intensity):
        if not file_name in self.corpus:
            self.corpus[file_name] = {}
        if not doc_name in self.corpus[file_name]:
            self.corpus[file_name][doc_name] = {}
        if not word_name in self.corpus[file_name][doc_name]:
            self.corpus[file_name][doc_name][word_name] = intensity
        else:
            self.corpus[file_name][doc_name][word_name] += intensity
