import numpy as np
from loguru import logger


class SpectraSampler(object):
    def __init__(self, variational_lda):
        self.vlda = variational_lda
        self.compute_avg_word_count()
        logger.info("Average {} words per document".format(self.mean_word_count))
        self.vocab_size = len(self.vlda.word_index)
        self.K = self.vlda.K
        self.compute_MS1_dist()

    def compute_MS1_dist(self):
        self.ms1_masses = []
        for doc in self.vlda.corpus:
            mass = float(doc.split("_")[0])
            self.ms1_masses.append(mass)

        self.ms1_mass_mean = np.array(self.ms1_masses).mean()
        self.ms1_mass_var = np.array(self.ms1_masses).var()

    def compute_avg_word_count(self):
        self.wcounts = []
        for doc in self.vlda.corpus:
            new_count = 0
            for word in self.vlda.corpus[doc]:
                new_count += int(self.vlda.corpus[doc][word])
            self.wcounts.append(new_count)
        self.mean_word_count = int(np.array(self.wcounts).mean())

        temp = zip(self.vlda.word_index.keys(), self.vlda.word_index.values())
        temp = sorted(temp, key=lambda x: x[1])
        self.reverse_word_index, _ = zip(*temp)

    def generate_spectrum(self, n_words=None, include_losses=False):
        new_spectrum = {}
        beta_copy = self.vlda.beta_matrix.copy()
        if not include_losses:
            for word in self.vlda.word_index:
                if word.startswith("loss"):
                    pos = self.vlda.word_index[word]
                    beta_copy[:, pos] = 0
            beta_copy /= beta_copy.sum(axis=1)[:, None]
        if not n_words:
            # n_words = np.random.poisson(self.mean_word_count)
            n_words = self.wcounts[np.random.choice(len(self.vlda.corpus))]
            logger.info("Generating {} words".format(n_words))
            theta = np.random.dirichlet(self.vlda.alpha)
            s_theta = zip(theta, ["topic_{}".format(i) for i in range(len(theta))])
            s_theta = sorted(s_theta, key=lambda x: x[0], reverse=True)
            print(s_theta[:10])
            for word in range(n_words):
                # Select a topic
                topic = np.random.choice(self.K, p=theta)
                # Select a word
                word_pos = np.random.choice(
                    self.vocab_size, p=self.vlda.beta_matrix[topic, :]
                )
                word = self.reverse_word_index[word_pos]
                if not word in new_spectrum:
                    new_spectrum[word] = 1
                else:
                    new_spectrum[word] += 1
        return new_spectrum
