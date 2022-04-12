def compute_overlap_scores_from_dict(lda_dictionary):
    """
    Compute the overlap scores for the lda model in dictionary format
    :param lda_dictionary:
    :return:
    """

    overlap_scores = {}
    for doc, phi in lda_dictionary['phi'].items():
        motifs = lda_dictionary['theta'][doc].keys()
        doc_overlaps = {m: 0.0 for m in motifs}
        for word, probs in phi.items():
            for m in motifs:
                if word in lda_dictionary['beta'][m] and m in probs:
                    doc_overlaps[m] += lda_dictionary['beta'][m][word] * probs[m]
        overlap_scores[doc] = {}
        for m in doc_overlaps:
            overlap_scores[doc][m] = doc_overlaps[m]
    return overlap_scores


def compute_overlap_scores_from_model(vlda):
    import numpy as np
    K = len(vlda.topic_index)
    overlap_scores = {}
    for doc in vlda.doc_index:
        overlap_scores[doc] = {}
        os = np.zeros(K)
        pm = vlda.phi_matrix[doc]
        for word, probs in pm.items():
            word_index = vlda.word_index[word]
            os += probs * vlda.beta_matrix[:, word_index]
        for motif, m_pos in vlda.topic_index.items():
            overlap_scores[doc][motif] = os[m_pos]
    return overlap_scores