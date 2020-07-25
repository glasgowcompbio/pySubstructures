def compute_overlap_scores(lda_dictionary):
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
