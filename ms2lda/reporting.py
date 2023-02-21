import pickle
import sys

import numpy as np
import pandas as pd
from tqdm.auto import tqdm


def match_topics_across_dictionaries(lda1=None, lda2=None, file1=None, file2=None,
                                     same_corpus=True, copy_annotations=False, copy_threshold=0.5,
                                     summary_file=None,
                                     new_file2=None, mass_tol=5.0):
    # finds the closest topic matches from lda2 to lda1
    if lda1 == None:
        if file1 == None:
            print("Must specify either an lda dictionary object or a dictionary file for lda1")
            return
        else:
            with open(file1, 'r') as f:
                lda1 = pickle.load(f)
                print("Loaded lda1 from {}".format(file1))
    if lda2 == None:
        if file2 == None:
            print("Must specify either an lda dictionary object or a dictionary file for lda1")
            return
        else:
            with open(file2, 'r') as f:
                lda2 = pickle.load(f)
                print("Loaded lda2 from {}".format(file2))

    word_index = lda1['word_index']
    n_words = len(word_index)
    n_topics1 = lda1['K']
    n_topics2 = lda2['K']

    # Put lda1's topics into a nice matrix
    beta = np.zeros((n_topics1, n_words), np.float)
    topic_pos = 0
    topic_index1 = {}
    for topic in lda1['beta']:
        topic_index1[topic] = topic_pos
        for word in lda1['beta'][topic]:
            word_pos = word_index[word]
            beta[topic_pos, word_pos] = lda1['beta'][topic][word]
        topic_pos += 1

    # Make the reverse index
    ti = [(topic, topic_index1[topic]) for topic in topic_index1]
    ti = sorted(ti, key=lambda x: x[1])
    reverse1, _ = zip(*ti)

    if not same_corpus:
        fragment_masses = np.array(
            [float(f.split('_')[1]) for f in word_index if f.startswith('fragment')])
        fragment_names = [f for f in word_index if f.startswith('fragment')]
        loss_masses = np.array(
            [float(f.split('_')[1]) for f in word_index if f.startswith('loss')])
        loss_names = [f for f in word_index if f.startswith('loss')]

    beta /= beta.sum(axis=1)[:, None]
    best_match = {}
    temp_topics2 = {}
    for topic2 in lda2['beta']:
        temp_topics2[topic2] = {}
        temp_beta = np.zeros((1, n_words))
        if same_corpus:
            total_probability = 0.0
            for word in lda2['beta'][topic2]:
                word_pos = word_index[word]
                temp_beta[0, word_pos] = lda2['beta'][topic2][word]
                temp_topics2[topic2][word] = lda2['beta'][topic2][word]
                total_probability += temp_topics2[topic2][word]
            for word in temp_topics2[topic2]:
                temp_topics2[topic2][word] /= total_probability
            temp_beta /= temp_beta.sum(axis=1)[:, None]
        else:
            # we need to match across corpus
            total_probability = 0.0
            for word in lda2['beta'][topic2]:
                # try and match to a word in word_index
                split_word = word.split('_')
                word_mass = float(split_word[1])
                if split_word[0].startswith('fragment'):
                    ppm_errors = 1e6 * np.abs((fragment_masses - word_mass) / fragment_masses)
                    smallest_pos = ppm_errors.argmin()
                    if ppm_errors[smallest_pos] < mass_tol:
                        word1 = fragment_names[smallest_pos]
                        temp_topics2[topic2][word1] = lda2['beta'][topic2][word]
                        temp_beta[0, word_index[word1]] = lda2['beta'][topic2][word]
                if split_word[0].startswith('loss'):
                    ppm_errors = 1e6 * np.abs((loss_masses - word_mass) / loss_masses)
                    smallest_pos = ppm_errors.argmin()
                    if ppm_errors[smallest_pos] < 2 * mass_tol:
                        word1 = loss_names[smallest_pos]
                        temp_topics2[topic2][word1] = lda2['beta'][topic2][word]
                        temp_beta[0, word_index[word1]] = lda2['beta'][topic2][word]
                total_probability += lda2['beta'][topic2][word]
            for word in temp_topics2[topic2]:
                temp_topics2[topic2][word] /= total_probability
            temp_beta /= total_probability

        match_scores = np.dot(beta, temp_beta.T)
        best_score = match_scores.max()
        best_pos = match_scores.argmax()

        topic1 = reverse1[best_pos]
        w1 = lda1['beta'][topic1].keys()
        if same_corpus:
            w2 = lda2['beta'][topic2].keys()
        else:
            w2 = temp_topics2[topic2].keys()
        union = set(w1) | set(w2)
        intersect = set(w1) & set(w2)
        p1 = 0.0
        p2 = 0.0
        for word in intersect:
            word_pos = word_index[word]
            p1 += beta[topic_index1[topic1], word_pos]
            p2 += temp_topics2[topic2][word]

        annotation = ""
        if 'topic_metadata' in lda1:
            if topic1 in lda1['topic_metadata']:
                if type(lda1['topic_metadata'][topic1]) == str:
                    annotation = lda1['topic_metadata'][topic1]
                else:
                    annotation = lda1['topic_metadata'][topic1].get('annotation', "")
        best_match[topic2] = (topic1, best_score, len(union), len(intersect), p2, p1, annotation)

    if summary_file:
        with open(summary_file, 'w') as f:
            f.write('lda2_topic,lda1_topic,match_score,unique_words,shared_words,'
                    'shared_p_lda2,shared_p_lda1,lda1_annotation\n')
            for topic2 in best_match:
                topic1 = best_match[topic2][0]
                line = "{},{},{}".format(topic2, topic1, best_match[topic2][1])
                line += ",{},{}".format(best_match[topic2][2], best_match[topic2][3])
                line += ",{},{}".format(best_match[topic2][4], best_match[topic2][5])
                line += ",{}".format(best_match[topic2][6])
                f.write(line + '\n')

    if copy_annotations and 'topic_metadata' in lda1:
        print("Copying annotations")
        if not 'topic_metadata' in lda2:
            lda2['topic_metadata'] = {}
        for topic2 in best_match:
            lda2['topic_metadata'][topic2] = {'name': topic2}
            topic1 = best_match[topic2][0]
            p2 = best_match[topic2][4]
            p1 = best_match[topic2][5]
            if p1 >= copy_threshold and p2 >= copy_threshold:
                annotation = best_match[topic2][6]
                if len(annotation) > 0:
                    lda2['topic_metadata'][topic2]['annotation'] = annotation
        if new_file2 is None:
            with open(file2, 'wb') as f:
                pickle.dump(lda2, f)
            print("Dictionary with copied annotations saved to {}".format(file2))
        else:
            with open(new_file2, 'wb') as f:
                pickle.dump(lda2, f)
            print("Dictionary with copied annotations saved to {}".format(new_file2))

    return best_match, lda2


def alpha_report(vlda, overlap_scores=None, overlap_thresh=0.3):
    ta = []
    for topic, ti in vlda.topic_index.items():
        ta.append((topic, vlda.alpha[ti]))
    ta = sorted(ta, key=lambda x: x[1], reverse=True)
    for t, a in ta:
        to = []
        if overlap_scores:
            for doc in overlap_scores:
                if t in overlap_scores[doc]:
                    if overlap_scores[doc][t] >= overlap_thresh:
                        to.append((doc, overlap_scores[doc][t]))
        print(t, vlda.topic_metadata[t].get('SHORT_ANNOTATION', None), a)
        to = sorted(to, key=lambda x: x[1], reverse=True)
        for t, o in to:
            print('\t', t, o)


def write_csv(vlda, overlap_scores, filename, metadata, p_thresh=0.01, o_thresh=0.3):
    import csv
    probs = vlda.get_expect_theta()
    motif_dict = {}
    with open(filename, 'w') as f:
        writer = csv.writer(f)
        heads = ['Document', 'Motif', 'Probability', 'Overlap Score', 'Precursor Mass',
                 'Retention Time', 'Document Annotation']
        writer.writerow(heads)
        all_rows = []
        for doc, doc_pos in vlda.doc_index.items():
            for motif, motif_pos in vlda.topic_index.items():
                if probs[doc_pos, motif_pos] >= p_thresh and overlap_scores[doc][
                    motif] >= o_thresh:
                    new_row = []
                    new_row.append(doc)
                    new_row.append(motif)
                    new_row.append(probs[doc_pos, motif_pos])
                    new_row.append(overlap_scores[doc][motif])
                    new_row.append(metadata[doc]['parentmass'])
                    new_row.append("None")
                    new_row.append(metadata[doc]['featid'])
                    all_rows.append(new_row)
                    motif_dict[motif] = True
        all_rows = sorted(all_rows, key=lambda x: x[0])
        for new_row in all_rows:
            writer.writerow(new_row)
    return motif_dict


def write_summary_file(lda_dictionary, filename):
    import csv
    with open(filename, 'w') as f:
        writer = csv.writer(f)
        heads = ['clusterID', 'motif', 'probability', 'overlap']
        writer.writerow(heads)
        for doc, motifs in lda_dictionary['theta'].items():
            for motif, probability in motifs.items():
                row = [doc, motif, probability,
                       lda_dictionary['overlap_scores'][doc].get(motif, 0.0)]
                writer.writerow(row)


def write_topic_report(lda_dictionary, filename, backend='agg'):
    # import pylab as plt
    import matplotlib
    matplotlib.use(backend)
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    with PdfPages(filename) as pdf:

        topic_list = lda_dictionary['beta'].keys()
        # TODO fix motif order in report
        # topic_list = zip(topic_list,[int(t.split('_')[1]) for t in topic_list])
        # topic_list.sort(key = lambda x: x[1])
        # topic_list,_ = zip(*topic_list)
        for topic in tqdm(topic_list):
            # sys.stdout.write(topic + ' ')
            # sys.stdout.flush()

            word_probs = lda_dictionary['beta'][topic]
            plt.figure(figsize=(20, 10))
            for word, prob in word_probs.items():
                word_type, word_mz = word.split('_')
                word_mz = float(word_mz)
                if word_type == 'fragment':
                    plt.plot([word_mz, word_mz], [0, prob], 'r')
                    if prob >= 0.025:
                        plt.text(word_mz, prob, '{:.0f}'.format(word_mz))
                if word_type == 'loss':
                    plt.plot([word_mz, word_mz], [0, -prob], 'g')
                    if prob >= 0.025:
                        plt.text(word_mz, -prob - 0.005, '{:.0f}'.format(word_mz))

            plt.plot(plt.xlim(), [0, 0], 'k--')
            plt.xlabel('m/z')
            plt.ylabel('probability')
            plt.title('{}'.format(topic))
            pdf.savefig()
            plt.close()

            # add some text
            textPage = plt.figure(figsize=(20, 10))
            textPage.clf()
            topic_probs = list(zip(lda_dictionary['beta'][topic].keys(),
                              lda_dictionary['beta'][topic].values()))
            topic_probs.sort(key=lambda x: x[1], reverse=True)
            col = 0
            n_rows = 20
            textPage.text(0.3, 0.9, "{}".format(topic), size=24)
            for i, (word, prob) in enumerate(topic_probs):
                if prob < 0.005:
                    break
                rowpos = i % n_rows
                textPage.text(0.15 + (col) * 0.2, 0.8 - rowpos / 40.0,
                              "{:30s}: {:.2f}".format(word, prob),
                              transform=textPage.transFigure, size=12)
                if rowpos == n_rows - 1:
                    col += 1

            pdf.savefig()
            plt.close()


def write_motifs_in_scans(vd, metadata, overlap_thresh, p_thresh, X, motifdb_metadata,
                          output_prefix):
    # Writing additional output, creates a list of all motifs found in data, one motif per row
    # and MS/MS Scan
    all_motifs_in_scans = get_motifs_in_scans(vd, metadata,
                                              overlap_thresh=overlap_thresh,
                                              p_thresh=p_thresh,
                                              X=X,
                                              motif_metadata=motifdb_metadata)
    # Outputting motif list, one by line
    fieldnames = ['scan', 'precursor.mass',
                  'retention.time',
                  "motif",
                  "probability",
                  "overlap",
                  "motifdb_url",
                  "motifdb_annotation"]
    output_motifs_scans_filename = output_prefix + "_motifs_in_scans.tsv"
    motif_list_df = pd.DataFrame(all_motifs_in_scans)
    motif_list_df = motif_list_df[fieldnames]
    motif_list_df.to_csv(output_motifs_scans_filename, sep="\t", index=False)


# Outputting datastructures for GNPS to write out
def get_motifs_in_scans(lda_dictionary, metadata, overlap_thresh=0.3, p_thresh=0.1, X=5,
                        motif_metadata={}):
    # write a nodes file
    all_motifs = lda_dictionary['beta'].keys()
    all_docs = lda_dictionary['theta'].keys()

    doc_to_motif = {}
    for d in lda_dictionary['theta']:
        for m, p in lda_dictionary['theta'][d].items():
            if p >= p_thresh and lda_dictionary['overlap_scores'][d].get(m, 0.0) >= overlap_thresh:
                if not d in doc_to_motif:
                    doc_to_motif[d] = set()
                doc_to_motif[d].add(m)

    # This list includes a list of motifs found in all scans and metadata associated with each motif
    motifs_in_scans = []

    for doc in all_docs:
        try:
            motifs = list(doc_to_motif[doc])
        except:
            motifs = []

        for m in motifs:
            motif_dict = {}
            motif_dict["scan"] = doc
            try:
                motif_dict["precursor.mass"] = metadata[doc]['precursormass']
            except:
                motif_dict["precursor.mass"] = metadata[doc]['precursor_mass']
            try:
                motif_dict["retention.time"] = metadata[doc]['parentrt']
            except:
                motif_dict["retention.time"] = "0.0"
            motif_dict["motif"] = m
            motif_dict["probability"] = lda_dictionary['theta'][doc][m]
            motif_dict["overlap"] = lda_dictionary['overlap_scores'][doc][m]

            try:
                md = motif_metadata.get(m, None)
                link_url = md.get('motifdb_url', None)
                annotation = md.get('annotation', None)

                motif_dict["motifdb_url"] = link_url.encode('utf8')
                motif_dict["motifdb_annotation"] = annotation.encode('utf8')
            except:
                motif_dict["motifdb_url"] = ""
                motif_dict["motifdb_annotation"] = ""

            motifs_in_scans.append(motif_dict)

    return motifs_in_scans
