import sys


def write_summary_file(lda_dictionary, filename):
    import csv
    with open(filename, 'w') as f:
        writer = csv.writer(f)
        heads = ['clusterID', 'motif', 'probability', 'overlap']
        writer.writerow(heads)
        for doc, motifs in lda_dictionary['theta'].items():
            for motif, probability in motifs.items():
                row = [doc, motif, probability, lda_dictionary['overlap_scores'][doc].get(motif, 0.0)]
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
        for topic in topic_list:
            sys.stdout.write(topic + ' ')
            sys.stdout.flush()

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
            topic_probs = zip(lda_dictionary['beta'][topic].keys(), lda_dictionary['beta'][topic].values())
            topic_probs.sort(key=lambda x: x[1], reverse=True)
            col = 0
            n_rows = 20
            textPage.text(0.3, 0.9, "{}".format(topic), size=24)
            for i, (word, prob) in enumerate(topic_probs):
                if prob < 0.005:
                    break
                rowpos = i % n_rows
                textPage.text(0.15 + (col) * 0.2, 0.8 - rowpos / 40.0, "{:30s}: {:.2f}".format(word, prob),
                              transform=textPage.transFigure, size=12)
                if rowpos == n_rows - 1:
                    col += 1

            pdf.savefig()
            plt.close()
