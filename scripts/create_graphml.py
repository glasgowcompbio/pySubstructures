import argparse
import os

from ms2lda.molnet_integration import create_graphml


def main():
    parser = argparse.ArgumentParser(description='Creates MS2LDA')
    parser.add_argument('ms2lda_results', help='ms2lda_results')
    parser.add_argument('input_network_edges', help='input_network_edges')
    parser.add_argument('output_graphml', help='output_graphml')
    parser.add_argument('output_pairs', help='output_pairs')
    parser.add_argument('input_network_overlap', type=float, help='input_network_overlap')
    parser.add_argument('input_network_pvalue', type=float, help='input_network_pvalue')
    parser.add_argument('input_network_topx', type=int, help='input_network_topx')

    args = parser.parse_args()

    motif_filename = os.path.join(args.ms2lda_results, "output_motifs_in_scans.tsv")
    create_graphml(motif_filename, args.input_network_edges, args.output_graphml,
                   args.output_pairs, pvalue=args.input_network_pvalue,
                   overlap=args.input_network_overlap, topx=args.input_network_topx)


if __name__ == "__main__":
    main()
