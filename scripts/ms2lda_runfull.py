import argparse
import os
import sys


from constants import MOTIFDB_SERVER_URL
from motifdb.main import acquire_motifdb, FeatureMatcher
from ms2lda.feature_maker import MakeBinnedFeatures
from ms2lda.lda_variational import VariationalLDA
from ms2lda.loaders import LoadMGF
from ms2lda.reporting import write_topic_report, write_motifs_in_scans

"""Parsing Args"""

parser = argparse.ArgumentParser(description='Creates MS2LDA')
parser.add_argument('input_format', help='input_format')
parser.add_argument('input_iterations', type=int, help='input_iterations')
parser.add_argument('input_minimum_ms2_intensity', type=float, help='input_minimum_ms2_intensity')
parser.add_argument('input_free_motifs', type=int, help='input_free_motifs')
parser.add_argument('input_bin_width', type=float, help='input_bin_width')
parser.add_argument('input_network_overlap', type=float, help='input_network_overlap')
parser.add_argument('input_network_pvalue', type=float, help='input_network_pvalue')
parser.add_argument('input_network_topx', type=int, help='input_network_topx')

parser.add_argument('gnps_motif_include', help='gnps_motif_include')
parser.add_argument('massbank_motif_include', help='massbank_motif_include')
parser.add_argument('urine_motif_include', help='urine_motif_include')
parser.add_argument('euphorbia_motif_include', help='euphorbia_motif_include')
parser.add_argument('rhamnaceae_motif_include', help='rhamnaceae_motif_include')
parser.add_argument('strep_salin_motif_include', help='strep_salin_motif_include')
parser.add_argument('photorhabdus_motif_include', help='photorhabdus_motif_include')
parser.add_argument('user_motif_sets', help='user_motif_sets')

parser.add_argument('input_mgf_file', help='input_mgf_file')
parser.add_argument('input_pairs_file', help='input_pairs_file')
parser.add_argument('input_mzmine2_folder', help='input_mzmine2_folder')
parser.add_argument('output_prefix', help='output_prefix')

args = parser.parse_args()

"""Grabbing the latest Motifs from MS2LDA"""
import requests

motifset_dict = requests.get(MOTIFDB_SERVER_URL + 'list_motifsets/', verify=False).json()
# db_list = ['gnps_binned_005']  # Can update this later with multiple motif sets
db_list = []

if args.gnps_motif_include == "yes":
    db_list.append(2)
if args.massbank_motif_include == "yes":
    db_list.append(4)
if args.urine_motif_include == "yes":
    db_list.append(1)
if args.euphorbia_motif_include == "yes":
    db_list.append(3)
if args.rhamnaceae_motif_include == "yes":
    db_list.append(5)
if args.strep_salin_motif_include == "yes":
    db_list.append(6)
if args.photorhabdus_motif_include == "yes":
    db_list.append(16)

if not ("None" in args.user_motif_sets):
    try:
        db_list += [int(motif_db_id) for motif_db_id in args.user_motif_sets.split(",")]
    except:
        print(
            "User motif set improperly formatted. Please have numbers separated by commas or enter None")
        exit(1)

db_list = list(set(db_list))

# Acquire motifset from MS2LDA.org
motifdb_spectra, motifdb_metadata, motifdb_features = acquire_motifdb(db_list)

"""Parsing the input files"""

# following should be mscluster or mzmine
input_format = args.input_format
input_iterations = args.input_iterations
input_minimum_ms2_intensity = args.input_minimum_ms2_intensity
input_free_motifs = args.input_free_motifs
input_bin_width = args.input_bin_width

# mgf file name
input_mgf_file = args.input_mgf_file

# pairs file name
pairs_file = args.input_pairs_file

# output prefix
output_prefix = args.output_prefix

if os.path.isdir(output_prefix):
    output_prefix = os.path.join(output_prefix, "output")

print(input_format, input_mgf_file)

ldacodepath = os.path.join(os.path.dirname(
    os.path.realpath(__file__)), 'lda/code')

sys.path.append(ldacodepath)

# from ms2lda_feature_extraction import LoadMGF, MakeBinnedFeatures

# Assuming only one mgf file...
name_field = "scans"

if input_format == 'mscluster':
    csv_id_field = None
    mgf_id_field = None
    input_csv_file = None

    l = LoadMGF(name_field=name_field, min_ms2_intensity=input_minimum_ms2_intensity)
elif input_format == 'mzmine':
    mzmine_path = args.input_mzmine2_folder
    if os.path.isdir(mzmine_path):
        all_files = [f for f in os.listdir(mzmine_path) if
                     os.path.isfile(os.path.join(mzmine_path, f))]
        if len(all_files) != 1:
            print("Requires exactly one quantification file")
            exit(1)
        input_csv_file = os.path.join(mzmine_path, all_files[0])
    else:
        input_csv_file = mzmine_path

    csv_id_col = 'row ID'
    mgf_id_field = 'scans'
    l = LoadMGF(name_field=name_field, \
                peaklist=input_csv_file, \
                csv_id_col=csv_id_col, \
                id_field=mgf_id_field, \
                min_ms2_intensity=input_minimum_ms2_intensity, \
                mz_col_name='row m/z', \
                rt_col_name='row retention time')

ms1, ms2, metadata = l.load_spectra([input_mgf_file])
print("Loaded {} spectra".format(len(ms1)))

m = MakeBinnedFeatures(
    bin_width=input_bin_width)  # What value do you want here?? TODO: Parameterize
corpus, features = m.make_features(ms2)
corpus = corpus[corpus.keys()[0]]

fm = FeatureMatcher(motifdb_features, features)
motifdb_spectra = fm.convert(motifdb_spectra)

# Add the motifdb features to avoid problems when loading the dict into vlda later
bin_width = m.bin_width
added = 0
for f in motifdb_features:
    if not f in features:
        word_mz = float(f.split('_')[1])
        word_mz_min = word_mz - bin_width / 2
        word_mz_max = word_mz + bin_width / 2
        features[f] = (word_mz_min, word_mz_max)
        added += 1

print("Added {} features".format(added))

# from lda import VariationalLDA

# K = 300  # number of *new* topics
K = input_free_motifs  # number of *new* topics

vlda = VariationalLDA(corpus, K=K, normalise=1000.0,
                      fixed_topics=motifdb_spectra,
                      fixed_topics_metadata=motifdb_metadata)

# note that for real runs the number of iterations is recommended to be 1000 or higher
vlda.run_vb(initialise=True, n_its=input_iterations)

vd = vlda.make_dictionary(
    features=features, metadata=metadata, filename=output_prefix + '.dict')

from ms2lda.molnet_integration import write_output_files

write_output_files(vd, pairs_file, output_prefix, metadata,
                   overlap_thresh=args.input_network_overlap, p_thresh=args.input_network_pvalue,
                   X=args.input_network_topx, motif_metadata=motifdb_metadata)

# Writing the report - ntoe that you might need to set the 'backend' argument
# for this method to work (see the method in lda.py) as it depends what on
# your system will render the pdf...
# from lda import write_topic_report
try:
    write_topic_report(vd, output_prefix + '_topic_report.pdf')
except:
    print("PDF Generation Failed")

overlap_thresh = args.input_network_overlap
p_thresh = args.input_network_pvalue
X = args.input_network_topx
write_motifs_in_scans(vd, metadata, overlap_thresh, p_thresh, X, motifdb_metadata)

# Reformatting the list of cluster summary
# TODO


sys.exit(0)
