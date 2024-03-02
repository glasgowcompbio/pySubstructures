import argparse
import os
import requests
import sys

from loguru import logger
import pandas as pd

import certifi

# Modify sys.path to include the parent directory
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)

# Assuming these imports are necessary for your context
from pySubstructures.motifdb.main import acquire_motifdb
from pySubstructures.motifdb.constants import MOTIFDB_SERVER_URL

# Restore sys.path to its original state if needed
sys.path.remove(parent_dir)

def save_motifs(motifdb_spectra, motifdb_metadata, output_dir):
    for motif_name, spectra in motifdb_spectra.items():
        file_path = os.path.join(output_dir, motif_name)
        with open(file_path, 'w') as f:
            f.write("#NAME {}\n".format(motifdb_metadata[motif_name]['name']))
            f.write("#ANNOTATION {}\n".format(motifdb_metadata[motif_name].get('annotation', 'No annotation available')))
            f.write("#SHORT_ANNOTATION {}\n".format(motifdb_metadata[motif_name].get('short_annotation', 'No short annotation available')))
            f.write("#COMMENT {}\n".format(motifdb_metadata[motif_name].get('comment', 'No comment available')))

            sorted_spectra = sorted(spectra.items(), key=lambda x: x[1], reverse=True)

            for fragment, intensity in sorted_spectra:
                f.write("{},{}\n".format(fragment, intensity))

def download_motifset(base_dir):
    motifset_dict = requests.get(MOTIFDB_SERVER_URL + 'list_motifsets/', verify=certifi.where()).json()
    motifset_rev_dict = {value: key for key, value in motifset_dict.items()}

    # Define whitelist here
    # whitelist_ids = None
    whitelist_ids = [1, 2, 3, 4, 5, 6, 16, 17, 33, 37, 32, 31, 30, 29, 38]

    db_list = [motif_id for motif_id in motifset_dict.values() if not whitelist_ids or motif_id in whitelist_ids]

    motif_data = []

    for motif_id in db_list:
        motifdb_spectra, motifdb_metadata, _ = acquire_motifdb([motif_id], filter_threshold=0.95)
        motifset_name = motifset_rev_dict[motif_id]
        logger.info(f'Downloading {motifset_name} motifset ({len(motifdb_spectra)} motifs)')

        output_dir = os.path.join(base_dir, motifset_name)
        os.makedirs(output_dir, exist_ok=True)
        save_motifs(motifdb_spectra, motifdb_metadata, output_dir)

        motif_data.append((motif_id, motifset_name, len(motifdb_spectra)))

    # Create DataFrame and set Motif ID as the index
    df = pd.DataFrame(motif_data, columns=['Motif ID', 'Motif Set', 'Number of Motifs']).set_index('Motif ID')
    df = df.sort_values(by='Number of Motifs', ascending=False)
    return df

def main():
    parser = argparse.ArgumentParser(description='Download all motif sets from MS2LDA and save them to files.')
    parser.add_argument('--base_dir', type=str, default='data/MOTIFDB',
                        help='Base directory to save the motif sets (default: data/MOTIFDB).')

    args = parser.parse_args()

    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', args.base_dir))
    os.makedirs(base_dir, exist_ok=True)

    df = download_motifset(base_dir)
    print(df)

    logger.info(f"Downloaded and saved all motif sets to '{base_dir}'. See the motif set report above.")

if __name__ == '__main__':
    main()
