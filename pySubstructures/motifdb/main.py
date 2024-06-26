import glob
import os

from loguru import logger
import numpy as np
import requests
import urllib3
import certifi

from pySubstructures.motifdb.constants import METADATA_FIELDS, MOTIFDB_SERVER_URL
from pySubstructures.motifdb.constants import (
    MOTIF_NAMES_TO_MS2LDA_DB,
    GNPS_LIBRARY_DERIVED_MASS2MOTIFS,
    MASSBANK_LIBRARY_DERIVED_MASS2MOTIFS,
    URINE_DERIVED_MASS2MOTIFS,
    EUPHORBIA_PLANT_MASS2MOTIFS,
    RHAMNACEAE_PLANT_MASS2MOTIFS,
    STREPTOMYCES_AND_SALINISPORA_MASS2MOTIFS,
    PHOTORHABDUS_AND_XENORHABDUS_MASS2MOTIFS,
)

http = urllib3.PoolManager(cert_reqs="CERT_REQUIRED", ca_certs=certifi.where())


def acquire_motifsets(args):
    """
    Acquires motif sets based on the user's selection from MS2LDA.org.

    Args:
        args (Namespace): Parsed command line arguments containing user preferences
                          for including specific motif sets in the analysis.

    Returns:
        tuple: Contains three elements:
               - motifdb_spectra: The spectra from the motif database.
               - motifdb_metadata: Metadata associated with the motifs.
               - motifdb_features: Features of the motifs.
    """
    # Initialize list to hold database IDs based on user selection
    db_list = []

    # Map user selections to motif database IDs
    motif_selections = generate_motif_selections()

    # Append selected database IDs to the list
    for motif, db_id in motif_selections.items():
        if getattr(args, motif) == "yes":
            db_list.append(db_id)

    # Handle user-defined motif sets, if any
    if args.user_motif_sets not in [None, "None"]:
        try:
            user_motif_ids = [int(id) for id in args.user_motif_sets.split(",")]
            db_list.extend(user_motif_ids)
        except ValueError:
            logger.warning(
                "User motif set improperly formatted. Please ensure numbers are separated by commas or enter 'None'."
            )

    # Remove duplicates and acquire motifs from MS2LDA.org
    db_list = list(set(db_list))
    motifdb_spectra, motifdb_metadata, motifdb_features = acquire_motifdb(db_list)

    return motifdb_spectra, motifdb_metadata, motifdb_features


def generate_motif_selections():
    """
    Generates motif selections based on constants
    """
    return {
        "gnps_motif_include": MOTIF_NAMES_TO_MS2LDA_DB[
            GNPS_LIBRARY_DERIVED_MASS2MOTIFS
        ],
        "massbank_motif_include": MOTIF_NAMES_TO_MS2LDA_DB[
            MASSBANK_LIBRARY_DERIVED_MASS2MOTIFS
        ],
        "urine_motif_include": MOTIF_NAMES_TO_MS2LDA_DB[URINE_DERIVED_MASS2MOTIFS],
        "euphorbia_motif_include": MOTIF_NAMES_TO_MS2LDA_DB[
            EUPHORBIA_PLANT_MASS2MOTIFS
        ],
        "rhamnaceae_motif_include": MOTIF_NAMES_TO_MS2LDA_DB[
            RHAMNACEAE_PLANT_MASS2MOTIFS
        ],
        "strep_salin_motif_include": MOTIF_NAMES_TO_MS2LDA_DB[
            STREPTOMYCES_AND_SALINISPORA_MASS2MOTIFS
        ],
        "photorhabdus_motif_include": MOTIF_NAMES_TO_MS2LDA_DB[
            PHOTORHABDUS_AND_XENORHABDUS_MASS2MOTIFS
        ],
    }


def acquire_motifdb(db_list, filter_threshold=0.95):
    data = {}
    data["motifset_id_list"] = db_list
    data["filter"] = "True"
    data["filter_threshold"] = filter_threshold

    output = requests.post(
        MOTIFDB_SERVER_URL + "get_motifset/", data=data, verify=certifi.where()
    ).json()
    motifdb_spectra = output["motifs"]
    motifdb_metadata = output["metadata"]
    motifdb_features = set()
    for m, spec in motifdb_spectra.items():
        for f in spec:
            motifdb_features.add(f)

    return motifdb_spectra, motifdb_metadata, motifdb_features


def post_motifsets(motifsets, filter_threshold=0.95):
    client, token = _get_motifdb_token()

    url = MOTIFDB_SERVER_URL + "get_motifset/"
    data = {"csrfmiddlewaretoken": token}

    motifset_list = get_motifset_list()
    id_list = [motifset_list[motifset] for motifset in motifsets]
    data["motifset_id_list"] = id_list
    data["filter"] = "True"
    data["filter_threshold"] = filter_threshold  # Default value - not required
    output = client.post(url, data=data).json()
    return output


def get_motifset_list():
    url = MOTIFDB_SERVER_URL + "list_motifsets"
    output = http.request("GET", url)
    motifset_list = output.json()
    return motifset_list


def _get_motifdb_token():
    url = MOTIFDB_SERVER_URL + "initialise_api"
    client = requests.session()
    token = client.get(url).json()["token"]
    return client, token


def get_motifset_metadata(motif_id):
    url = MOTIFDB_SERVER_URL + "get_motifset_metadata/{}/".format(motif_id)
    output = http.request("GET", url)
    motif_metadata = output.json()
    return motif_metadata


def load_db(db_list, db_path):
    # loads the dbs listed in the list
    # items in the list should be folder names in the dirctory indicated by db_path
    filenames = []
    for dir_name in db_list:
        glob_path = os.path.join(db_path, dir_name, "*.m2m")
        print("Looking in {}".format(glob_path))
        new_filenames = glob.glob(glob_path)
        filenames += new_filenames
        print("\t Found {}".format(len(new_filenames)))

    print("Found total of {} motif files".format(len(filenames)))

    metadata = {}
    spectra = {}

    for filename in filenames:
        motif_name = os.path.split(filename)[-1]
        spectra[motif_name], metadata[motif_name] = parse_m2m(filename)

    features = set()
    for m, s in spectra.items():
        for f in s:
            features.add(f)
    return spectra, metadata, features


def parse_m2m(filename):
    metadata = {}
    spectrum = {}
    with open(filename, "r") as f:
        for line in f:
            if line.startswith("#"):
                # it's some metadata
                tokens = line.split()
                key = tokens[0][1:].lower()
                if key in METADATA_FIELDS:
                    new_value = " ".join(tokens[1:])
                    if not key in metadata:
                        metadata[key] = new_value
                    else:
                        # is it a list already?
                        current_value = metadata[key]
                        if isinstance(current_value, list):
                            metadata[key].append(new_value)
                        else:
                            metadata[key] = [current_value].append(new_value)
                else:
                    print("Found unknown key ({}) in {}".format(key, filename))
            elif len(line) > 0:
                mz, intensity = line.split(",")
                spectrum[mz] = float(intensity)
    return spectrum, metadata


class MotifFilter(object):
    def __init__(self, spectra, metadata, threshold=0.95):
        self.input_spectra = spectra
        self.input_metadata = metadata
        self.threshold = threshold

    def filter(self):
        # Greedy filtering
        # Loops through the spectra and for each one computes its similarity with
        # the remaining. Any that exceed the threshold are merged
        # Merging invovles the latter one and puts it into the metadata of the
        # original so we can always check back.
        spec_names = sorted(self.input_metadata.keys())
        final_spec_list = []
        while len(spec_names) > 0:
            current_spec = spec_names[0]
            final_spec_list.append(current_spec)
            del spec_names[0]
            merge_list = []
            for spec in spec_names:
                sim = self.compute_similarity(current_spec, spec)
                if sim >= self.threshold:
                    merge_list.append((spec, sim))
            if len(merge_list) > 0:
                merge_data = []
                spec_list = []
                for spec, sim in merge_list:
                    spec_list.append(spec)
                    print("Merging: {} and {} ({})".format(current_spec, spec, sim))
                    # chuck the merged motif into metadata so that we can find it later
                    merge_data.append(
                        (spec, self.input_spectra[spec], self.input_metadata[spec], sim)
                    )
                    pos = spec_names.index(spec)
                    del spec_names[pos]
                # self.input_metadata[current_spec]['merged'] = merge_data
                self.input_metadata[current_spec]["merged"] = ",".join(spec_list)

        output_spectra = {}
        output_metadata = {}
        for spec in final_spec_list:
            output_spectra[spec] = self.input_spectra[spec]
            output_metadata[spec] = self.input_metadata[spec]
        print("After merging, {} motifs remain".format(len(output_spectra)))
        return output_spectra, output_metadata

    def compute_similarity(self, k, k2):
        # compute the cosine similarity of the two spectra
        prod = 0
        i1 = 0
        for mz, intensity in self.input_spectra[k].items():
            i1 += intensity**2
            for mz2, intensity2 in self.input_spectra[k2].items():
                if mz == mz2:
                    prod += intensity * intensity2
        i2 = sum([i**2 for i in self.input_spectra[k2].values()])
        return prod / (np.sqrt(i1) * np.sqrt(i2))


class FeatureMatcher(object):
    def __init__(self, db_features, other_features, bin_width=0.005):
        self.db_features = db_features
        self.other_features = other_features
        self.fmap = {}
        self.bin_width = bin_width
        self.augmented_features = {f: v for f, v in other_features.items()}

        self.match()
        self.match(ftype="loss")

    def match(self, ftype="fragment"):
        import bisect

        other_names = [f for f in self.other_features if f.startswith(ftype)]
        other_min_mz = [
            self.other_features[f][0]
            for f in self.other_features
            if f.startswith(ftype)
        ]
        other_max_mz = [
            self.other_features[f][1]
            for f in self.other_features
            if f.startswith(ftype)
        ]

        temp = list(zip(other_names, other_min_mz, other_max_mz))
        temp.sort(key=lambda x: x[1])
        other_names, other_min_mz, other_max_mz = zip(*temp)
        other_names = list(other_names)
        other_min_mz = list(other_min_mz)
        other_max_mz = list(other_max_mz)

        exact_match = 0
        new_ones = 0
        overlap_match = 0
        for f in [f for f in self.db_features if f.startswith(ftype)]:
            if f in other_names:
                self.fmap[f] = f
                exact_match += 1
            else:
                fmz = float(f.split("_")[1])
                if fmz < other_min_mz[0] or fmz > other_max_mz[-1]:
                    self.fmap[f] = f
                    self.augmented_features[f] = (
                        fmz - self.bin_width / 2,
                        fmz + self.bin_width / 2,
                    )
                    new_ones += 1
                    continue
                fpos = bisect.bisect_right(other_min_mz, fmz)
                fpos -= 1
                if fmz <= other_max_mz[fpos]:
                    self.fmap[f] = other_names[fpos]
                    overlap_match += 1
                else:
                    self.fmap[f] = f
                    self.augmented_features[f] = (
                        fmz - self.bin_width / 2,
                        fmz + self.bin_width / 2,
                    )
                    new_ones += 1
        print(
            "Finished matching ({}). {} exact matches, {} overlap matches, {} new features".format(
                ftype, exact_match, overlap_match, new_ones
            )
        )

    def convert(self, dbspectra):
        for doc, spec in dbspectra.items():
            newspec = {}
            for f, i in spec.items():
                newspec[self.fmap[f]] = i
            dbspectra[doc] = newspec
        return dbspectra
