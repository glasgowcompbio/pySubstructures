import requests

from ms2lda.constants import SERVER_URL


def get_motifset_list():
    output = requests.get(SERVER_URL + '/motifdb/list_motifsets')
    motifset_list = output.json()
    return motifset_list


def _get_motifdb_token():
    url = SERVER_URL + '/motifdb/initialise_api'
    client = requests.session()
    token = client.get(url).json()['token']
    return client, token


def post_motifsets(motifsets, filter_threshold=0.95):
    client, token = _get_motifdb_token()

    url = SERVER_URL + '/motifdb/get_motifset/'
    data = {'csrfmiddlewaretoken': token}

    motifset_list = get_motifset_list()
    id_list = [motifset_list[motifset] for motifset in motifsets]
    data['motifset_id_list'] = id_list
    data['filter'] = "True"
    data['filter_threshold'] = filter_threshold  # Default value - not required
    output = client.post(url, data=data).json()
    return output


def get_motifset_metadata(motif_id):
    url = SERVER_URL + '/motifdb/get_motifset_metadata/{}/'.format(motif_id)
    output = requests.get(url)
    motif_metadata = output.json()
    return motif_metadata


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
                    merge_data.append((spec, self.input_spectra[spec], self.input_metadata[spec], sim))
                    pos = spec_names.index(spec)
                    del spec_names[pos]
                # self.input_metadata[current_spec]['merged'] = merge_data
                self.input_metadata[current_spec]['merged'] = ",".join(spec_list)

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
            i1 += intensity ** 2
            for mz2, intensity2 in self.input_spectra[k2].items():
                if mz == mz2:
                    prod += intensity * intensity2
        i2 = sum([i ** 2 for i in self.input_spectra[k2].values()])
        return prod / (np.sqrt(i1) * np.sqrt(i2))
