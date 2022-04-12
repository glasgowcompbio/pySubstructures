import csv

import networkx as nx
import pandas as pd
from pyMolNetEnhancer import Mass2Motif_2_Network, make_motif_graphml


def write_output_files(lda_dictionary, pairs_file, output_name_prefix, metadata,
                       overlap_thresh=0.3, p_thresh=0.1, X=5, motif_metadata={}):
    # load the pairs file

    components_to_ignore = set()
    components_to_ignore.add('-1')

    rows = []
    with open(pairs_file, 'rU') as f:
        reader = csv.reader(f, dialect='excel', delimiter='\t')
        heads = reader.next()
        for line in reader:
            rows.append(line)

    # find the components (mol families) for each doc
    ci_pos = heads.index('ComponentIndex')  # will it always be called this?
    ci = set([x[ci_pos] for x in rows])

    component_to_doc = {c: set() for c in ci}
    doc_to_component = {}
    for row in rows:
        component = row[ci_pos]
        component_to_doc[component].add(row[0])
        if not row[0] in doc_to_component:
            doc_to_component[row[0]] = set()
        if not row[1] in doc_to_component:
            doc_to_component[row[1]] = set()

        component_to_doc[component].add(row[1])
        doc_to_component[row[0]] = component
        doc_to_component[row[1]] = component

    # map components to motif
    component_to_motif = {c: {} for c in ci}
    doc_to_motif = {}
    motif_to_doc = {}
    for d in lda_dictionary['theta']:
        for m, p in lda_dictionary['theta'][d].items():
            if p >= p_thresh and lda_dictionary['overlap_scores'][d].get(m, 0.0) >= overlap_thresh:
                if not d in doc_to_motif:
                    doc_to_motif[d] = set()
                if not m in motif_to_doc:
                    motif_to_doc[m] = set()
                doc_to_motif[d].add(m)
                motif_to_doc[m].add(d)
                c = doc_to_component[d]
                if not m in component_to_motif[c]:
                    component_to_motif[c][m] = 1
                else:
                    component_to_motif[c][m] += 1

    # find the top X motifs for each component
    topX = {c: [] for c in component_to_motif}
    for c, motifs in component_to_motif.items():
        mco = zip(motifs.keys(), motifs.values())
        if len(mco) > 0:
            mco.sort(key=lambda x: x[1], reverse=True)
            n = max(X, len(mco))
            m, _ = zip(*mco)
            topX[c] = m[:n]

    # write the new pairs file
    new_heads = ['CLUSTERID1', 'interact', 'CLUSTERID2', 'DeltaMZ', 'MEH', 'Cosine', 'OtherScore',
                 'ComponentIndex', 'SharedMotifs', 'topX']
    outfile = output_name_prefix + '_ms2lda_edges.tsv'
    with open(outfile, 'w') as f:
        writer = csv.writer(f, delimiter='\t')
        writer.writerow(new_heads)
        for row in rows:
            new_row = [row[0], 'cosine', row[1], row[2], row[3], row[4], row[5], row[6]]
            # find shared motifs
            m1 = doc_to_motif.get(row[0], set())
            m2 = doc_to_motif.get(row[1], set())
            shared = m1.intersection(m2)
            shared = ",".join(shared)
            new_row += [shared]
            if row[6] == '-1':
                top = ''
            else:
                top = ','.join(topX[row[6]])
            new_row += [top]
            writer.writerow(new_row)

        # the new edges - each pair that share in the same component that share a motif
        for m, docs in motif_to_doc.items():
            if len(docs) > 1:
                ld = list(docs)
                for i, d in enumerate(ld[:-1]):
                    for dp in ld[i + 1:]:
                        # check they're in the same component
                        c = doc_to_component[d]
                        if c in components_to_ignore:  # don't write for singletons
                            continue
                        if dp in component_to_doc[c]:
                            new_row = [d, m, dp, '', '', '', '', c, '', '']
                            writer.writerow(new_row)

    # write a nodes file
    all_motifs = lda_dictionary['beta'].keys()
    all_docs = lda_dictionary['theta'].keys()
    nodes_file = output_name_prefix + '_ms2lda_nodes.tsv'
    no_edge = 0
    with open(nodes_file, 'w') as f:
        writer = csv.writer(f, delimiter='\t')
        heads = ['scans', 'document', 'motif', 'probability', 'overlap', 'precursor.mass',
                 'retention.time', 'document.annotation']
        if len(motif_metadata) > 0:
            motifs_with_links = []
            for m in all_motifs:
                md = motif_metadata.get(m, None)
                if md:
                    motif_string = m
                    link_url = md.get('motifdb_url', None)
                    if link_url:
                        motif_string += " " + link_url
                    annotation = md.get('annotation', None)
                    if annotation:
                        motif_string += " " + annotation
                    motifs_with_links.append(motif_string)
                else:
                    motifs_with_links.append(m)
            heads += motifs_with_links
        else:
            heads += all_motifs
        writer.writerow([s.encode('utf-8') for s in heads])
        for doc in all_docs:
            try:
                motifs = list(doc_to_motif[doc])
            except:
                motifs = []
            new_row = [doc, doc]
            probs = [lda_dictionary['theta'][doc][m] for m in motifs]
            overlaps = [lda_dictionary['overlap_scores'][doc][m] for m in motifs]
            new_row += [",".join(motifs),
                        ",".join(["{:.4f}".format(p) for p in probs]),
                        ",".join(["{:.4f}".format(o) for o in overlaps])]
            precursor_mz = metadata[doc]['precursormass']
            try:
                retention_time = metadata[doc]['parentrt']
            except:
                retention_time = 'NA'
            annotation = metadata[doc].get('annotation', '')
            new_row += [precursor_mz, retention_time, annotation]
            motif_list = [0.0 for m in all_motifs]
            for m in motifs:
                pos = all_motifs.index(m)
                o = lda_dictionary['overlap_scores'][doc][m]
                motif_list[pos] = o
            new_row += motif_list
            writer.writerow(new_row)


def create_graphml(input_motifs_filename, input_network_edges, output_graphml, output_pairs,
                   pvalue=0.05, overlap=0.1, topx=5):
    try:
        # prepare data
        motifs = pd.read_csv(input_motifs_filename, sep='\t')
        motifs = motifs.rename(columns={'scan': 'scans'})
        motifs = motifs.rename(columns={'precursor.mass': 'precursormass'})
        motifs = motifs.rename(columns={'retention.time': 'parentrt'})
        motifs['document'] = motifs['scans']
        # motifs = motifs[['scans','precursormass','parentrt','document','motif','probability','overlap']]
        motifs = motifs[
            ['scans', 'precursormass', 'parentrt', 'document', 'motif', 'probability', 'overlap',
             'motifdb_url', 'motifdb_annotation']]

        edges = pd.read_csv(input_network_edges, sep="\t")
        edges = edges[["CLUSTERID1", "CLUSTERID2", "DeltaMZ", "MEH", "Cosine", "OtherScore",
                       "ComponentIndex"]]

        # Run pyMolNetEnhancer
        motif_network = Mass2Motif_2_Network(edges, motifs, prob=pvalue, overlap=overlap, top=topx)
        motif_network['nodes']['motifdb_url'] = motif_network['nodes']['motifdb_url'].agg(
            lambda x: ','.join(map(str, x)))
        motif_network['nodes']['motifdb_annotation'] = motif_network['nodes'][
            'motifdb_annotation'].agg(lambda x: ','.join(map(str, x)))

        # Creating Output Edges
        edges_df = pd.DataFrame(motif_network['edges'])
        edges_df.to_csv(output_pairs, sep='\t', index=False)

        # Create Graphml File
        MG = make_motif_graphml(motif_network['nodes'], motif_network['edges'])

        # Consistently annotating edges with column headers consistent with IIN and Merge Network Polarity, it will be noted as EdgeType
        for edge in MG.edges(data=True):
            for multiedge in MG[edge[0]][edge[1]]:
                edge_type = "Cosine"
                edge_annotation = ""
                interaction_name = MG[edge[0]][edge[1]][multiedge]['interaction']

                if "m2m" in interaction_name or "motif" in interaction_name:
                    edge_type = "Mass2Motif"
                    edge_annotation = interaction_name
                MG[edge[0]][edge[1]][multiedge]['EdgeType'] = edge_type
                MG[edge[0]][edge[1]][multiedge]['EdgeAnnotation'] = edge_annotation

        nx.write_graphml(MG, output_graphml, infer_numeric_types=True)
    except:
        print("Error in Creating Graphml")
        raise
