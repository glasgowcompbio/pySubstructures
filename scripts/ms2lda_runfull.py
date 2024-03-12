import argparse
import os
import sys
from loguru import logger

# Modify sys.path to include the parent directory
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)

from pySubstructures.motifdb.main import acquire_motifdb, FeatureMatcher
from pySubstructures.ms2lda.feature_maker import MakeBinnedFeatures
from pySubstructures.ms2lda.lda_variational import VariationalLDA
from pySubstructures.ms2lda.loaders import LoadMGF
from pySubstructures.ms2lda.reporting import write_topic_report, write_motifs_in_scans
from pySubstructures.ms2lda.molnet_integration import write_output_files

# Restore sys.path to its original state if needed
sys.path.remove(parent_dir)


def parse_args():
    """
    Parses command-line arguments for the MS2LDA analysis workflow.

    Returns:
        argparse.Namespace: An object containing the parsed command-line options.
            - input_mgf_file (str): Path to the spectrum files (MGF, mzML, or MSP).
            - input_pairs_file (str): Path to the molecular network pairs file.
            - input_mzmine2_folder (str, optional): Path to the MZMine2 Quantification CSV.
            - input_format (str): Specifies the input format ('mscluster' or 'mzmine').
            - input_iterations (int): Number of LDA iterations.
            - input_minimum_ms2_intensity (float): Minimum MS2 intensity to consider.
            - input_free_motifs (int): Number of free motifs to be used in LDA.
            - input_bin_width (float): Bin width for feature extraction.
            - input_network_overlap (float): Overlap score threshold for linking Mass2Motifs and spectra.
            - input_network_pvalue (float): Probability value threshold for linking.
            - input_network_topx (int): The top X Mass2Motifs with the highest overlap scores.
            - Various flags for including specific motif sets in the analysis (e.g., gnps_motif_include,
              massbank_motif_include, etc.).
            - user_motif_sets (str, optional): Comma-separated list of user motif set IDs.
            - output_prefix (str): Output prefix for the generated files.

    Raises:
        SystemExit: If --input_mzmine2_folder is required but not provided when --input_format is 'mzmine'.
    """

    parser = argparse.ArgumentParser(
        description="This is the workflow to run ms2lda on GNPS using nextflow. "
        "The input of ms2lda is an MGF file and the molecular network pairs file. "
        "The output is the graphml representing the clusters of motifs and the pairs score similarity. "
        "For documentation, please visit https://ms2lda.org/user_guide/."
    )

    parser.add_argument(
        "--input_mgf_file",
        required=True,
        help="Path to the spectrum files (MGF, mzML, or MSP). This is essential for MS2LDA analysis.",
    )
    parser.add_argument(
        "--input_pairs_file",
        required=True,
        help="Path to the molecular network pairs file. Necessary for establishing connections between motifs.",
    )
    parser.add_argument(
        "--input_mzmine2_folder",
        help='Path to the MZMine2 Quantification CSV. Required if input_format is "mzmine".',
    )
    parser.add_argument(
        "--input_format",
        default="mzmine",
        help='Specifies the input format. Use "mscluster" for MSCluster or MZMine2 without quantification, '
        'and "mzmine" for MZMine2 with quantification.',
    )
    parser.add_argument(
        "--input_iterations", type=int, default=1000, help="Number of LDA iterations."
    )
    parser.add_argument(
        "--input_minimum_ms2_intensity",
        type=float,
        default=100,
        help="Minimum MS2 intensity to consider in the analysis.",
    )
    parser.add_argument(
        "--input_free_motifs",
        type=int,
        default=300,
        help="Number of free motifs to be used in LDA.",
    )
    parser.add_argument(
        "--input_bin_width",
        type=float,
        default=0.005,
        help="Bin width for feature extraction. Options include 0.005 for Q-Exactive Data, 0.01 for TOF Data, etc.",
    )
    parser.add_argument(
        "--input_network_overlap",
        type=float,
        default=0.3,
        help="Overlap score threshold for linking Mass2Motifs and spectra. A number between 0 and 1; default is 0.3.",
    )
    parser.add_argument(
        "--input_network_pvalue",
        type=float,
        default=0.1,
        help="Probability value threshold for linking Mass2Motifs and spectra. A number between 0 and 1; "
        "default is 0.1.",
    )
    parser.add_argument(
        "--input_network_topx",
        type=int,
        default=5,
        help="The top X Mass2Motifs with the highest overlap scores listed in the new edge file. Default is 5.",
    )
    parser.add_argument(
        "--gnps_motif_include",
        default="yes",
        help='Include GNPS motifs in the analysis. Options are "yes" or "no".',
    )
    parser.add_argument(
        "--massbank_motif_include",
        default="yes",
        help='Include MassBank motifs in the analysis. Options are "yes" or "no".',
    )
    parser.add_argument(
        "--urine_motif_include",
        default="yes",
        help='Include urine motifs in the analysis. Options are "yes" or "no".',
    )
    parser.add_argument(
        "--euphorbia_motif_include",
        default="no",
        help='Include Euphorbia motifs in the analysis. Options are "yes" or "no".',
    )
    parser.add_argument(
        "--rhamnaceae_motif_include",
        default="no",
        help='Include Rhamnaceae plant motifs in the analysis. Options are "yes" or "no".',
    )
    parser.add_argument(
        "--strep_salin_motif_include",
        default="no",
        help='Include Streptomyces and Salinisporus motifs in the analysis. Options are "yes" or "no".',
    )
    parser.add_argument(
        "--photorhabdus_motif_include",
        default="no",
        help='Include Photorhabdus and Xenorhabdus motifs in the analysis. Options are "yes" or "no".',
    )
    parser.add_argument(
        "--user_motif_sets",
        default="None",
        help='Comma-separated list of user motif set IDs. Use "None" if no custom motifs are to be included. '
        "The id for the motif set can be found on ms2lda.org.",
    )
    parser.add_argument(
        "--output_prefix",
        default="output",
        help="Output prefix for the files generated by this script.",
    )

    args = parser.parse_args()

    # Conditional requirement check for input_mzmine2_folder based on input_format
    if args.input_format == "mzmine" and not args.input_mzmine2_folder:
        logger.warning(
            "Error: --input_mzmine2_folder is required when --input_format is 'mzmine'."
        )
        sys.exit(1)

    return args


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
    motif_selections = {
        "gnps_motif_include": 2,
        "massbank_motif_include": 4,
        "urine_motif_include": 1,
        "euphorbia_motif_include": 3,
        "rhamnaceae_motif_include": 5,
        "strep_salin_motif_include": 6,
        "photorhabdus_motif_include": 16,
    }

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
            sys.exit(1)

    # Remove duplicates and acquire motifs from MS2LDA.org
    db_list = list(set(db_list))
    motifdb_spectra, motifdb_metadata, motifdb_features = acquire_motifdb(db_list)

    return motifdb_spectra, motifdb_metadata, motifdb_features


def get_single_quant_file_path(mzmine_path):
    """
    Ensures there is exactly one quantification file in the given directory and returns its path.

    Args:
        mzmine_path (str): The directory path where the quantification file is located.

    Returns:
        str: The path to the single quantification file.

    Raises:
        ValueError: If there are no files or more than one file in the directory.
    """
    if not os.path.isdir(mzmine_path):
        # Assuming the path is a direct path to a file.
        return mzmine_path

    quant_files = [
        f
        for f in os.listdir(mzmine_path)
        if os.path.isfile(os.path.join(mzmine_path, f))
    ]
    if len(quant_files) != 1:
        raise ValueError("MZMine2 folder must contain exactly one quantification file.")

    return os.path.join(mzmine_path, quant_files[0])


def parse_input(args):
    """
    Loads spectral data from MGF files based on user-specified input format and other parameters.

    This function is responsible for loading MS1 and MS2 spectra from the provided MGF file. It supports
    two input formats: 'mscluster' and 'mzmine'. For 'mzmine', it also requires specifying a folder
    containing a single quantification CSV file. The function adjusts the loading process based on the input
    format, ensuring the appropriate handling of spectral data for further analysis.

    Args:
        args (Namespace): An argparse Namespace object containing all command-line arguments.
        Relevant arguments include:
            - input_format (str): The format of the input data. Expected values are 'mscluster' or 'mzmine'.
            - input_mgf_file (str): The path to the MGF file containing the spectral data to be analyzed.
            - input_mzmine2_folder (str, optional): For 'mzmine' format, the path to the folder containing the
              quantification CSV file.
            - input_minimum_ms2_intensity (float): The minimum MS2 intensity threshold for spectra to be considered
              in the analysis.

    Returns:
        tuple: A tuple containing three elements:
            - ms1 (list): A list of MS1 spectra (may be empty depending on the input format and data).
            - ms2 (list): A list of MS2 spectra.
            - metadata (dict): A dictionary containing metadata associated with the spectra.

    Raises:
        ValueError: If the input format is 'mzmine' and the specified MZMine2 folder does not contain exactly
                    one quantification file, or if an unsupported input format is provided.
    """
    input_format = args.input_format
    input_mgf_file = args.input_mgf_file
    logger.info(f"Input format: {input_format}, MGF file: {input_mgf_file}")

    # Common parameters for LoadMGF, applicable to both input formats
    loader_params = {
        "name_field": "scans",
        "min_ms2_intensity": args.input_minimum_ms2_intensity,
    }

    if input_format == "mscluster":
        loader = LoadMGF(**loader_params)
    elif input_format == "mzmine":
        try:
            quant_file_path = get_single_quant_file_path(args.input_mzmine2_folder)
        except ValueError as e:
            logger.warning(e)
            sys.exit(1)

        # Extend loader_params with mzmine specific parameters
        loader_params.update(
            {
                "peaklist": quant_file_path,
                "csv_id_col": "row ID",
                "id_field": "scans",
                "mz_col_name": "row m/z",
                "rt_col_name": "row retention time",
            }
        )
        loader = LoadMGF(**loader_params)
    else:
        raise ValueError(f"Unsupported input format: {input_format}")

    ms1, ms2, metadata = loader.load_spectra([input_mgf_file])
    logger.info(f"Loaded {len(ms1)} MS1 and {len(ms2)} MS2 spectra.")

    return ms1, ms2, metadata


def feature_binning(bin_width, motifdb_features, motifdb_spectra, ms2):
    """
    Bins the MS2 spectral features and aligns them with the motif database features.

    This method creates binned features from the MS2 spectra, matches these features against the motif database
    features, and updates the motif database spectra to ensure compatibility. This step is crucial for the
    subsequent LDA analysis, as it ensures that the spectra and the motif database features are aligned.

    Args:
        bin_width (float): The width of the bin for feature extraction.
        motifdb_features (list): A list of features from the motif database.
        motifdb_spectra (dict): A dictionary of spectra from the motif database.
        ms2 (list): A list of MS2 spectra to be processed.

    Returns:
        tuple: A tuple containing the processed corpus, features, and updated motifdb_spectra.
    """

    # Initialize the feature maker with the specified bin width
    m = MakeBinnedFeatures(bin_width=bin_width)
    corpus, features = m.make_features(ms2)

    # Select a representative corpus if multiple are present
    corpus = corpus[list(corpus.keys())[0]]

    # Feature matching between MS2 and motif database features
    fm = FeatureMatcher(motifdb_features, features)
    motifdb_spectra = fm.convert(motifdb_spectra)

    # Add the motifdb features to avoid problems when loading the dict into vlda later
    added_features = 0
    for feature in motifdb_features:
        if feature not in features:
            word_mz = float(feature.split("_")[1])
            features[feature] = (word_mz - bin_width / 2, word_mz + bin_width / 2)
            added_features += 1
    logger.info("Added {} features".format(added_features))

    return corpus, features, motifdb_spectra


def run_lda(
    input_iterations,
    K,
    corpus,
    features,
    metadata,
    motifdb_metadata,
    motifdb_spectra,
    output_prefix,
):
    """
    Executes the LDA analysis using the provided corpus and motif database, generating a topic model.

    This function initializes the Variational LDA with the corpus and fixed topics derived from the motif database.
    It then runs the LDA analysis for a specified number of iterations and generates a dictionary that maps each
    spectrum to its most likely topic. The output is saved with a specified prefix.

    Args:
        input_iterations (int): Number of iterations to run the LDA analysis.
        K (int): The number of new topics to discover in the LDA model.
        corpus (dict): The corpus of binned features to be used in the LDA analysis.
        features (dict): The dictionary of feature mappings used in the LDA analysis.
        metadata (dict): Metadata associated with the corpus.
        motifdb_metadata (dict): Metadata from the motif database.
        motifdb_spectra (dict): Spectra from the motif database, used as fixed topics in the LDA model.
        output_prefix (str): The prefix for naming output files generated by this analysis.

    Returns:
        dict: A dictionary representing the LDA model's topics, where each key is a spectrum identifier
              and its value is the most likely topic assignment.
    """
    # Initialize the VariationalLDA model with corpus, fixed topics, and additional settings
    vlda = VariationalLDA(
        corpus,
        K=K,
        normalise=1000.0,
        fixed_topics=motifdb_spectra,
        fixed_topics_metadata=motifdb_metadata,
    )

    # Run the LDA model
    # Note that for real runs the number of iterations is recommended to be 1000 or higher
    vlda.run_vb(initialise=True, n_its=input_iterations)

    # Generate a dictionary of LDA output
    vd = vlda.make_dictionary(
        features=features, metadata=metadata, filename=f"{output_prefix}.dict"
    )

    logger.info(
        f"LDA analysis completed with {input_iterations} iterations and {K} topics."
    )
    return vd


def main():
    # Parse the command line arguments
    args = parse_args()

    # Acquiring the latest Motifs from MS2LDA
    logger.info("Getting motifs from MS2LDA")
    motifdb_spectra, motifdb_metadata, motifdb_features = acquire_motifsets(args)

    # Parsing the input files
    logger.info("Parsing input files")
    ms1, ms2, metadata = parse_input(args)

    # Perform feature binning and match with motif database features
    logger.info("Running feature binning and matching")
    corpus, features, motifdb_spectra = feature_binning(
        args.input_bin_width, motifdb_features, motifdb_spectra, ms2
    )

    # Preparing output file paths
    output_prefix = args.output_prefix
    if os.path.isdir(output_prefix):
        output_prefix = os.path.join(output_prefix, "output")

    # Running LDA analysis
    logger.info(f"Running LDA analysis, output={output_prefix}")
    vd = run_lda(
        args.input_iterations,
        args.input_free_motifs,
        corpus,
        features,
        metadata,
        motifdb_metadata,
        motifdb_spectra,
        output_prefix,
    )

    # Generating output files and reports
    logger.info("Generating output files and reports")
    write_output_files(
        vd,
        args.input_pairs_file,
        output_prefix,
        metadata,
        overlap_thresh=args.input_network_overlap,
        p_thresh=args.input_network_pvalue,
        X=args.input_network_topx,
        motif_metadata=motifdb_metadata,
    )

    # Writing the report - note that you might need to set the 'backend' argument
    # for this method to work (see the method in lda.py) as it depends on what
    # your system will render the pdf...
    try:
        report_path = output_prefix + "_topic_report.pdf"
        logger.info(f"Writing PDF report to {report_path}")
        write_topic_report(vd, report_path)
    except Exception as e:
        logger.warning(f"PDF Generation Failed: {e}")

    # TODO: Reformatting the list of cluster summary
    logger.info("Writing motifs in scans")
    write_motifs_in_scans(
        vd,
        metadata,
        args.input_network_overlap,
        args.input_network_pvalue,
        args.input_network_topx,
        motifdb_metadata,
        output_prefix,
    )


if __name__ == "__main__":
    main()
