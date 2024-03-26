#/bin/bash

cd ..
python scripts/ms2lda_runfull.py --input_mgf_file example_data/specs_ms.mgf --input_pairs_file example_data/pairs.tsv --input_mzmine2_folder example_data/mzmine --input_iterations 100
