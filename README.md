# pySubstructures

A Python package to perform unsupervised discoveries of motifs from tandem mass spectrometry data.

## Python TODO:
- [X] Add codes from the lda repo.
- [X] Tidy up codes, and keep only the ones we actually use.
- [X] MolnetEnhancer support
- [X] Gensim support
- [ ] Online/offline MotifDB support when running LDA inference
- [ ] Create a Python package

## Maybe:
- [ ] [MESSAR](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0226770) support
- [ ] [Labelled LDA](https://github.com/gkreder/ms2-topic-model) support
- [ ] [Matchms](https://github.com/matchms/matchms) compatibility
- [ ] [BERTopic](https://maartengr.github.io/BERTopic/index.html) support
- [ ] [PALS](https://pals.glasgowcompbio.org/) support
- [ ] MAGMa-MS2LDA support
- [ ] Classyfire functions

## Visualisation:
- [ ] Stand-alone viewer using Dash

## MS2LDA.org TODO:
- [ ] Replace the lda codes used on the server with this package

## Environment Setup

This project uses a Conda environment to manage dependencies. To set up the environment, ensure you have Conda installed, then run the following command from the root directory of this project:

```bash
conda env create -f environment.yml
```
This will create a new Conda environment named pySubstructures and install all required dependencies, including Black for code formatting.

### Activating the Environment
After installing the environment, activate it using:
```bash
conda activate pySubstructures
```

## For Contributors
We use Black, the Python code formatter, to ensure code consistency. It's included in the project's Conda environment. Before submitting any code, activate the pySubstructures environment and run Black:

```bash
black path/to/your/python/file_or_directory
```

This will format your code according to the project's standards. Ensure you do this before creating a pull request.
You can also configure your IDE to auto-reformat using black upon saving (recommended).

### Testing
To run tests, ensure the pySubstructures environment is activated and run pytest from the project root:

```bash
pytest
```
