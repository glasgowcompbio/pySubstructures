# pySubstructures

A Python package to perform unsupervised discoveries of motifs from tandem mass spectrometry data.

Python TODO:
- [X] Add codes from the lda repo.
- [X] Tidy up codes, and keep only the ones we actually use.
- [ ] MolnetEnhancer support
- [ ] MotifDB support when running LDA inference
- [ ] [MESSAR](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0226770) support
- [ ] [Labelled LDA](https://github.com/gkreder/ms2-topic-model) support
- [ ] [Matchms](https://github.com/matchms/matchms) compatibility
- [ ] [PALS](https://pals.glasgowcompbio.org/) support
- [ ] Create a Python package

Maybe:
- [ ] MAGMa-MS2LDA support
- [ ] Gensim support
- [ ] Classyfire functions

Visualisation TODO:
- [ ] Setup skeleton project for the offline viewer

MS2LDA.org TODO:
- [ ] Replace the lda codes used on the server with this package
- [X] Migrate server to Django 2.0 / Python 3

## Installation for development
### Prepare environment

```
conda create --name pySubstructures python=3.7
conda activate pySubstructures
```
### Clone repository
Clone the present repository, e.g. by running
```
git clone https://github.com/glasgowcompbio/pySubstructures.git
```
And then install the required dependencies, e.g. by running the following from within the cloned directory
```
pip install -e .
```