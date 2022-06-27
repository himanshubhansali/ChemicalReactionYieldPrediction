# Predicting Chemical Reaction yield using machine learning
Artificial intelligence has recently seen numerous applications in synthetic organic chemistry. One such application is predicting the yield of the known chemical reactions which could guide chemists and help them select high-yielding reactions and score synthesis routes, reducing the number of attempt. Here, we use nlp inspired Drfp encoder for the yield prediction task for two different chemical reaction dataset. A differential reaction fingerprint DRFP algorithm takes a reaction SMILES as an input and creates a binary fingerprint based on the symmetric difference of two sets containing the circular molecular n-grams generated from the molecules listed left and right from the reaction arrow, respectively, without the need for distinguishing between reactants and reagents. 

## Approach 
We used Drfp encoder for converting the reaction smiles into a binary fingerprint. These fingerprints are given as a input to the XGB Regressor Model which is then used for our yield prediction task.


## Installation 
As the library is based on the chemoinformatics toolkit [RDKit](http://www.rdkit.org) it is best installed using the [Anaconda](https://docs.conda.io/en/latest/miniconda.html) package manager. Once you have conda, you can simply run:

```
conda create -n yields python=3.6 -y
conda activate yields
conda install -c rdkit rdkit=2020.03.3.0 -y
conda install -c tmap tmap -y
```

```
git clone https://github.com/rxn4chemistry/rxn_yields.git
cd rxn_yields
pip install -e .
```
*DRFP* can be installed from pypi using `pip install drfp`


## Documentation

The library contains the class `DrfpEncoder` with one public method `encode`.

| `DrfpEncoder.encode()` | Description | Type | Default |
|-|-|-|-|
| `X` | An iterable (e.g. a list) of reaction SMILES or a single reaction SMILES to be encoded | `Iterable` or `str` |  |
| `n_folded_length` | The folded length of the fingerprint (the parameter for the modulo hashing) | `int` | `2048` |
| `min_radius` | The minimum radius of a substructure (0 includes single atoms) | `int` | `0` |
| `radius` | The maximum radius of a substructure | `int` | `3` |
| `rings` | Whether to include full rings as substructures | `bool` | `True` |
| `mapping` |  Return a feature to substructure mapping in addition to the fingerprints. If true, the return signature of this method is `Tuple[List[np.ndarray], Dict[int, Set[str]]]` | `bool` | `False` |


## Preparation of dataset 
 
The Buchwald-Hatwig reaction dataset consists of four columns ,i.e, Ligands,Bases,Aryl Halides and additives in their molecular form. We used the MolToSmiles function from rdkit library to convert these four columns into single text based representation known as smiles.   

<div style="text-align: center">
<img src="images/dataset.png" width="800" style="max-width: 800px">

---



