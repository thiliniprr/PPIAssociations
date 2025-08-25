# Biomedical Text Mining and Literature Ranking

This repository contains two main scripts for biomedical NLP and literature enrichment:

1. **chemprot_relation_classification.py**: Fine-tunes [BioMedBERT](https://huggingface.co/microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract) on the ChemProt relation extraction task.
2. **ppi_literature_ranker.py**: Retrieves recent high-impact literature on protein-protein interactions (PPI) from PubMed, enriches them with citation and impact data, and ranks them by composite quality.

---

## Table of Contents

- [chemprot_relation_classification.py](#chemprot_relation_classificationpy)
  - [Purpose](#purpose)
  - [Requirements](#requirements)
  - [Usage](#usage)
  - [Output](#output)
- [ppi_literature_ranker.py](#ppi_literature_rankerpy)
  - [Purpose](#purpose-1)
  - [Requirements](#requirements-1)
  - [Usage](#usage-1)
  - [Output](#output-1)


---

## chemprot_relation_classification.py

### Purpose

This script fine-tunes the **BioMedBERT** language model for binary relation classification (chemical-gene relations) using the [ChemProt](https://huggingface.co/datasets/bigbio/chemprot) dataset. It preprocesses annotated abstracts, generates all possible CHEMICAL-GENE pairs, and predicts whether a given pair has a relation.

### Requirements

- Python 3.7+
- [transformers](https://huggingface.co/docs/transformers)
- [datasets](https://huggingface.co/docs/datasets)
- [scikit-learn](https://scikit-learn.org/stable/)
- [torch](https://pytorch.org/)

Install requirements:
```bash
pip install transformers datasets scikit-learn torch
```

### Usage

```bash
python chemprot_relation_classification.py
```

**Customizations**:
- You may change the `num_train_epochs`, `learning_rate`, or other hyperparameters in the script.
- By default, all possible CHEMISTRY-GENE pairs are classified (not just those with known relations).

### Output

- Trained model checkpoints will be written to `./bio-chemprot-bert` (by default).
- After training, the script prints predicted labels for the test set (1 = relation, 0 = no relation).


---

## ppi_literature_ranker.py

### Purpose

This script automates the discovery and ranking of recent, high-impact publications about **protein-protein interactions** (PPI), using data from PubMed and other APIs such as [SemanticScholar](https://www.semanticscholar.org/product/api), [CrossRef](https://www.crossref.org/), [Unpaywall](https://unpaywall.org/products/api), and [Altmetric](https://api.altmetric.com/).

#### Features
- **Freshness** control (e.g., only last 5 years)
- Composite impact score (citations, influential citations, journal impact proxy, altmetrics, open access, etc)
- Output: ranked table of top papers and their abstracts

### Requirements

- Python 3.7+
- [biopython](https://biopython.org/)
- [requests](https://docs.python-requests.org/)
- [tqdm](https://tqdm.github.io/)
- [pandas](https://pandas.pydata.org/)

Install requirements:
```bash
pip install biopython requests tqdm pandas
```

### Usage

1. **Set your email** at the top of the script (required by NCBI and Unpaywall):

    ```python
    Entrez.email = "your.email@domain"    # Replace with your email
    ```

2. **Run the script**:

    ```bash
    python ppi_literature_ranker.py
    ```

### Output

- Prints a ranked list of top N (default: 20) PPI publications sorted by a composite quality score, including basic metadata and abstract excerpts.

---

**Contact**:  
If you have questions or suggestions, please open an issue or contact nadeesha.meegahage@gmail.com.

---

**Notes**:
- API keys or higher rate limits may be necessary for large-scale queries to some APIs (Unpaywall, Altmetric, etc).
- The ChemProt script marks entity pairs in text with simple string replacement. For production workloads, improved entity handling may be necessary.

---

**Citation**:  
If you use this repository, please cite as appropriate and give credit to [microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract](https://huggingface.co/microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract) and relevant datasets/APIs.

