# ALTEGRAD Kaggle Challenge - Predicting the h-index of authors

## Description

The goal of this challenge is to study and apply machine learning/artificial intelligence techniques to
a real-world regression problem. In this regression problem, each sample corresponds to a researcher
(i. e., an author of research papers), and the goal is to build a model that can predict accurately the
h-index of each author. More specifically, the h-index of an author measures his/her productivity and
the citation impact of his/her publications. It is defined as the maximum value of h such that the
given author has published h papers that have each been cited at least h times.

## Get started

### Requirements

At the root of the project, run `pip install -r requirements.txt`

### Data/Input

First, make sure you have the data downloaded from the handout at `data/`.

Then, under `src/processing`, run:

```
python paper_representations.py
python author_representations.py
```

### File orgnization

All the code is found under the `src/` directory:

- `baselines/` contains the baseline models for evaluating h-indices
- `notebooks/` is for data exploration and ideas on modeling
- `processing/` is for the processing logic (input/model)
