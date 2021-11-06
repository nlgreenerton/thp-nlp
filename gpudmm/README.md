## GPU-based Dirichlet multinomial mixture model (GPUDMM)

Largely based on [this code](https://github.com/WHUIR/GPUDMM/tree/2b89d949557e99cbfce714a7c486f954f0c65ee8), but implemented in Python. Changed to include `random_state` variable to standardize initial topic settings of documents and for reproducibility. TFIDF-weighting can be toggled with `tfidf` setting. Documents without any tokens are assigned label -1 and not involved in any fitting steps. The code will run some specified number of iterations then, upon completion of those iterations, do a final topic assignment based on the highest probability label for each document.

Usage:
```python
from gpudmm import gpudmm
similarity_matrix = gpudmm.buildSchema(wv, docs, threshold=0.5, min_df=1, max_df=1.0)
```

- `wv`: word vectors in [Gensim](https://radimrehurek.com/gensim_3.8.3/index.html) word2vec KeyedVectors format

- `docs`: iterable of iterables, tokenized documents for fitting

- `threshold`: float between 0 and 1, similarity cutoff above which word pair is considered similar enough to be retained in the schema

- `min_df`: integer lower bound of occurrences for tokens

- `max_df`: float between 0 and 1, upper bound for tokens

This first step creates the SparseTermSimilarityMatrix from Gensim, which supports save and load methods. The parameters here must match those used in later steps. As this first step is quite time-consuming so it wouldn't hurt to save the similarity_matrix output for later reuse.

```python
gpu = gpudmm.GPUDMM(K=40, alpha=.1, beta=.1, num_iter=30, weight=.1)
```

- `K`: integer maximum number of clusters, which should be well above the expected final number

- `alpha`: float between 0 and 1 that controls the tendency for documents to be placed in empty clusters

- `beta`: float between 0 and 1 that controls the importance for documents within a cluster to share topic terms

- `num_iter`: integer number of iterations

- `weight`: float between 0 and 1 that promotes similar words also in the overall vocabulary

Hyperparameters `alpha` and `beta` are also described in the [GSDMM model](https://github.com/nlgreenerton/thp-nlp/tree/main/gsdmm).

```python
gpu.initNewModel(docs, min_df=1, max_df=1.0, tfidf=False, random_state=40)
gpu.loadSchema(similarity_matrix, threshold=0.5)
gpu.run_iteration()
```

- `docs`: iterable of iterables, tokenized documents for fitting

- `min_df`: integer lower bound of occurrences for tokens

- `max_df`: float between 0 and 1, upper bound for tokens

- `tfidf`: boolean whether or not to implement tfidf-weighting

- `random_state`: integer seed

- `schema`: output from `buildSchema()`

- `threshold`: float between 0 and 1, threshold used in `buildSchema()`

After fitting, the GPUDMM object includes other methods:

`label_top_words(nwords=10)` returns a list of length `K` populated with top `nwords` words found in each populated topic. Topics without assigned documents return as empty lists.

`compare_topic_terms(wv=None)` returns the number of topics with cosine similarity >= 0.5, compared either by a bag-of-words methodology or using Gensim word2vec word vectors `wv`, if specified.

`wmd(docs, wv)` returns the [word mover's distance](http://proceedings.mlr.press/v37/kusnerb15.pdf) by averaging the word mover's distance between all possible pairs of documents within each topic.

`compute_pdz()` calculates the topic probability given document for the fitted model, assigning the result to the pdz attribute.
