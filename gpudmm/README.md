## GPU-based Dirichlet multinomial mixture model (GPUDMM)

Largely based on [this code](https://github.com/WHUIR/GPUDMM/tree/2b89d949557e99cbfce714a7c486f954f0c65ee8), but implemented in Python. Changed to include `random_state` variable to standardize initial topic settings of documents and for reproducibility. TFIDF-weighting can be toggled with `tfidf` setting. Documents without any tokens are assigned label -1 and not involved in any fitting steps.

Usage:
```python
from gpudmm import gpudmm
schema = gpudmm.buildSchema(wv, docs, threshold=0.5)
```

`wv`: [Gensim](f)
`docs`: iterable of iterables, tokenized documents for fitting

A schema dictionary connects tokens with similarity >= `threshold` according to the supplied word vectors `wv` in Gensim word2vec KeyedVectors format.

```python
gpu = gpudmm.GPUDMM(K=40, alpha=.1, beta=.1, num_iter=30, weight=.1)
gpu.initNewModel(docs, random_state=40)
gpu.loadSchema(schema, threshold=0.5)
gpu.run_iteration(random_state=40)
```


Hyperparameters `alpha` and `beta` are as described in the [GSDMM model](https://github.com/nlgreenerton/thp-nlp/tree/main/gsdmm).

`weight`: float between 0 and 1 that promotes similar words also in the overall vocabulary.

After fitting, the GPUDMM object includes perhaps useful methods:

`label_top_words(nwords=10)` returns a list of length `K` populated with top `nwords` words found in each populated topic. Topics without assigned documents return as empty lists.

`compare_topic_terms(docs, wv=None)` returns the number of topics with cosine similarity >= 0.5, coompared either by a bag-of-words methodology or using Gensim word2vec word vectors `wv`, if specified.

`wmd(docs, wv)` returns the [word mover's distance](http://proceedings.mlr.press/v37/kusnerb15.pdf) by averaging the word mover's distance between all possible pairs of documents within each topic.
