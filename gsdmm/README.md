# Gibbs sampling algorithm for the Dirichlet multinomial mixture model (GSDMM)

Largely based on [this code](https://github.com/rwalk/gsdmm), but edited to include `random_state` variable to standardize initial topic settings of documents.

Usage:
```python
from gsdmm import mgp
V = mgp.compute_V(docs)
mg = mgp.MovieGroupProcess(K=40, alpha=.1, beta=.1, n_iters=30)
mg.fit(docs, V, random_state=40)
```

After fitting, the MovieGroupProcess object includes perhaps useful methods:

`label_top_words(nwords=10)` returns a list of length `K` populated with top `nwords` words found in each populated topic. Topics without assigned documents return as empty lists.

`compare_topic_terms(docs, wv=None)` returns the number of topics with similarity >= 0.5, coompared either by a bag-of-words methodology or using Gensim word2vec word vectors `wv`, if specified.

`wmd(docs, wv)` returns the [word mover's distance](http://proceedings.mlr.press/v37/kusnerb15.pdf) by averaging the word mover's distance between all possible pairs of documents within each topic.
