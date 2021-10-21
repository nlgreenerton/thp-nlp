## Gibbs sampling algorithm for the Dirichlet multinomial mixture model (GSDMM)

Largely based on [this code](https://github.com/rwalk/gsdmm), but edited to include `random_state` variable to standardize initial topic settings of documents.

Usage:
```python
from gsdmm import mgp
V = mgp.compute_V(docs)
mg = mgp.MovieGroupProcess(K=40, alpha=.1, beta=.1, n_iters=30)
mg.fit(docs, V, random_state=40)
```

`K`: maximum number of clusters, which should be well above the expected final number.

`alpha`: float between 0 and 1 that controls the tendency for documents to be placed in empty clusters.

`beta`: float bewteen 0 and 1 that controls the importance for documents within a cluster to share topic terms.

More information and figures illustrating the effects of hyperparameters can be found [here](https://dl.acm.org/doi/10.1145/2623330.2623715).

After fitting, the MovieGroupProcess object includes perhaps useful methods:

`label_top_words(nwords=10)` returns a list of length `K` populated with top `nwords` words found in each populated topic. Topics without assigned documents return as empty lists.

`compare_topic_terms(docs, wv=None)` returns the number of topics with similarity >= 0.5, coompared either by a bag-of-words methodology or using Gensim word2vec word vectors in KeyedVectors format `wv`, if specified.

`wmd(docs, wv)` returns the [word mover's distance](http://proceedings.mlr.press/v37/kusnerb15.pdf) by averaging the word mover's distance between all possible pairs of documents within each topic. Word vectors `wv` again in Gensim KeyedVectors format must previously be normalized using the `init_sims()` method.
