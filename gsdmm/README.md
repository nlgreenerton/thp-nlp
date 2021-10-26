## Gibbs sampling algorithm for the Dirichlet multinomial mixture model (GSDMM)

Largely based on [this code](https://github.com/rwalk/gsdmm) but edited to include `random_state` variable to standardize initial topic settings of documents and added tfidf-weighting option. Documents without any tokens are assigned label -1 and not involved in any fitting steps.

Usage:
```python
from gsdmm import mgp
mg = mgp.MovieGroupProcess(K=40, alpha=.1, beta=.1, n_iters=30)
```

- `K`: integer maximum number of clusters, which should be well above the expected final number

- `alpha`: float between 0 and 1 that controls the tendency for documents to be placed in empty clusters

- `beta`: float between 0 and 1 that controls the importance for documents within a cluster to share topic terms

- `n_iters`: integer number of iterations

```python
mg.initNewModel(docs, min_df=1, max_df=1.0, tfidf=False, random_state=40)
mg.fit()
```

- `docs`: iterable of iterables, tokenized documents for fitting

- `min_df`: integer lower bound of occurrences for tokens

- `max_df`: float between 0 and 1, upper bound for tokens

- `tfidf`: boolean whether or not to implement tfidf-weighting

- `random_state`: integer seed

More information and figures illustrating the effects of hyperparameters can be found [here](https://dl.acm.org/doi/10.1145/2623330.2623715).

After fitting, the MovieGroupProcess object includes perhaps useful methods:

`label_top_words(nwords=10)` returns a list of length `K` populated with top `nwords` words found in each populated topic. Topics without assigned documents return as empty lists.

`compare_topic_terms(docs, wv=None)` returns the number of topics with similarity >= 0.5, compared either by a bag-of-words methodology or using [Gensim](https://radimrehurek.com/gensim_3.8.3/index.html) word2vec word vectors in KeyedVectors format `wv`, if specified.

`wmd(docs, wv)` returns the [word mover's distance](http://proceedings.mlr.press/v37/kusnerb15.pdf) by averaging the word mover's distance between all possible pairs of documents within each topic. Word vectors `wv` again in Gensim KeyedVectors format must previously be normalized using the `init_sims()` method.
