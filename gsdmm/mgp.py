import numpy as np
from scipy.stats import multinomial
from gensim.corpora import Dictionary
from gensim.models import TfidfModel
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix


class MovieGroupProcess:
    def __init__(self, K=8, alpha=0.1, beta=0.1, n_iters=30):
        '''
        A MovieGroupProcess is a conceptual model introduced by Yin and Wang
        2014 to describe their Gibbs sampling algorithm for a Dirichlet Mixture
        Model for the clustering short text documents.
        Reference: https://doi.org/10.1145/2623330.2623715
        Imagine a professor is leading a film class. At the start of the class,
        the students are randomly assigned to K tables. Before class begins,
        the students make lists of their favorite films. The teacher reads the
        role n_iters times. When a student is called, the student must select a
        new table satisfying either:
        1) The new table has more students than the current table.
        OR
        2) The new table has students with similar lists of favorite movies.
        :param K: int
            Upper bound on the number of possible clusters.
        :param alpha: float between 0 and 1
            Alpha controls the probability that a student will join a table
            that is currently empty
            When alpha is 0, no one will join an empty table.
        :param beta: float between 0 and 1
            Beta controls the student's affinity for other students with
            similar interests. A low beta means that students desire to sit
            with students of similar interests. A high beta means they are less
            concerned with affinity and are more influenced by the popularity
            of a table
        :param n_iters:
            Number of iterations
        '''
        self.K = K
        self.alpha = alpha
        self.beta = beta
        self.n_iters = n_iters

        # slots for computed variables
        self.number_docs = None
        self.vocab_size = None
        self.m_z = [0 for _ in range(K)]  # cluster doc count
        self.n_z = [0 for _ in range(K)]  # cluster word count
        self.n_z_w = [{} for i in range(K)]  # cluster word distrib
        self.d_z = []
        self.pdz = []
        self.dict_ = Dictionary()
        self.corpus = []
        self.corpus_size = None
        self.to_include = []
        self.to_exclude = []
        self.tfidf = False
        self.tf_model = None

    @staticmethod
    def _sample(p, random_state=None):
        '''
        Sample with probability vector p from a multinomial distribution
        :param p: list
            List of probabilities representing probability vector for the
            multinomial distribution
        :return: int
            index of randomly selected output
        '''

        return np.argmax(multinomial.rvs(1, p, random_state=random_state))

    def initNewModel(self, docs, min_df=1, max_df=1.0, tfidf=False, random_state=None):
        '''
        Cluster the input documents
        '''
        K = self.K

        D = len(docs)
        self.number_docs = D
        self.tfidf = tfidf

        self.d_z = [-1 for i in range(len(docs))]

        # create dictionary
        self.dict_ = Dictionary(docs)
        self.dict_.filter_extremes(min_df, max_df)
        if '' in self.dict_.token2id:
            id_ = self.dict_.token2id['']
            self.dict_.filter_tokens([id_])

        self.corpus = [self.dict_.doc2bow(line) for line in docs]
        self.vocab_size = len(self.dict_)

        corpus = self.corpus
        if tfidf:
            tf = TfidfModel(corpus, dictionary=self.dict_)
            self.tf_model = tf
        self.to_include = np.flatnonzero(corpus)
        # self.to_exclude = np.array(list(set(range(D)).difference(set(self.to_include))))

        # self.corpus = [self.dict_.doc2bow(line) for line in docs[to_include]]
        # corpus = self.corpus
        self.corpus_size = len(self.to_include)
        C = self.corpus_size

        # initialize the clusters
        zs = multinomial.rvs(1, [1.0 / K for _ in range(K)], C, random_state)
        zs = np.argmax(zs, 1)

        for i, id_ in enumerate(self.to_include):
            doc = corpus[id_]
            if tfidf:
                doc = tf[corpus[id_]]

            z = zs[i]
            self.d_z[id_] = z
            self.m_z[z] += 1
            self.n_z[z] += np.sum(doc, 0)[1]

            for wordID, count_ in doc:
                if wordID not in self.n_z_w[z]:
                    self.n_z_w[z][wordID] = 0
                self.n_z_w[z][wordID] += count_

            self.pdz = [list(np.zeros(K)) for _ in range(D)]

    def fit(self):
        n_iters, corpus, K = self.n_iters, self.corpus, self.K
        cluster_count = K

        for _iter in range(n_iters):
            total_transfers = 0

            for id_ in self.to_include:
                doc = corpus[id_]
                if self.tfidf:
                    doc = self.tf[corpus[id_]]

                z_old = self.d_z[id_]
                self.m_z[z_old] -= 1
                self.n_z[z_old] -= np.sum(doc, 0)[1]

                for wordID, count_ in doc:
                    self.n_z_w[z_old][wordID] -= count_

                    if self.n_z_w[z_old][wordID] == 0:
                        del self.n_z_w[z_old][wordID]

                p = self.score(doc, id_)
                z_new = self._sample(p) #, random_state)

                if z_old != z_new:
                    total_transfers += 1

                self.d_z[id_] = z_new
                self.m_z[z_new] += 1
                self.n_z[z_new] += np.sum(doc, 0)[1]

                for wordID, count_ in doc:
                    if wordID not in self.n_z_w[z_new]:
                        self.n_z_w[z_new][wordID] = 0
                    self.n_z_w[z_new][wordID] += count_

            cluster_count_new = sum([1 for v in self.m_z if v > 0])
            if (total_transfers == 0) and (_iter > 25) and (cluster_count_new == cluster_count):
                print("Converged.  Breaking out.")
                break

            cluster_count = cluster_count_new
        return

    def score(self, doc, docID):
        '''
        Score a document
        Implements formula (3) of Yin and Wang 2014.
        http://dbgroup.cs.tsinghua.edu.cn/wangjy/papers/KDD14-GSDMM.pdf
        :param doc: list[str]:
            The doc token stream
        :return: list[float]:
            A length K probability vector where each component represents
            the probability of the document appearing in a particular cluster
        '''
        alpha, beta, K = self.alpha, self.beta, self.K
        V, C = self.vocab_size, self.corpus_size

        p = [0 for _ in range(K)]

        lD1 = np.log(C - 1 + K * alpha)
        doc_size = np.sum(doc, 0)[1]
        if self.tfidf:
            bow = self.corpus[docID]
            doc_size = np.sum(bow, 0)[1]

        for label in range(K):
            lN1 = np.log(self.m_z[label] + alpha)
            lN2 = 0
            lD2 = 0
            for wordID, count_ in doc:
                lN2 += count_*np.log(self.n_z_w[label].get(wordID, 0) + beta)
            for j in range(1, doc_size + 1):
                lD2 += np.log(self.n_z[label] + V * beta + j - 1)
            p[label] = np.exp(lN1 - lD1 + lN2 - lD2)

        # normalize the probability vector
        pnorm = sum(p)
        pnorm = pnorm if pnorm > 0 else 1

        return [pp/pnorm for pp in p]

    def choose_best_label(self, doc, docID):
        '''
        Choose the highest probability label for the input document
        :param doc: list[str]:
            The doc token stream
        '''
        p = self.score(doc, docID)
        return np.argmax(p)

    def label_top_words(self, nwords=10):
        '''
        List of top nwords for each cluster
        '''
        beta, K, V = self.beta, self.K, self.vocab_size

        p = [[] for _ in range(K)]
        for ii in range(K):
            if self.m_z[ii]:
                p_word = []
                for word in self.n_z_w[ii].keys():
                    phi_z_w = (self.n_z_w[ii][word] + beta)/(self.n_z[ii] + V * beta)
                    if word:
                        p_word.append((word, phi_z_w))
                if p_word:
                    if len(p_word) >= nwords:
                        p[ii] = [self.dict_[x[0]] for x in sorted(p_word, key=lambda x: x[1],
                                 reverse=True)][:nwords]
                    else:
                        p[ii] = [self.dict_[x[0]] for x in sorted(p_word, key=lambda x: x[1],
                                 reverse=True)]
        return p

    def compare_topic_terms(self, wv=None):
        '''
        Compare topic terms with or without word vectors from calculated model
        '''
        dim, V = self.K, self.vocab_size
        words = self.label_top_words()
        tri = np.diag(np.ones(dim))

        if not wv:
            for ii in range(dim):
                for jj in range(ii+1, dim):
                    nwords = min(len(words[ii]), len(words[jj]))
                    if nwords == 10:
                        v1 = csr_matrix(([1]*nwords, ([0]*nwords, list(map(lambda x: self.dict_.token2id[x], words[ii][:nwords])))), shape=(1, V), dtype=float)
                        v2 = csr_matrix(([1]*nwords, ([0]*nwords, list(map(lambda x: self.dict_.token2id[x], words[jj][:nwords])))), shape=(1, V), dtype=float)
                    else:
                        v1_nwords = len(words[ii])
                        v2_nwords = len(words[jj])
                        v1 = csr_matrix(([1]*v1_nwords, ([0]*v1_nwords, list(map(lambda x: self.dict_.token2id[x], words[ii])))), shape=(1, V), dtype=float)
                        v2 = csr_matrix(([1]*v2_nwords, ([0]*v2_nwords, list(map(lambda x: self.dict_.token2id[x], words[jj])))), shape=(1, V), dtype=float)
                    tri[ii, jj] = cosine_similarity(v1, v2)
        else:
            for ii in range(dim):
                for jj in range(ii+1, dim):
                    v1 = list(filter(lambda q: q in wv.vocab, words[ii]))
                    v2 = list(filter(lambda q: q in wv.vocab, words[jj]))
                    if v1 and v2:
                        tri[ii, jj] = wv.n_similarity(v1, v2)

        if np.argwhere(np.triu(tri, 1) >= 0.5).any():
            c = np.argwhere(np.triu(tri, 1) >= 0.5)
            return np.unique(c.flatten('F')[:c.shape[0]]).shape[0]
        return 0

    def wmd(self, docs, wv):
        '''
        Compute word mover's distance from calculated model. Word vectors are of normalized Gensim word2vec type
        '''
        sum_, tot = 0.0, 0
        topic_no = np.unique(self.d_z)

        for q in topic_no:
            subset = docs[np.array(self.d_z) == q]
            ndocs = len(subset)
            if ndocs > 1:
                for ii in range(ndocs):
                    d1 = subset.iloc[ii]
                    for jj in range(ii+1, ndocs):
                        d2 = subset.iloc[jj]
                        if d1[0] and d2[0]:
                            tot += 1
                            sum_ += wv.wmdistance(' '.join(d1), ' '.join(d2))
        return sum_/tot
