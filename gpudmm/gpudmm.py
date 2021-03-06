import numpy as np
from scipy.stats import multinomial, bernoulli
from gensim.corpora import Dictionary
from gensim.models import TfidfModel
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix
from gensim.models.keyedvectors import WordEmbeddingSimilarityIndex
from gensim.similarities.termsim import SparseTermSimilarityMatrix
from copy import copy

class GPUDMM:
    def __init__(self, K=8, alpha=0.1, beta=0.1, num_iter=30, weight=0.1):

        self.K = K
        self.alpha = alpha
        self.beta = beta
        self.num_iter = num_iter
        self.weight = weight
        self.threshold = 0.5
        self.number_docs = None
        self.vocab_size = None
        self.corpus_vocab_size = None

        self.mz = []
        self.nz = []
        self.nzw = []

        self.wordGPUFlag = []
        self.docToWordIDList = []
        self.assignmentList = []

        self.topicProbabilityGivenWord = []
        self.schemaMap = {}
        self.pz = []
        self.pdz = []
        self.phi = []
        self.dict_ = Dictionary()
        self.corpus = []
        self.corpus_size = None
        self.to_include = []
        self.to_exclude = []
        self.tfidf = False
        self.tf_model = None
        self.weight_min = 1.0

    def _sample(self, p, random_state=None):
        '''
        Sample with probability vector p from a multinomial distribution
        :param p: list
            List of probabilities representing probability vector for the
            multinomial distribution
        :return: int
            index of randomly selected output
        '''

        return np.argmax(multinomial.rvs(1, p, random_state=random_state))

    def initNewModel(self, docs, tfidf=False, random_state=None):

        D = len(docs)
        K = self.K

        self.number_docs = D
        self.tfidf = tfidf
        self.mz = [0 for _ in range(K)]  # cluster doc count
        self.nz = [0 for _ in range(K)]  # cluster word count
        self.nzw = [[] for _ in range(K)]  # cluster word distribution

        corpus = self.corpus
        if tfidf:
            tf = TfidfModel(corpus, dictionary=self.dict_)
            self.tf_model = tf

        self.to_include = np.flatnonzero(corpus)
        self.to_exclude = np.array(list(set(range(D)).difference(set(self.to_include))))

        self.corpus_size = len(self.to_include)

        d_z = [-1 for i in range(D)]
        V, K = self.vocab_size, self.K
        d2wil = [None for _ in range(D)]
        mz, nz, nzw = self.mz, self.nz, self.nzw
        wordGPUFlag = [None for _ in range(D)]

        for id_ in self.to_include:
            doc = corpus[id_]
            tia = []  # termIDArray
            for wordID, count_ in doc:
                if count_ == 1:
                    tia.append(wordID)
                else:
                    tia.extend([wordID]*count_)
            docWordGPUFlag = [False]*len(tia)

            wordGPUFlag[id_] = docWordGPUFlag
            d2wil[id_] = tia

        nzw = [list(np.zeros(V)) for _ in range(K)]
        tpgw = [list(np.zeros(K)) for _ in range(V)]

        zs = multinomial.rvs(1, [1.0 / K for _ in range(K)], D, random_state)
        zs = np.argmax(zs, 1)

        for id_ in self.to_include:
            doc = corpus[id_]
            if tfidf:
                doc = tf[corpus[id_]]
                doc_weight_min = np.min(doc, 0)[1]
                if self.weight_min > doc_weight_min:
                    self.weight_min = doc_weight_min

            z = zs[id_]
            d_z[id_] = z
            mz[z] += 1
            nz[z] += np.sum(doc, 0)[1]

            for wordID, count_ in doc:
                nzw[z][wordID] += count_

        self.mz = mz
        self.nz = nz
        self.nzw = nzw
        self.topicProbabilityGivenWord = tpgw
        self.docToWordIDList = d2wil
        self.assignmentList = d_z
        self.wordGPUFlag = wordGPUFlag
        self.pz = list(np.zeros(K))
        self.phi = [list(np.zeros(V)) for _ in range(K)]
        self.pdz = [list(np.zeros(K)) for _ in range(D)]
        return

    def loadSchema(self, docs, similarity_matrix, full_dict, threshold):
        schema = {}
        self.dict_ = full_dict
        d = self.dict_

        reduced_dict = copy(d)
        # reduced_dict.filter_extremes(min_df,max_df)

        self.corpus = [d.doc2bow(line) for line in docs]
        reduced_dict = Dictionary.from_corpus(self.corpus)
        self.vocab_size = len(full_dict)
        self.corpus_vocab_size = len(reduced_dict)

        for k in reduced_dict.iterkeys():
            similarWordID = similarity_matrix.matrix[k].nonzero()[1]
            if len(similarWordID) > 1:
                similarWordID = [x for x in similarWordID if (x != k) and (x in reduced_dict)]
                if similarWordID:
                    schema.update({k: similarWordID})

        self.schemaMap = schema
        self.threshold = threshold
        return

    def run_iteration(self, random_state=None):
        num_iter, d2wil = self.num_iter, self.docToWordIDList
        K, Vc = self.K, self.corpus_vocab_size
        alpha, beta, C = self.alpha, self.beta, self.corpus_size

        for _iter in range(num_iter):
            self.updateTopicProbabilityGivenWord()
            total_transfers = 0

            for s, id_ in enumerate(self.to_include):
                tia = d2wil[id_]
                preTopic = self.assignmentList[id_]
                self.ratioCount(preTopic, id_, tia, -1, random_state)
                pzDist = []
                for topic in range(K):
                    pz = 1.0 * (self.mz[topic] + alpha) / (C - 1 + K * alpha)
                    value = 1.0

                    for t, termID in enumerate(tia):
                        value *= (self.nzw[topic][termID] + beta) / (self.nz[topic] + Vc * beta + t)

                    value = value * pz
                    pzDist.append(value)

# topic assignment using _sample and pzDist
                pzDist = np.asarray(pzDist)
                pzNorm = np.sum(pzDist)
                pzNorm = pzNorm if pzNorm > 0 else 1
                pzDist = pzDist/pzNorm

                newTopic = self._sample(pzDist, random_state)
                if newTopic != preTopic:
                    total_transfers += 1
                    self.assignmentList[id_] = newTopic

                self.ratioCount(newTopic, id_, tia, 1, random_state)
            if total_transfers == 0 and _iter > 25:
                print("Converged. Breaking out.")
                break

        self.updateTopicProbabilityGivenWord()
        for s, id_ in enumerate(self.to_include):
            tia = d2wil[id_]
            preTopic = self.assignmentList[id_]
            self.ratioCount(preTopic, id_, tia, -1, random_state)
            pzDist = []
            for topic in range(K):
                pz = 1.0 * (self.mz[topic] + alpha) / (C - 1 + K * alpha)
                value = 1.0

                for t, termID in enumerate(tia):
                    value *= (self.nzw[topic][termID] + beta) / (self.nz[topic] + Vc * beta + t)

                value = value * pz
                pzDist.append(value)

# topic assignment as max according to pzDist
            pzDist = np.asarray(pzDist)
            pzNorm = sum(pzDist)
            pzNorm = pzNorm if pzNorm > 0 else 1
            pzDist = pzDist/pzNorm
            newTopic = np.argmax(pzDist)

            if newTopic != preTopic:
                self.assignmentList[id_] = newTopic

            self.ratioCount(newTopic, id_, tia, 1, random_state)

        return

    def ratioCount(self, topic, docID, termIDArray, flag, random_state=None):
        weight = self.weight
        self.mz[topic] += flag
        doc = self.corpus[docID]
        if self.tfidf:
            doc = self.tf_model[doc]
            for wordTuple in doc:
                self.nzw[topic][wordTuple[0]] += flag * wordTuple[1]
                self.nz[topic] += flag * wordTuple[1]

        else:
            for wordID in termIDArray:
                self.nzw[topic][wordID] += flag
                self.nz[topic] += flag

        schemaMap = self.schemaMap

        if flag > 0:
            self.updateWordGPUFlag(docID, topic, random_state)
            wordGPUFlag = self.wordGPUFlag

            for t, wordID in enumerate(termIDArray):
                gpuFlag = wordGPUFlag[docID][t]

                if gpuFlag:
                    if wordID in schemaMap:
                        valueMap = schemaMap[wordID]

                        for entry in valueMap:
                            self.nzw[topic][entry] += weight * self.weight_min
                            self.nz[topic] += weight * self.weight_min

        else:
            wordGPUFlag = self.wordGPUFlag
            for t, wordID in enumerate(termIDArray):
                gpuFlag = wordGPUFlag[docID][t]

                if gpuFlag:
                    if wordID in schemaMap:
                        valueMap = schemaMap[wordID]

                        for entry in valueMap:
                            self.nz[topic] -= weight * self.weight_min
                            self.nzw[topic][entry] -= weight * self.weight_min

    def updateWordGPUFlag(self, docID, newTopic, random_state=None):
        d2wil, wordGPUFlag = self.docToWordIDList, self.wordGPUFlag
        tia = d2wil[docID]
        docWordGPUFlag = []

        for t, termID in enumerate(tia):
            maxProbability = self.findTopicMaxProbabilityGivenWord(termID)
            ratio = self.getTopicProbabilityGivenWord(newTopic, termID) / maxProbability
            if ratio > 0:
                a = bernoulli.rvs(ratio, random_state=random_state)
            else:
                a = 0
            docWordGPUFlag.append(a)

        wordGPUFlag[docID] = docWordGPUFlag
        self.wordGPUFlag = wordGPUFlag
        return

    def getTopicProbabilityGivenWord(self, topic, wordID):
        tpgw = self.topicProbabilityGivenWord
        return tpgw[wordID][topic]

    def findTopicMaxProbabilityGivenWord(self, wordID):
        K, tpgw = self.K, self.topicProbabilityGivenWord
        max_ = -1.0
        for i in range(K):
            tmp = tpgw[wordID][i]

            if tmp > max_:
                max_ = tmp
        return max_

    def updateTopicProbabilityGivenWord(self):
        self.compute_pz()
        self.compute_phi()

        V = self.vocab_size
        tpgw, K = self.topicProbabilityGivenWord, self.K

        for i in range(V):
            row_sum = 0.0
            for j in range(K):
                tpgw[i][j] = self.pz[j] * self.phi[j][i]
                row_sum += tpgw[i][j]

            for j in range(K):
                tpgw[i][j] = tpgw[i][j] / row_sum

        self.topicProbabilityGivenWord = tpgw
        return

    def compute_pz(self):
        nz, K, alpha = self.nz, self.K, self.alpha
        sum_ = sum(nz)

        for i in range(K):
            self.pz[i] = (nz[i] + alpha) / (sum_ + K * alpha)

    def compute_phi(self):
        nzw, K, beta = self.nzw, self.K, self.beta
        Vc, V = self.corpus_vocab_size, self.vocab_size

        for i in range(K):
            sum_ = sum(nzw[i])

            for j in range(V):
                self.phi[i][j] = (nzw[i][j] + beta) / (sum_ + Vc * beta)

    def label_top_words(self, nwords=10):
        '''
        List of top nwords for each cluster
        '''
        beta, K, Vc = self.beta, self.K, self.corpus_vocab_size
        p = [[] for _ in range(K)]

        for ii in range(K):
            p_word = []
            if self.mz[ii]:
                for wordID, value in enumerate(self.nzw[ii]):
                    phi_z_w = (value + beta)/(self.nz[ii] + Vc * beta)
                    if self.dict_[wordID]:
                        p_word.append((self.dict_[wordID], phi_z_w) if self.nzw[ii][wordID] > 0 else (self.dict_[wordID], 0))
            if p_word:
                if len(p_word) >= nwords:
                    p[ii] = [x[0] for x in sorted(p_word, key=lambda x: x[1],
                             reverse=True) if x[1] > 0][:nwords]
                else:
                    p[ii] = [x[0] for x in sorted(p_word, key=lambda x: x[1],
                             reverse=True) if x[1] > 0]
        return p

    def compare_topic_terms(self, wv=None):
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
        topic_no = np.unique(self.assignmentList)
        for q in topic_no:
            subset = docs[np.array(self.assignmentList) == q]
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

    def compute_pdz(self):
        pdz, d2wil, tpgw = self.pdz, self.docToWordIDList, self.topicProbabilityGivenWord
        for id_ in self.to_include:
            tia = d2wil[id_]
            row_sum = 0.0
            for j in range(self.K):
                for wordID in tia:
                    pdz[id_][j] += tpgw[wordID][j]
                row_sum += pdz[id_][j]

            for j in range(self.K):
                pdz[id_][j] = pdz[id_][j] / row_sum

        self.pdz = pdz

    def intercluster(self):
        '''Intercluster distance. Must have run compute_pdz() prior.'''
        K = self.K
        inter_ = 0.0
        for k in range(K):
            if self.mz[k] > 0:
                docs_k = np.where(np.array(self.assignmentList) == k)[0]
                for kp in range(k+1, K):
                    if self.mz[kp] > 0:
                        docs_kp = np.where(np.array(self.assignmentList) == kp)[0]
                        for id_k in docs_k:
                            for id_kp in docs_kp:
                                m = np.multiply(0.5, np.add(self.pdz[id_k], self.pdz[id_kp]))
                                kl_k = k_l(self.pdz[id_k], m)
                                kl_kp = k_l(self.pdz[id_kp], m)
                                dis = 0.5*(kl_k+kl_kp)
                                inter_ += dis/(self.mz[k]*self.mz[kp])
        populated_clusters = np.count_nonzero(self.mz)
        return inter_/(populated_clusters*(populated_clusters - 1))

    def intracluster(self):
        '''Intracluster distance. Must have run compute_pdz() prior.'''
        K = self.K
        intra_k = np.zeros(K)
        for k in range(K):
            intra = 0.0
            if self.mz[k] > 0:
                docs = np.where(np.array(self.assignmentList) == k)[0]
                D = docs.shape[0]
                for i in range(D):
                    id_i = docs[i]
                    for j in range(i+1, D):
                        id_j = docs[j]
                        m = np.multiply(0.5, np.add(self.pdz[id_i],self.pdz[id_j]))
                        kl_i = k_l(self.pdz[id_i], m)
                        kl_j = k_l(self.pdz[id_j], m)
                        dis_ij = 0.5*(kl_i + kl_j)
                        intra += 2*dis_ij/(self.mz[k]*(self.mz[k]-1))
            intra_k[k] = intra
        populated_clusters = np.count_nonzero(self.mz)
        return np.sum(intra_k)/populated_clusters


def buildSchema(wordVectors, docs, threshold=0.5, min_df=1, max_df=1.0):
    '''
    Create the similarity matrix and dictionary that map words to other similar words in the documents.
    Docs here are the full corpus, if using train/test sets later.
    Must use the same min_df and max_df here desired as when running the subsequent steps.

    Can use save and load methods on similarity_matrix and dictionary outputs
    '''
    wordVectors_index = WordEmbeddingSimilarityIndex(wordVectors, threshold, 1.0)
    d = Dictionary(docs)

    if '' in d.token2id:
        id_ = d.token2id['']
        d.filter_tokens([id_])
    d.filter_extremes(min_df, max_df)

    similarity_matrix = SparseTermSimilarityMatrix(wordVectors_index, d)
    return similarity_matrix, d


def k_l(p, q):
    '''Kullback-Leibler divergence'''
    sum_ = 0.0
    for i in range(len(p)):
        sum_ += p[i]*np.log(p[i]/q[i])
    return sum_
