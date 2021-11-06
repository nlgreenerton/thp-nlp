import numpy as np
from scipy.stats import multinomial, bernoulli
from gensim.corpora import Dictionary
from gensim.models import TfidfModel
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix
from gensim.models.keyedvectors import WordEmbeddingSimilarityIndex
from gensim.similarities.termsim import SparseTermSimilarityMatrix


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

        self.mz = [0 for _ in range(K)]  # cluster doc count
        self.nz = [0 for _ in range(K)]  # cluster word count
        self.nzw = [[] for _ in range(K)]  # cluster word distribution

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

    def initNewModel(self, docs, min_df=1, max_df=1.0, tfidf=False, random_state=None):

        D = len(docs)
        self.number_docs = D
        self.tfidf = tfidf
        self.dict_ = Dictionary(docs)

        if '' in self.dict_.token2id:
            id_ = self.dict_.token2id['']
            self.dict_.filter_tokens([id_])
        self.dict_.filter_extremes(min_df, max_df)

        self.corpus = [self.dict_.doc2bow(line) for line in docs]
        self.vocab_size = len(self.dict_)

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

    def loadSchema(self, similarity_matrix, threshold):
        schema = {}
        V = self.vocab_size
        for k in range(V):
            similarWordID = similarity_matrix.matrix.nonzero()[1][
                            similarity_matrix.matrix.nonzero()[0] == k]
            similarWordID = similarWordID[similarWordID != k]
            schema.update({k: similarWordID})

        self.schemaMap = schema
        self.threshold = threshold
        return

    def run_iteration(self, random_state=None):
        num_iter, d2wil = self.num_iter, self.docToWordIDList
        K, V = self.K, self.vocab_size
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
                        value *= (self.nzw[topic][termID] + beta) / (self.nz[topic] + V * beta + t)

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

# original random topic assignment
                # for i in range(1,K):
                #     pzDist[i] += pzDist[i-1]
                # u = np.random.rand() * pzDist[-1]
                # newTopic = -1
                #
                # for i in range(K):
                #     if pzDist[i] >= u:
                #         newTopic = i
                #         self.assignmentList[id_] = newTopic
                #         total_transfers += 1
                #         break

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
                    value *= (self.nzw[topic][termID] + beta) / (self.nz[topic] + V * beta + t)

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
        nzw, K, beta, V = self.nzw, self.K, self.beta, self.vocab_size

        for i in range(K):
            sum_ = sum(nzw[i])

            for j in range(V):
                self.phi[i][j] = (nzw[i][j] + beta) / (sum_ + V * beta)

    def label_top_words(self, nwords=10):
        '''
        List of top nwords for each cluster
        '''
        beta, K, V = self.beta, self.K, self.vocab_size
        p = [[] for _ in range(K)]

        for ii in range(K):
            p_word = []
            if self.mz[ii]:
                for wordID, value in enumerate(self.nzw[ii]):
                    phi_z_w = (value + beta)/(self.nz[ii] + V * beta)
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


def buildSchema(wordVectors, docs, threshold=0.5, min_df=1, max_df=1.0):
    '''
    Create the schema that maps words to other similar words in the documents.
    Docs here should match docs later used.
    Must use the same min_df and max_df here as when running the initNewModel() step.

    can use save and load methods on similarity_matrix
    '''
    wordVectors_index = WordEmbeddingSimilarityIndex(wordVectors, threshold, 1.0)
    d = Dictionary(docs)

    if '' in d.token2id:
        id_ = d.token2id['']
        d.filter_tokens([id_])
    d.filter_extremes(min_df, max_df)

    similarity_matrix = SparseTermSimilarityMatrix(wordVectors_index, d)
    return similarity_matrix
