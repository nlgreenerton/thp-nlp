import numpy as np
# from numpy.random import multinomial
from scipy.stats import multinomial, bernoulli
from gensim.corpora import Dictionary
from gensim.models.keyedvectors import WordEmbeddingSimilarityIndex
from gensim.similarities.termsim import SparseTermSimilarityMatrix
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix


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
        self.word2id = {}
        self.id2word = {}
        self.wordIDFMap = {}
        self.docUsefulWords = {}
        self.wordSet = set()
        self.topWordIDList = []
        self.assignmentList = []
        self.wordGPUInfo = []

        self.topicProbabilityGivenWord = []
        self.schemaMap = {}
        self.pz = []
        self.pdz = []
        self.phi = []

    def _sample(self, p, random_state=None):
        '''
        Sample with probability vector p from a multinomial distribution
        :param p: list
            List of probabilities representing probability vector for the
            multinomial distribution
        :return: int
            index of randomly selected output
        '''
        # return [i for i, entry in enumerate(multinomial.rvs(1, p, random_state=random_state)) if entry != 0][0]
        return np.argmax(multinomial.rvs(1, p, random_state=random_state))

    def initNewModel(self, docs, random_state=None):
        self.id2word = {k[0]: k[1] for k in Dictionary(docs).items()}
        self.word2id = {k[1]: k[0] for k in Dictionary(docs).items()}
        w2i = self.word2id

        D = len(docs)
        self.number_docs = D
        self.vocab_size = len(w2i)
        V = self.vocab_size
        d2wil = self.docToWordIDList
        mz, nz, nzw = self.mz, self.nz, self.nzw
        K, d_z = self.K, self.assignmentList
        wordGPUFlag, wordGPUInfo = self.wordGPUFlag, self.wordGPUInfo

        for i in range(D):
            doc = docs[i]
            tia = list(map(lambda x: w2i[x], doc))  # termIDArray
            docWordGPUFlag = [False]*len(tia)
            docWordGPUInfo = {}

            wordGPUFlag.append(docWordGPUFlag)
            wordGPUInfo.append(docWordGPUInfo)
            d2wil.append(tia)

        nzw = [list(np.zeros(V)) for _ in range(K)]
        tpgw = [list(np.zeros(K)) for _ in range(V)]

        zs = multinomial.rvs(1, [1.0 / K for _ in range(K)], D, random_state)
        zs = np.argmax(zs, 1)
        for i, doc in enumerate(docs):
            # z = self._sample([1.0 / K for _ in range(K)])
            z = zs[i]
            d_z.append(z)
            mz[z] += 1
            nz[z] += len(doc)

            for word in doc:
                wordID = w2i[word]
                nzw[z][wordID] += 1

        self.mz = mz
        self.nz = nz
        self.nzw = nzw
        self.topicProbabilityGivenWord = tpgw
        self.docToWordIDList = d2wil
        self.assignmentList = d_z
        self.wordGPUFlag = wordGPUFlag
        self.wordGPUInfo = wordGPUInfo
        self.pz = list(np.zeros(K))
        self.phi = [list(np.zeros(V)) for _ in range(K)]
        self.pdz = [list(np.zeros(K)) for _ in range(D)]
        return

    def loadSchema(self, schema, threshold):
        keep = self.id2word.keys()
        filt_schema = dict(filter(lambda x: x[0] in keep, schema.items()))
        final_schema = dict()
        for k, v in filt_schema.items():
            final_schema[k] = []
            for num in v:
                if num in keep:
                    final_schema[k].append(num)
        self.schemaMap = final_schema
        self.threshold = threshold
        return

    def run_iteration(self, random_state=None):
        num_iter, d2wil = self.num_iter, self.docToWordIDList
        K, V = self.K, self.vocab_size
        alpha, beta, D = self.alpha, self.beta, self.number_docs

        for _iter in range(num_iter):
            self.updateTopicProbabilityGivenWord()
            total_transfers = 0

            for s, tia in enumerate(d2wil):
                preTopic = self.assignmentList[s]
                self.ratioCount(preTopic, s, tia, -1, random_state)
                pzDist = []
                for topic in range(K):
                    pz = 1.0 * (self.mz[topic] + alpha) / (D - 1 + K * alpha)
                    value, logSum = 1.0, 0.0

                    for t, termID in enumerate(tia):
                        value *= (self.nzw[topic][termID] + beta) / (self.nz[topic] + V * beta + t)

                    value = value * pz
                    pzDist.append(value)

# original random-esque topic assignment
                # for i in range(1, K):
                #     pzDist[i] += pzDist[i-1]
                #
                # u = np.random.rand() * pzDist[-1]
                # newTopic = -1
                #
                # for i in range(K):
                #     if pzDist[i] >= u:
                #         newTopic = i
                #         self.assignmentList[s] = newTopic
                #         total_transfers += 1
                #         break

# topic assignment using _sample and pzDist
                pzDist = np.asarray(pzDist)
                pzNorm = np.sum(pzDist)
                pzNorm = pzNorm if pzNorm > 0 else 1
                pzDist = pzDist/pzNorm

                newTopic = self._sample(pzDist, random_state)
                if newTopic != preTopic:
                    total_transfers += 1
                    self.assignmentList[s] = newTopic

# topic assignment as max according to pzDist
                # pzDist = np.asarray(pzDist)
                # pzNorm = sum(pzDist)
                # pzNorm = pzNorm if pzNorm > 0 else 1
                # pzDist = pzDist/pzNorm
                # newTopic = np.argmax(pzDist)
                # if newTopic != preTopic:
                   # total_transfers += 1

                self.ratioCount(newTopic, s, tia, 1, random_state)
            if total_transfers == 0 and _iter > 25:
                print("Converged.  Breaking out.")
                break
        return  # self.assignmentList

    def ratioCount(self, topic, docID, termIDArray, flag, random_state=None):
        wordGPUInfo, weight = self.wordGPUInfo, self.weight
        self.mz[topic] += flag
        for wordID in termIDArray:
            self.nzw[topic][wordID] += flag
            self.nz[topic] += flag

        schemaMap = self.schemaMap
        gpuInfo = {}
        if flag > 0:
            self.updateWordGPUFlag(docID, topic, random_state=random_state)
            wordGPUFlag = self.wordGPUFlag

            for t, wordID in enumerate(termIDArray):
                gpuFlag = wordGPUFlag[docID][t]

                if gpuFlag:
                    if wordID in schemaMap:
                        valueMap = schemaMap[wordID]

                        for entry in valueMap:
                            self.nzw[topic][entry] += weight
                            self.nz[topic] += weight
                            gpuInfo[entry] = weight

                wordGPUInfo[docID].update(gpuInfo)

        else:
            wordGPUFlag = self.wordGPUFlag
            for t, wordID in enumerate(termIDArray):
                gpuFlag = wordGPUFlag[docID][t]

                if gpuFlag:
                    if wordID in schemaMap:
                        valueMap = schemaMap[wordID]

                        for entry in valueMap:
                            self.nz[topic] -= weight
                            self.nzw[topic][entry] -= weight

        self.wordGPUInfo = wordGPUInfo
        return #self.assignmentList

    def updateWordGPUFlag(self, docID, newTopic, random_state=None):
        d2wil, wordGPUFlag = self.docToWordIDList, self.wordGPUFlag
        tia = d2wil[docID]
        docWordGPUFlag = []

        for t, termID in enumerate(tia):
            maxProbability = self.findTopicMaxProbabilityGivenWord(termID)
            ratio = self.getTopicProbabilityGivenWord(newTopic, termID) / maxProbability
            # a = np.random.rand()
            # docWordGPUFlag.append(ratio > a)
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

        V, D = self.vocab_size, self.number_docs
        tpgw, K = self.topicProbabilityGivenWord, self.K
        d2wil = self.docToWordIDList

        for i in range(V):
            row_sum = 0.0
            for j in range(K):
                tpgw[i][j] = self.pz[j] * self.phi[j][i]
                row_sum += tpgw[i][j]

            for j in range(K):
                tpgw[i][j] = tpgw[i][j] / row_sum

        for i in range(D):
            doc_word_id = d2wil[i]
            row_sum = 0.0
            for j in range(K):
                self.pdz[i][j] = 0
                for wordID in doc_word_id:
                    self.pdz[i][j] += tpgw[wordID][j]

                row_sum += self.pdz[i][j]
            for j in range(K):
                self.pdz[i][j] = self.pdz[i][j]/row_sum

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
        topics = np.unique(self.assignmentList)

        for ii in topics:
            p_word = []
            for wordID, value in enumerate(self.nzw[ii]):
                phi_z_w = (value + beta)/(self.nz[ii] + V * beta)
                if self.id2word[wordID]:
                    p_word.append((self.id2word[wordID], phi_z_w) if self.nzw[ii][wordID] > 0 else (self.id2word[wordID], 0))
            if p_word:
                if len(p_word) >= nwords:
                    p[ii] = [x[0] for x in sorted(p_word, key=lambda x: x[1],
                             reverse=True) if x[1] > 0][:nwords]
                else:
                    p[ii] = [x[0] for x in sorted(p_word, key=lambda x: x[1],
                             reverse=True) if x[1] > 0]
        return p

    def compare_topic_terms(self, docs, wv=None):
        dim, V = self.K, self.vocab_size
        words = self.label_top_words()
        tri = np.diag(np.ones(dim))

        if not wv:
            for ii in range(dim):
                for jj in range(ii+1, dim):
                    nwords = min(len(words[ii]), len(words[jj]))
                    if nwords == 10:
                        v1 = csr_matrix(([1]*nwords, ([0]*nwords, list(map(lambda x: self.word2id[x], words[ii][:nwords])))), shape=(1, V), dtype=float)
                        v2 = csr_matrix(([1]*nwords, ([0]*nwords, list(map(lambda x: self.word2id[x], words[jj][:nwords])))), shape=(1, V), dtype=float)
                    else:
                        v1_nwords = len(words[ii])
                        v2_nwords = len(words[jj])
                        v1 = csr_matrix(([1]*v1_nwords, ([0]*v1_nwords, list(map(lambda x: self.word2id[x], words[ii])))), shape=(1, V), dtype=float)
                        v2 = csr_matrix(([1]*v2_nwords, ([0]*v2_nwords, list(map(lambda x: self.word2id[x], words[jj])))), shape=(1, V), dtype=float)
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


def buildSchema(wordVectors, docs, threshold=0.5):
    wordVectors_index = WordEmbeddingSimilarityIndex(wordVectors, threshold, 1.0)
    d = Dictionary(docs)
    similarity_matrix = SparseTermSimilarityMatrix(wordVectors_index, d)

    schema = {}
    for k in range(len(d)):
        similarWordID = similarity_matrix.matrix.nonzero()[1][
                        similarity_matrix.matrix.nonzero()[0] == k]
        similarWordID = similarWordID[similarWordID != k]
        schema.update({k: similarWordID})

    return schema
