"""
Copyright:
    Copyright 2021 by tonywenuon.
License:
    Apach License 2.0, see LICENSE for details.
"""
import sys
from nltk.corpus import stopwords
import numpy as np
from gensim.models import KeyedVectors

class Coherence(object):
    def __init__(
            self,
            emb_type: str,
            emb_path: str):
        self.stop_words = set(stopwords.words('english')) 
        self.dic_word2vec = self._read_embeddings(emb_type, emb_path)

    def _read_raw_data(self, path):
        """Return sentences."""
        lines = []
        with open(path) as f:
            for line in f:
                lines.append(line.strip())
        return lines

    def _read_embeddings(self, emb_type, emb_path):
        """Return word embeddings dict."""
        print('[!] Loading word embeddings')
        dic = dict()
        if emb_type == 'glove':
            with open(emb_path, 'r') as file1:
                for line in file1.readlines():
                    row = line.strip().split(' ')
                    emb = np.asarray(row[1:]).astype(np.float32)
                    dic[row[0]] = emb
        elif emb_type == 'word2vec':
            dic = KeyedVectors.load_word2vec_format(emb_path, binary=True)
        #print('[!] Embedding size: ', len(dic))
        assert 'dog' in dic 
        print('[!] Load the embedding over')
        return dic

    def _get_vector_of_sentene(self, sentence):
        """Return contains word vector list."""
        remove_stop_word_sentence = [w.lower() for w in sentence.split(' ') if not w.lower() in self.stop_words] 
        vectors = [] 
        for w in remove_stop_word_sentence:
            if w in self.dic_word2vec:
                vectors.append(self.dic_word2vec[w])
        vectors = np.asarray(vectors)
        return np.sum(vectors, axis=0)

    def _calc_cosine_sim(self, vectors1, vectors2):
        """Calculate cosine similarity."""
        vectors1 /= np.linalg.norm(vectors1, axis=-1, keepdims=True)
        vectors2 /= np.linalg.norm(vectors2, axis=-1, keepdims=True)
        return np.dot(vectors1, vectors2.T)

    def sentence_coherence_score(
            self,
            reference: str,
            hypothesis: str) -> float:
        """
        Args:
            reference (str): reference sentence.
            hypothesis: (str): hypothesis sentence.

        Return:
            float: sentence cosine similarity score

        """
        emb_ref = self._get_vector_of_sentene(reference)
        emb_hyp = self._get_vector_of_sentene(hypothesis)
        return self._calc_cosine_sim(emb_ref, emb_hyp)

    def corpus_coherence_score(
            self,
            ref_path: str,
            hyp_path: str) -> float:
        """
        Args:
            ref_path(str): reference file path, i.e. the ground truth responses file path.
            hyp_path(str): hypothesis file path, i.e. the generated responses file path.

        Return:
            float: corpus level coherence score

        """
        ref_list = self._read_raw_data(ref_path)
        hyp_list = self._read_raw_data(hyp_path)
        assert len(ref_list) == len(hyp_list), 'The number of reference sentences should be the same with that of hypothesis sentences'
        scores = []
        for ref, hyp in zip(ref_list, hyp_list):
            score = self.sentence_coherence_score(ref, hyp)
            scores.append(score)
        scores = np.asarray(scores)
        return np.mean(scores)

if __name__ == '__main__':
    assert len(sys.argv) == 3, 'Please provide a reference file and a hypothesis file.'
    ref_path = sys.argv[1]
    hyp_path = sys.argv[2]

    emb_type = 'glove'
    emb_path = './glove.42B.300d.txt'
    coh = Coherence(emb_type, emb_path)
    s1 = 'I like basketable'
    s2 = 'I love footable'
    score = coh.sentence_coherence_score(s1, s2)
    print('single sentence score:', score)

    score = coh.corpus_coherence_score(ref_path, hyp_path)
    print('GloVe corpus level score:', score)

    emb_type = 'word2vec'
    emb_path = './GoogleNews-vectors-negative300.bin'
    coh = Coherence(emb_type, emb_path)
    score = coh.corpus_coherence_score(ref_path, hyp_path)
    print('Word2Vec corpus level score:', score)


