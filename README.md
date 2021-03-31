# Dialog Generation Coherence Metric

## Introduction

This is a coherence metric implementation of Better Conversations by Modeling, Filtering, and Optimizing for Coherence and Diversity, EMNLP 2018. 
The paper introduces a measure of coherence as the GloVe embedding similarity between the dialogue context and the generated response.
More details please referring to the paper: [Better Conversations by Modeling, Filtering, and Optimizing for Coherence and Diversity](https://www.aclweb.org/anthology/D18-1432.pdf)

## Usage
We contain two embeddings instantiations: GloVe and Word2Vec. If you report results from this repo, it would be better to include the embeddings version you used.
### Access to the GloVe Embeddings.
In this implementation, we adopt pre-trained GloVe embedding: Common Crawl (42B tokens, 1.9M vocab, uncased, 300d vectors, 1.75 GB download), which can be obtained in [GloVe](https://nlp.stanford.edu/projects/glove/). After you have it, please put it under the same folder with the `coherence.py`.

For more pre-trained GloVe embeddings, visiting the [GloVe](https://nlp.stanford.edu/projects/glove/) web page and download it.
If you want to train the GloVe embedding by yourself with customised data set, please referring to the [glove-python](https://github.com/maciejkula/glove-python), which is given in the original Coherence paper.

### Access to the Word2Vec Embeddings.
This is mirroring the data from the official [word2vec website](https://code.google.com/archive/p/word2vec/):

[GoogleNews-vectors-negative300.bin.gz](https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit). 

One can download it and put it under the same folder with the `coherence.py` as well.

### Quick Start

``` python
    emb_type = 'glove'
    emb_path = './glove.42B.300d.txt'
    coh = Coherence(emb_type, emb_path)
    s1 = 'I like basketable'
    s2 = 'I love footable'
    score = coh.sentence_coherence_score(s1, s2)
    print('GloVe single sentence score:', score)
    score = coh.corpus_coherence_score(ref_path, hyp_path)
    print('GloVe corpus level score:', score)

    emb_type = 'word2vec'
    emb_path = './GoogleNews-vectors-negative300.bin'
    coh = Coherence(emb_type, emb_path)
    score = coh.sentence_coherence_score(s1, s2)
    print('Word2Vec single sentence score:', score)
    score = coh.corpus_coherence_score(ref_path, hyp_path)
    print('Word2Vec corpus level score:', score)
```

