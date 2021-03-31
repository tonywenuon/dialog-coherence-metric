# Dialog Generation Coherence Metric

## Introduction

This is a coherence metric implementation of Better Conversations by Modeling, Filtering, and Optimizing for Coherence and Diversity, EMNLP 2018. 
The paper introduces a measure of coherence as the GloVe embedding similarity between the dialogue context and the generated response.
More details please referring to the paper: [Better Conversations by Modeling, Filtering, and Optimizing for Coherence and Diversity](https://www.aclweb.org/anthology/D18-1432.pdf)

## Usage
### Access to the GloVe Embeddings.
In this implementation, we adopt pre-trained GloVe embedding: Common Crawl (42B tokens, 1.9M vocab, uncased, 300d vectors, 1.75 GB download), which can be obtained in [GloVe](https://nlp.stanford.edu/projects/glove/). After you have it, please put it under the same folder with the `coherence.py`.

For more pre-trained GloVe embeddings, visiting the [GloVe](https://nlp.stanford.edu/projects/glove/) web page and download it.
If you want to train the GloVe embedding by yourself with customised data set, please referring to the [glove-python](https://github.com/maciejkula/glove-python), which is given in the original Coherence paper.
