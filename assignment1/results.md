# 9085915255

## Implementation

- For this HW assignment, I implemented the following as per the assignment description
    - define_model_parameters()
    - init_model_parameters()
    - load_embedding()
    - copy_embedding_from_numpy
    - pad_sentences()
    - DAN Model architecture was replicated in full and parameters were intitialized in the range of -0.08, 0.08 using uniform initialization.

- Based on my experimentation, I decided to modify the following hyperparameters
    - torch seed was set to 42 for easy replication.
    - word_drop was set to 0.0 as I found this parameter really did not affect performance.
    - emb_drop was set to 0.2
    - hid_drop was set to 0.2
    - Using pretrained word embeddings yielded improvement in results.

## Experimental Results

- Baseline as per HW definition
    - SST
        - Test: 0.4122
        - Dev: 0.3951
    - IMDB
        - Dev: 0.9224

- Without Embeddings
    - SST
        - Test: 0.4303 (951/2210)
        - Dev: 0.3933 (433/1101)
    - IMDB
        - Test: 0.5000 (244/488)
        - Dev: 0.9184 (225/245)
        
- With Embeddings
    - SST
        - Uses [Glove 42B 300d Embeddings](http://nlp.stanford.edu/data/glove.42B.300d.zip)
        - Test: 0.4520 (999/2210)
        - Dev: 0.4242 (467/1101)
    - IMDB (Fasttext)
        -  Uses [Fasttext 300d 2M Embeddings](https://dl.fbaipublicfiles.com/fasttext/vectors-english/crawl-300d-2M.vec.zip)
        - Test: 0.4877 (238/488)
        - Dev: 0.9429 (231/245)
    
- Percentage Change (With Embeddings - Without Embeddings)
    - SST
        - Test: 0.4520 - 0.4122 = 3.98%
        - Dev: 0.4242 - 0.3951 = 2.91%
    - IMDB
        - Dev: 0.9429 - 0.9224 = 2.05%
    - Based on the empirical results we can state that using pretrained word embeddings increased model performance by
        - 3.98% on the SST Test Dataset
        - 2.91% on the SST Dev Dataset.
        - 2.05% on the IMDB Dev Dataset.