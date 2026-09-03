# ELEPHANT 🐘 


## Evaluating Lexical Embeddings Parallel to Hierarchical And Neural Taxonomies

Elephant is a knowledge graph constructed with the aim of evaluating the hierarchical reasoning capabilities of neural models.

A comparison between how the lexical relations of hypernymy and hyponymy (is-a relation and its inverse) are differently described in humanly curated lexical resources and encoded by pretrained language models.

The aim of this project is to compare human lexical knowledge graphs and neural language models semantics.
To do so, we compare how hierarchical lexical meaning is represented in WordNet and how it is encoded by pretrained neural models.

---------------------------------

[Open English WordNet](https://en-word.net/) was takes as the lexical resource to investigate, and the subset animals with part of speech NOUN was selected.

The final created ELEPHANT knowledge graph reports the following numbers:
- total triples: 7,128
- total entities: 1,840
- lexical entries: 299
- synsets (modelled as `ontolex:LexicalConcept`): 295
- similarity scores: 702
- hypernymy relations: 117
- hypomymy relations: 117

------------

As neural models, we test the following:
- BERT-based:
  - [bert-base-uncased](https://huggingface.co/google-bert/bert-base-uncased)
  - [bert-base-cased](https://huggingface.co/google-bert/bert-base-cased)
  - [FacebookAI/roberta-base](https://huggingface.co/FacebookAI/roberta-base)
  - [FacebookAI/xlm-roberta-base](https://huggingface.co/FacebookAI/xlm-roberta-base)
- Sentence transformers:
  - [all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
  - [sentence-transformers/all-mpnet-base-v2](https://huggingface.co/sentence-transformers/all-mpnet-base-v2)
  - [sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2](https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2)
