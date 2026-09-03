# ELEPHANT

## Evaluating Lexical Embeddings Parallel to Hierarchical And Neural Taxonomies

Elephant is a knowledge graph constructed with the aim of evaluating the hierarchical reasoning capabilities of neural models.

A comparison between how the lexical relations of hypernymy and hyponymy (is-a relation and its inverse) are differently described in humanly curated lexical resources and encoded by pretrained language models.


---------------------------------

[Open English WordNet](https://en-word.net/) was takes as the lexical resource to investigate, and the subset animals with part of speech NOUN was selected.

The final created ELEPHANT knowledge graph reports the following numbers:
- total triples: 7,128
- total entities: 1,840
- lexical entries: 299
- synsets (modelled as `ontolex:LexicalConcept`: 295
- similarity scores: 702
- hypernymy relations: 117
- hypomymy relations: 117
