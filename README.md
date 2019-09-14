# The Structure of Cognition Across Computational Cognitive Neuroscience

Definitions for concepts in the sciences of cognition are hard to pin down exactly, so we resort to looking at how scientists use them in relation to each other in practice by generating a data-driven ontology. Further, we ask whether cognitive science, cognitive neuroscience, computational neuroscience, and AI all study the same cognitive phenomena, relating implementation to computation/algorithm. This project was presented at CogSci2017 (London), and the newest results submitted to CCN2019 (Berlin). For further details, see included conference submission.

This repo contains code to scrape CognitiveAtlas (http://www.cognitiveatlas.org/) for a list of ~800 cognitive "terms", as well as scraping and preprocessing abstracts from various conference proceedings. Main result relies on hierarchical and unsupervised clustering of word embeddings learned using Word2Vec to parse the "empirical structure of cognition".

See [CCN paper](./presentations/Gao2019_CCN.pdf) for most updated version.

### Abstract
Computational Cognitive Neuroscience aims to characterize the neural computations underlying behavior. To do so, we must integrate our understanding of cognition across its different subfields: cognitive science, computational neuroscience, cognitive neuroscience, and machine learning. One key challenge is evaluating whether the structure of cognitive processes – their definitions and interrelations – in each subfield is similar. If not, how different are they and how can we measure and ameliorate those differences? To answer these questions, we mined scientific abstracts from conferences representative of subfields to learn field-specific word embeddings of cognitive concepts using Word2Vec. Vector representations are then used to generate hierarchical and 2D visualizations, forming empirical cognitive ontologies for each subfield. We find that robust ontologies, such as clusters representing language-related concepts, are automatically generated from each corpus. While differences between corpora are evident, exploratory analysis with word vectors can perform similarity queries, as well as more complex algebraic queries, e.g., “working memory” without “memory” retrieves “attention”. These results demonstrate the utility of automated text-mining and natural language processing in serving as a hypothesis-generating procedure to populate manually-maintained ontologies in cognitive science, as well as suggesting potentially overlooked research opportunities across subfields.
