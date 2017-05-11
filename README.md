# Automated Generation of Cognitive Ontology via Web Text-Mining

Definitions for concepts in the sciences of cognition are hard to pin down exactly, so we resort to looking at how scientists use them in relation to each other in practice by generating a data-driven ontology. Further, we ask whether neuroscience studies the same cognitive phenomena as cognitive science does, relating implementation to theory/algorithm. This project was presented at CogSci2017 (London). For further details, see included conference submission.

This repo contains code to scrape CognitiveAtlas (http://www.cognitiveatlas.org/) for a list of ~800 cognitive concepts, then scraping PubMed and Proceedings to the CogSci conference in the last 8 years for how frequently those terms are used, and how often pairs of them are used. Analyses look at the distribution and discrepancies for how these terms are used in cognitive science and neuroscience, and perform hierarchical clustering to automatically identify related conceptual clusters.

The core code used for this project is based on ERP_SCANR (https://github.com/TomDonoghue/ERP_SCANR).
