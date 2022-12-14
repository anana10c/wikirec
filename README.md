# WikiRec: A Wikipedia Recommender System

05-318 Human-AI Interaction Final Project (Track B)

WikiRec is a Wikipedia page recommendation system. Given a Wikipedia page, it will recommend additional pages for further reading based on network distance within Wikipedia, similarity of topic and content, and article importance/quality.

To use: First download (or train) the pretrained English embeddings from Wikipedia2Vec, pageranks from Wikidata PageRank, and mapping indices from wikimapper (see links below). Make sure to include the file paths in `MODEL_FILE`, `PAGERANK_FILE`, and `MAPPER_FILE` respectively in app.py.

To run: `python app.py`

## Requirements
Python 3.6, Bootstrap 5, NumPy, Flask

Wikipedia API: https://github.com/goldsmith/Wikipedia

Wikipedia2Vec:\
Ikuya Yamada, Akari Asai, Jin Sakuma, Hiroyuki Shindo, Hideaki Takeda, Yoshiyasu Takefuji, Yuji Matsumoto, Wikipedia2Vec: An Efficient Toolkit for Learning and Visualizing the Embeddings of Words and Entities from Wikipedia.\
https://github.com/wikipedia2vec/wikipedia2vec

Wikidata PageRank: https://danker.s3.amazonaws.com/index.html

wikimapper: https://github.com/jcklie/wikimapper
