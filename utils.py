from pyvi import ViTokenizer, ViPosTagger
from tqdm.notebook import trange, tqdm
from typing import List, Union, Optional, Mapping, Any
from pyserini.search.lucene import LuceneSearcher, JLuceneSearcherResult
import json

def preprocess(text):
    text = text.lower()
    text = ViTokenizer.tokenize(text)
    return text

# class Context:
#     """
#     Class representing a Context to find answer from.
#     A text is unspecified with respect to it length; in principle, it
#     could be a full-length document, a paragraph-sized passage, or
#     even a short phrase.
#     Parameters
#     ----------
#     text : str
#         The context that contains potential answer.
#     metadata : Mapping[str, Any]
#         Additional metadata and other annotations.
#     score : Optional[float]
#         The score of the context. For example, the score might be the BM25 score
#         from an initial retrieval stage.
#     """

#     def __init__(self,
#                  text: str,
#                  title: Optional[str] = "",
#                  language: str = "en",
#                  metadata: Mapping[str, Any] = None,
#                  score: Optional[float] = 0):
#         self.text = text
#         self.title = title
#         self.language = language
#         if metadata is None:
#             metadata = dict()
#         self.metadata = metadata
#         self.score = score

#     def __repr__(self):
#         return str(self)

#     def __str__(self):
#         return "<Passage:{},\n score:{}>".format(self.text, self.score)


# def hits_to_contexts(hits: List[JLuceneSearcherResult], language="en", field='raw', blacklist=[]) -> List[Context]:
#     """
#         Converts hits from Pyserini into a list of texts.
#         Parameters
#         ----------
#         hits : List[JLuceneSearcherResult]
#             The hits.
#         field : str
#             Field to use.
#         language : str
#             Language of corpus
#         blacklist : List[str]
#             strings that should not contained
#         Returns
#         -------
#         List[Text]
#             List of texts.
#      """
#     contexts = []
#     for i in range(0, len(hits)):
#         hit, score = hits[i]
#         try: # the previous chinese index stores the contents as "raw", while the english index stores the json string.
#             t = json.loads(hit)["ori_contents"]
#         except:
#             t = hit
#         for s in blacklist:
#             if s in t:
#                 continue
#         metadata = {}
#         contexts.append(Context(t, language, metadata, score))
#     return contexts