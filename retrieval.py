from pyserini.search.lucene import LuceneSearcher
from utils import preprocess
import json
from model import PairwiseModel
import torch
import numpy as np

'''
Example usage:

from retrieval import Pyserini_Search
retriever = Pyserini_Search(index_path='indexes/doc2query', device='cuda')
contexst = retriever.search("Cộng hòa Weimar chính thức thay thế đế quốc Đức kể từ sau sự kiện nào?", 10)
'''

class Pyserini_Search():
    def __init__(self, index_path='indexes/doc2query', path_pretrain_ranker4='rerank.bin', device='cuda'):
        self.searcher = LuceneSearcher(index_path)
        self.num_candidates_samples=100

        self.ranker = PairwiseModel("nguyenvulebinh/vi-mrc-base", device=device)
        self.ranker.load_state_dict(torch.load(path_pretrain_ranker4, map_location=device))
        self.ranker.eval()
        self.ranker.to(device)

    def search(self, question: str, topk: int=10, use_rerank=True):
        '''
        return: list contexts (ex: ['Pham van dong ...', 'chu tich nuoc ..',...])
        '''
        question = preprocess(question)
        hits = self.searcher.search(question, k=topk)

        contexts = [json.loads(hits[j].raw)['ori_contents'] for j in range(min(len(hits),topk))]

        ranking_scores = np.ones((len(contexts),))
        if use_rerank:
            # try:
                question = question.replace('_', ' ')
                contexts, ranking_scores = self.rerank(question, contexts)
            # except:
            #     print(f'question \"{question}\" not have enough retrieve context!!! (len(context)={len(contexts)})')
        return contexts, ranking_scores

    def rerank(self, question, contexts, top_rerank=10):
        ranking_preds = self.ranker.stage1_ranking(question, contexts[:top_rerank])
        ranking_scores = ranking_preds 

        #Question answering
        best_idxs = np.argsort(ranking_scores)[-10:]
        ranking_scores = np.array(ranking_scores)[best_idxs]
        texts = np.array(contexts)[best_idxs][::-1]

        contexts[:top_rerank] = texts
        return contexts, ranking_scores[::-1]
    
    def sample(self, query_str, relevant_docs, topk, max_query_len = 512):
        """
        Samples from a list of candidates using BM25.
        
        If the samples match the relevant doc, 
        then removes it and re-samples randomly.

        Args:
            query_str: the str of the query to be used for BM25
            relevant_docs: the str of the relevant documents, to avoid sampling them as negative sample.
            max_query_len: int containing the maximum number of characters to use as input. (Very long queries will raise a maxClauseCount from anserini.)                

        Returns:
            A triplet containing the list of negative samples, 
            whether the method had retrieved the relevant doc and 
            if yes its rank in the list.
        """
        #Some long queryies exceeds the maxClauseCount from anserini, so we cut from right to left.
        query_str = query_str[-max_query_len:]
        query_str = self.preprocess(query_str)
        sampled_initial = [ json.loads(hit.raw)['contents'] for hit in self.searcher.search(query_str, k=topk)]

        sampled = []
        count=0
        for i, d in enumerate(sampled_initial):
            if d == relevant_docs:
                continue
            else:
                sampled.append(d)
                count+=1
                if count==(topk-1):
                    break
        return sampled, relevant_docs
