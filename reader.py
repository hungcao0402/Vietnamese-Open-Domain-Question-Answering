from transformers import pipeline
import torch.nn as nn
import numpy as np
from base import Answer

class Reader(nn.Module):
    def __init__(self, model_checkpoint='nguyenvulebinh/vi-mrc-base', device=0):
        super(Reader, self).__init__()

        self.model = pipeline('question-answering', model=model_checkpoint,
                        tokenizer=model_checkpoint, device=device, use_auth_token='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')

    def forward(self, question, contexts, ranking_scores=None):
        if ranking_scores is None:
            ranking_scores = np.ones((len(contexts),))

        best_score = 0
        all_answers = []

        for text, score in zip(contexts, ranking_scores):
            QA_input = {
                'question': question,
                'context': text.replace('_',' ')
            }
            res = self.model(QA_input)

            res["score"] = res["score"] * score
            if res['score'] > best_score:
                answer = res["answer"]
                best_score = res["score"]
        
            
            all_answers.append(Answer(
                text=res["answer"],
                score=res["score"],
                ctx_score=score,
                context=text
            ))

        return answer, all_answers