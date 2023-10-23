from retrieval import Pyserini_Search
from reader import Reader
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI()

retriever = Pyserini_Search(index_path='indexes/full_wiki', device='cuda:4')
reader_model = Reader(model_checkpoint='nguyenvulebinh/vi-mrc-base', device=4)

def get_answer(question):
    contexts, ranking_scores = retriever.search(question, 10, use_rerank=True)
    result, all_answers = reader_model(question, contexts, ranking_scores)

    all_answers_lst = []
    for ans in all_answers:
        all_answers_lst.append(
            { "answer": ans.text,
                "score": ans.score,
                'context': ans.context
                }
        )
    top5 = sorted(all_answers_lst, key=lambda k: k['score'], reverse=True)[:5]
    
    return result, top5

class TextRequest(BaseModel):
    input_text: str


@app.post("/predict/", response_model=dict)
def process_text(text_request: TextRequest):
    input_text = text_request.input_text
    # You can process the input_text here as needed
    output_text, top5 = get_answer(input_text)

    return {"output_text": output_text,
    'top5': top5
        }

