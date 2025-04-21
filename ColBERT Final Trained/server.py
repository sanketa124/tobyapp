from flask import Flask, render_template, request
from functools import lru_cache
import math
import os
from dotenv import load_dotenv

from colbert.infra import Run, RunConfig, ColBERTConfig
from colbert import Searcher
from datasets import load_dataset

load_dotenv()

# Load collection used during indexing
def load_collection(dataset='lifestyle', datasplit='dev', max_id=10000):
    collection_dataset = load_dataset("colbertv2/lotte_passages", dataset)
    return [x['text'] for x in collection_dataset[datasplit + '_collection']][:max_id]



INDEX_NAME = 'lifestyle.dev.2bits' #os.getenv("INDEX_NAME")
INDEX_ROOT = '/home/sanket/ColBERT-main/experiments/notebook/indexes'
COLLECTION = load_collection()

app = Flask(__name__)

searcher = Searcher(index=INDEX_NAME, index_root=INDEX_ROOT, collection=COLLECTION)
counter = {"api" : 0}

@lru_cache(maxsize=1000000)
def api_search_query(query, k):
    print(f"Query={query}")
    if k == None: k = 10
    k = min(int(k), 100)
    pids, ranks, scores = searcher.search(query, k=100)
    pids, ranks, scores = pids[:k], ranks[:k], scores[:k]
    passages = [searcher.collection[pid] for pid in pids]
    probs = [math.exp(score) for score in scores]
    probs = [prob / sum(probs) for prob in probs]
    topk = []
    for pid, rank, score, prob in zip(pids, ranks, scores, probs):
        text = searcher.collection[pid]            
        d = {'text': text, 'pid': pid, 'rank': rank, 'score': score, 'prob': prob}
        topk.append(d)
    topk = list(sorted(topk, key=lambda p: (-1 * p['score'], p['pid'])))
    return {"query" : query, "topk": topk}

@app.route("/api/search", methods=["GET"])
def api_search():
    if request.method == "GET":
        counter["api"] += 1
        print("API request count:", counter["api"])
        return api_search_query(request.args.get("query"), request.args.get("k"))
    else:
        return ('', 405)

if __name__ == "__main__":
    app.run("0.0.0.0", 5000) #int(os.getenv("PORT"))

