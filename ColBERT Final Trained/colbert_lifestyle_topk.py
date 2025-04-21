import sys
from typing import List, Tuple
from colbert import Searcher
from colbert.infra import Run, RunConfig
from datasets import load_dataset


def load_data(dataset='lifestyle', datasplit='dev', max_id=10000):
    print("[*] Loading dataset...")

    collection_dataset = load_dataset("colbertv2/lotte_passages", dataset)
    collection = [x['text'] for x in collection_dataset[datasplit + '_collection']]

    queries_dataset = load_dataset("colbertv2/lotte", dataset)
    queries = [x['query'] for x in queries_dataset['search_' + datasplit]]

    print(f"[+] Loaded {len(queries)} queries and {len(collection):,} passages")

    answer_pids = [x['answers']['answer_pids'] for x in queries_dataset['search_' + datasplit]]
    filtered_queries = [q for q, apids in zip(queries, answer_pids) if any(x < max_id for x in apids)]

    print(f"[+] Filtered down to {len(filtered_queries)} queries with in-range passage IDs")

    return collection, queries, filtered_queries


def get_top_k_results(query: str, index_path: str, collection: List[str], k: int = 5) -> List[Tuple[int, float, str]]:
    with Run().context(RunConfig(experiment='topk-query-search')):
        searcher = Searcher(index=index_path, collection=collection)
        results = searcher.search(query, k=k)

        top_k_results = [
            (doc_id, score, searcher.collection[doc_id])
            for doc_id, rank, score in zip(*results)
        ]
        return top_k_results


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python colbert_lifestyle_topk.py <filtered_query_index> <top_k>")
        sys.exit(1)

    query_index = int(sys.argv[1])
    top_k = int(sys.argv[2])

    INDEX_PATH = '/home/sanket/ColBERT-main/experiments/notebook/indexes/lifestyle.dev.2bits'
    DATASET = 'lifestyle'
    SPLIT = 'dev'
    MAX_DOCS = 10000

    collection, queries, filtered_queries = load_data(dataset=DATASET, datasplit=SPLIT, max_id=MAX_DOCS)

    if query_index >= len(filtered_queries):
        print(f"[!] Invalid query index. Max allowed: {len(filtered_queries)-1}")
        sys.exit(1)

    query = filtered_queries[query_index]
    print(f"\n[*] Running search for:\nQuery: {query}\n")

    results = get_top_k_results(query, INDEX_PATH, collection, top_k)

    print("Top-k Results:")
    for i, (doc_id, score, passage) in enumerate(results):
        print(f"[{i+1}] Doc ID: {doc_id} | Score: {score:.2f}\n\tPassage: {passage}\n")
