# 🔍 ColBERTv2-Powered Semantic Search API

A lightweight and powerful search engine built on [ColBERTv2](https://github.com/stanford-futuredata/ColBERT) for fast, semantic retrieval over domain-specific text collections.

This project includes:
- A CLI utility for top-k retrieval (`colbert_lifestyle_topk.py`)
- A Flask API server (`server.py`) with real-time querying
- Dataset: [LoTTE - Lifestyle](https://huggingface.co/datasets/colbertv2/lotte)

---

## 🚀 Features

✅ End-to-end retrieval pipeline using ColBERTv2  
✅ Dataset filtering to restrict queries to valid passage IDs  
✅ Top-k search with ranks, scores, and relevance probabilities  
✅ Fast in-memory caching using `lru_cache`  
✅ Deployable REST API  
✅ Pre-indexed on top 10,000 passages from LoTTE lifestyle corpus  

---

## 🗂 Project Structure

```bash
.
├── colbert_lifestyle_topk.py     # Script for indexing + top-k querying
├── server.py                     # Flask API server
├── .env                          # Environment variables (INDEX_NAME, PORT, etc.)
└── README.md
