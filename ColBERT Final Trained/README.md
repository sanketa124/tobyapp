
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
```

---

## ⚙️ Setup

1. **Clone the repo**:
```bash
git clone https://github.com/yourusername/colbert-search-api.git
cd colbert-search-api
```

2. **Install dependencies**:
```bash
conda create -n colbert python=3.8
conda activate colbert
pip install -r requirements.txt
```

3. **Set environment variables** in `.env`:
```env
INDEX_NAME=lifestyle.dev.2bits
PORT=5000
```

4. **Run CLI example**:
```bash
python colbert_lifestyle_topk.py --query_index 13 --top_k 5
```

5. **Start the API server**:
```bash
python server.py
```

---

## 🔎 API Usage

### Endpoint

```
GET /api/search?query=<your_query>&k=<top_k>
```

### Example

```
curl "http://localhost:5000/api/search?query=can+rabbits+get+e+coli&k=5"
```

### Response
```json
{
  "query": "can rabbits get e coli",
  "topk": [
    {
      "text": "...",
      "pid": 123,
      "rank": 1,
      "score": 13.5,
      "prob": 0.56
    },
    ...
  ]
}
```

---

## 📦 Indexing Setup (ColBERT)

If you'd like to build the index yourself (optional):

```bash
python colbert_lifestyle_topk.py --build_index
```

Make sure to keep the `INDEX_NAME` and `INDEX_ROOT` consistent with the searcher.

---

## ✨ Sample Output

```bash
Query: can rabbits get e coli

Top-k Results:
 [1] Score: 13.4 | Passage: E. coli can affect many animals, including...
 [2] Score: 11.7 | Passage: Rabbits are susceptible to digestive...
```

---

## 📬 Connect

Feel free to reach out or contribute if you’re working on neural retrieval, semantic search, or intelligent question answering!

> Built by [Sanket Anand](https://www.linkedin.com/in/sanket-anand-a35606304/)  
> #ColBERT #NeuralSearch #NLP #Flask #SemanticSearch #DenseRetrieval

---

## 📁 Source Code

🔗 Full source code and scripts can be downloaded from the links below:  
👉 [Download Index via Google Drive](https://drive.google.com/file/d/1xHSvfvSrDWC-e83Fg1ZjRPOOyhvlUrJD/view?usp=sharing)
👉 [Tree of Clarifications Implementation](https://github.com/sanketa124/tobyapp/tree/master/Tree%20of%20Clarifications%20Implementation)
