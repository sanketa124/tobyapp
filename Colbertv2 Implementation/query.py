import json
import uuid

# Load your ASQA-style JSON file
with open('asqa_dataset.json') as f:
    data = json.load(f)

queries = []
corpus = {}
query_ids = {}

query_counter = 0
doc_counter = 0

# Assuming 'dev' key like: {"dev": {"some_id": { ... }}}
for entry_id, entry in data['dev'].items():
    for qa in entry["qa_pairs"]:
        question = qa["question"].strip()
        
        # Generate a unique query ID
        query_id = f"q{query_counter}"
        queries.append((query_id, question))
        query_ids[question] = query_id
        query_counter += 1

        # Add context if it's not "No context provided"
        context = qa["context"].strip()
        if context and context.lower() != "no context provided":
            doc_id = f"d{doc_counter}"
            corpus[doc_id] = context
            doc_counter += 1

    # Include extra wikipages as documents
    for page in entry.get("wikipages", []):
        doc_id = f"d{doc_counter}"
        page_title = page.get("title", "")
        page_url = page.get("url", "")
        full_text = f"{page_title}. {page_url}".strip()
        corpus[doc_id] = full_text
        doc_counter += 1

    # Also consider annotation knowledge as context
    for ann in entry.get("annotations", []):
        for know in ann.get("knowledge", []):
            doc_id = f"d{doc_counter}"
            context_text = know.get("content", "")
            if context_text:
                corpus[doc_id] = context_text
                doc_counter += 1

# Save queries.tsv
with open('queries.tsv', 'w', encoding='utf-8') as f:
    for qid, qtext in queries:
        f.write(f"{qid}\t{qtext}\n")

# Save corpus.txt
with open('corpus.txt', 'w', encoding='utf-8') as f:
    for doc_id, doc_text in corpus.items():
        f.write(f"{doc_id}\t{doc_text}\n")

print(f"Generated {len(queries)} queries and {len(corpus)} documents.")
