import json
import random
import os

# Load ASQA dataset
with open("ASQA.json", "r") as f:
    data = json.load(f)["dev"]

triplets = []
triplets.append(['query', 'positive', 'negative'])

# Gather all possible negative contexts from other QA pairs
all_negative_contexts = []

for qid, item in data.items():
    for pair in item["qa_pairs"]:
        ctx = pair.get("context")
        if ctx and ctx != "No context provided":
            all_negative_contexts.append(ctx)
    for annotation in item.get("annotations", []):
        for k in annotation.get("knowledge", []):
            content = k.get("content")
            if content:
                all_negative_contexts.append(content)

# Process each QA pair
for qid, item in data.items():
    annotations = item.get("annotations", [])
    long_answers = [a["long_answer"] for a in annotations if "long_answer" in a]

    for pair in item["qa_pairs"]:
        question = pair["question"]
        context = pair.get("context", "")

        # Prefer annotation long_answer if available
        if long_answers:
            positive = random.choice(long_answers)
        elif context and context != "No context provided":
            positive = context
        else:
            continue  # Skip if no usable positive

        # Sample a negative context that's not from the same question
        negative_candidates = [ctx for ctx in all_negative_contexts if ctx != context and ctx != positive]
        if not negative_candidates:
            continue

        negative = random.choice(negative_candidates)

        triplets.append((question.strip(), positive.strip(), negative.strip()))

# Write to TSV
os.makedirs("colbert-data", exist_ok=True)
with open("colbert-data/asqa_colbert.tsv", "w", encoding="utf-8") as f:
    for q, p, n in triplets:
        f.write(f"{q}\t{p}\t{n}\n")

print(f"Written {len(triplets)} training samples to colbert-data/asqa_colbert.tsv")