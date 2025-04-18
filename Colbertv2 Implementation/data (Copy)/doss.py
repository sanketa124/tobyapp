import json

input_path = "query_triplets.tsv"
output_path = "query_triplets.jsonl"

with open(input_path, "r") as tsv_file, open(output_path, "w") as jsonl_file:
    for line_num, line in enumerate(tsv_file):
        parts = line.strip().split('\t')
        if len(parts) != 3:
            print(f"Skipping malformed line {line_num}: {line}")
            continue
        query, pos, neg = parts

        # Output as a list
        json_obj = [query, pos, neg]

        jsonl_file.write(json.dumps(json_obj) + "\n")

print(f"✅ Fixed format saved to {output_path}")

