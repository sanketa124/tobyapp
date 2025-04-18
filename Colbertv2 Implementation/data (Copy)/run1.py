import json

input_file = "corpus.txt"
output_file = "collection.tsv"

with open(input_file, "r") as infile, open(output_file, "w", encoding="utf-8") as outfile:
    for line in infile:
        obj = json.loads(line)
        outfile.write(f"{obj['docid']}\t{obj['text']}\n")
