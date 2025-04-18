import csv

input_path = 'corpus.tsv'      # Your original file
output_path = 'corpus_cleaned.tsv'  # Output file with UTF-8 and filtered rows

with open(input_path, 'r', encoding='utf-8', errors='ignore') as infile, \
     open(output_path, 'w', encoding='utf-8', newline='') as outfile:

    reader = csv.reader(infile, delimiter='\t')
    writer = csv.writer(outfile, delimiter='\t')

    for row in reader:
        if not row:
            continue
        try:
            # Check if the first column is a number
            int(row[0])
            writer.writerow(row)
        except (ValueError, IndexError):
            # Skip rows where the first column is not a number
            continue

print(f"Cleaned file saved as {output_path}")
