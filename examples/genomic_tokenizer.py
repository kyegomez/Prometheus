from prometheus.tokenizer import GenomeTokenizer

# Initialize the GenomeTokenizer with a vocabulary size of 5000
tokenizer = GenomeTokenizer(
    vocab_size=5000, model_path="genomic_tokenizer.json"
)

# Sample genomic sequences
sequences = [
    "ATGCGTACGTTT",
    "GGGGATCTCGATAATGCGGG",
    "ATTGGCTCTTGA",
    "ATGNNNCTAG",
]

# Train the tokenizer
tokenizer.train(sequences)

# Tokenize a sequence
tokenized_seq = tokenizer.tokenize("ATGCGTACGTTT")
print(tokenized_seq)

# Detokenize back to sequence
detokenized_seq = tokenizer.detokenize(tokenized_seq["ids"])
print(detokenized_seq)

# Tokenize a batch of sequences
tokenized_batch = tokenizer.tokenize_batch(sequences)
print(tokenized_batch)

# Detokenize a batch
detokenized_batch = tokenizer.detokenize_batch(
    [item["ids"] for item in tokenized_batch]
)
print(detokenized_batch)
