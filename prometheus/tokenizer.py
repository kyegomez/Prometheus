from tokenizers import (
    Tokenizer,
    pre_tokenizers,
)
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.processors import TemplateProcessing
from loguru import logger
from typing import List


class GenomeTokenizer:
    """
    GenomeTokenizer class for tokenizing genomic sequences (DNA base pairs)
    using Byte Pair Encoding (BPE) and logging for reliable sequence processing.

    Attributes:
        vocab_size (int): Size of the vocabulary for subword tokenization.
        special_tokens (List[str]): List of special tokens for padding, start, end, etc.
        model_path (str): Path to save the trained tokenizer model.
    """

    def __init__(
        self,
        vocab_size: int = 5000,
        special_tokens: List[str] = None,
        model_path: str = "genomic_tokenizer.json",
        chunk_size: int = 1028,
    ):
        """
        Initializes the GenomeTokenizer with a Byte Pair Encoding (BPE) model.

        Args:
            vocab_size (int): Size of the vocabulary for subword tokenization (default 5000).
            special_tokens (List[str]): List of special tokens for padding, start, end, etc.
            model_path (str): Path to save the trained tokenizer model (default "genomic_tokenizer.json").
        """
        self.vocab_size = vocab_size
        self.special_tokens = special_tokens
        self.chunk_size = chunk_size
        self.special_tokens = special_tokens or [
            "[UNK]",
            "[PAD]",
            "[MASK]",
            "[START]",
            "[END]",
            "[SNP]",
            "[INS]",
            "[DEL]",
        ]
        self.model_path = model_path

        logger.info("Initializing GenomeTokenizer...")

        # Initialize BPE Tokenizer
        self.tokenizer = Tokenizer(BPE())
        logger.info(
            f"Initialized BPE tokenizer with vocab size {self.vocab_size}"
        )

        # Set pre-tokenizer for splitting sequences (whitespace pre-tokenizer)
        self.tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
        logger.info(
            "Pre-tokenizer for splitting sequences based on whitespace set."
        )

        # Set post-processor to add start and end tokens
        self.tokenizer.post_processor = TemplateProcessing(
            single="[START] $A [END]",
            pair="[START] $A $B [END]",
            special_tokens=[("[START]", 1), ("[END]", 2)],
        )
        logger.info("Post-processor for start and end tokens set.")

    def train(self, sequences: List[str]):
        """
        Trains the tokenizer on the provided genomic sequences.

        Args:
            sequences (List[str]): List of genomic sequences to train the tokenizer.
        """
        logger.info(
            f"Training tokenizer with {len(sequences)} sequences..."
        )

        trainer = BpeTrainer(
            vocab_size=self.vocab_size,
            special_tokens=self.special_tokens,
        )
        self.tokenizer.train_from_iterator(sequences, trainer=trainer)
        logger.info(
            f"Tokenizer trained with a vocabulary size of {self.vocab_size}"
        )

        # Save the trained tokenizer
        self.save_tokenizer()

    def save_tokenizer(self):
        """
        Saves the trained tokenizer to the specified model path.
        """
        self.tokenizer.save(self.model_path)
        logger.info(f"Tokenizer saved to {self.model_path}")

    def load_tokenizer(self):
        """
        Loads the tokenizer from the specified model path.
        """
        try:
            self.tokenizer = Tokenizer.from_file(self.model_path)
            logger.info(f"Tokenizer loaded from {self.model_path}")
        except FileNotFoundError:
            logger.error(
                f"Tokenizer model file not found at {self.model_path}"
            )
            raise

    def tokenize(self, sequence: str):
        """
        Tokenizes a given genomic sequence.

        Args:
            sequence (str): Genomic sequence to tokenize.

        Returns:
            Dict[str, Any]: A dictionary containing tokenized sequence and token IDs.
        """
        logger.info(f"Tokenizing sequence of length {len(sequence)}")
        encoded = self.tokenizer.encode(sequence)
        logger.debug(f"Tokenized sequence: {encoded.tokens}")
        return {"tokens": encoded.tokens, "ids": encoded.ids}

    def detokenize(self, token_ids: List[int]):
        """
        Detokenizes the given token IDs back into a genomic sequence.

        Args:
            token_ids (List[int]): List of token IDs to detokenize.

        Returns:
            str: The detokenized genomic sequence.
        """
        logger.info(
            f"Detokenizing sequence from {len(token_ids)} token IDs"
        )
        sequence = self.tokenizer.decode(token_ids)
        logger.debug(f"Detokenized sequence: {sequence}")
        return sequence

    def tokenize_batch(self, sequences: List[str]):
        """
        Tokenizes a batch of genomic sequences.

        Args:
            sequences (List[str]): List of genomic sequences to tokenize.

        Returns:
            List[Dict[str, Any]]: List of tokenized sequences with token IDs.
        """
        logger.info(
            f"Tokenizing a batch of {len(sequences)} sequences."
        )
        tokenized_batch = [self.tokenize(seq) for seq in sequences]
        return tokenized_batch

    def detokenize_batch(self, batch_ids: List[List[int]]):
        """
        Detokenizes a batch of token ID sequences.

        Args:
            batch_ids (List[List[int]]): List of token ID sequences to detokenize.

        Returns:
            List[str]: List of detokenized genomic sequences.
        """
        logger.info(
            f"Detokenizing a batch of {len(batch_ids)} token sequences."
        )
        detokenized_batch = [
            self.detokenize(ids) for ids in batch_ids
        ]
        return detokenized_batch


# # Initialize the GenomeTokenizer with a vocabulary size of 5000
# tokenizer = GenomeTokenizer(
#     vocab_size=5000, model_path="genomic_tokenizer.json"
# )

# # Sample genomic sequences
# sequences = [
#     "ATGCGTACGTTT",
#     "GGGGATCTCGATAATGCGGG",
#     "ATTGGCTCTTGA",
#     "ATGNNNCTAG",
# ]

# # Train the tokenizer
# tokenizer.train(sequences)

# # Tokenize a sequence
# tokenized_seq = tokenizer.tokenize("ATGCGTACGTTT")
# print(tokenized_seq)

# # Detokenize back to sequence
# detokenized_seq = tokenizer.detokenize(tokenized_seq["ids"])
# print(detokenized_seq)

# # Tokenize a batch of sequences
# tokenized_batch = tokenizer.tokenize_batch(sequences)
# print(tokenized_batch)

# # Detokenize a batch
# detokenized_batch = tokenizer.detokenize_batch(
#     [item["ids"] for item in tokenized_batch]
# )
# print(detokenized_batch)
