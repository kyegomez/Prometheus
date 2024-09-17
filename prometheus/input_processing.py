from typing import Dict, List

import tiktoken
import torch
import torch.nn as nn
from loguru import logger
from torch.nn.utils.rnn import pad_sequence

from prometheus.tokenizer import GenomeTokenizer


class TikTokenizer:
    def __init__(
        self,
        model_name: str = "o200k_base",
    ):
        """
        Initializes a TikTokenizer object.

        Args:
            model_name (str, optional): The name of the model to use for tokenization. Defaults to "gpt-4o".
        """
        try:
            self.model_name = model_name
            # self.tokenizer = tiktoken./(model_name)
            self.encoding = tiktoken.get_encoding(self.model_name)
        except Exception as e:
            raise ValueError(
                f"Failed to initialize tokenizer with model '{model_name}': {str(e)}"
            )

    def count_tokens(self, string: str) -> int:
        """
        Returns the number of tokens in a text string.

        Args:
            string (str): The input text string.

        Returns:
            int: The number of tokens in the text string.
        """
        """Returns the number of tokens in a text string."""
        num_tokens = len(self.encoding.encode(string))
        return num_tokens

    def tokenize(self, string: str) -> List[int]:
        return self.encoding.encode(string)

    def detokenize(self, tokens: List[int]) -> str:
        return self.encoding.decode(tokens)

    def batch_encode(self, text: List[str]) -> int:
        return self.encoding.encode_batch(text)


class TextEmbeddingModel(nn.Module):
    def __init__(
        self, vocab_size_text: int, text_embedding_dim: int = 128
    ):
        super(TextEmbeddingModel, self).__init__()
        self.text_embedding_dim = text_embedding_dim

        # Simple embedding layer for text data
        self.text_embedding_layer = nn.Embedding(
            vocab_size_text, text_embedding_dim
        )

    def forward(self, text_input_ids: List[int]):
        """
        Forward method to get the embeddings for text data.

        Args:
            text_input_ids (torch.Tensor): Tokenized input IDs for text data.

        Returns:
            Dict[str, torch.Tensor]: A dictionary with embeddings for text data.
        """

        # Get text embeddings
        text_embeddings = self.text_embedding_layer(
            text_input_ids
        )  # (batch_size, seq_len, text_embedding_dim)
        return text_embeddings


class GenomicEmbeddingModel(nn.Module):
    def __init__(
        self,
        vocab_size_genomic: int = 5000,
        genomic_embedding_dim: int = 128,
    ):
        super(GenomicEmbeddingModel, self).__init__()
        self.genomic_embedding_dim = genomic_embedding_dim

        # Simple embedding layer for genomic sequences
        self.genomic_embedding_layer = nn.Embedding(
            vocab_size_genomic, genomic_embedding_dim
        )

    def forward(
        self, genomic_input_ids: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Forward method to get the embeddings for genomic data.

        Args:
            genomic_input_ids (torch.Tensor): Tokenized input IDs for genomic data.

        Returns:
            Dict[str, torch.Tensor]: A dictionary with embeddings for genomic data.
        """
        # Get genomic embeddings
        genomic_embeddings = self.genomic_embedding_layer(
            genomic_input_ids
        )  # (batch_size, seq_len, genomic_embedding_dim)

        return genomic_embeddings


def embed_text(
    text: List[str],
    device: torch.device = torch.device("cpu"),
    embed_dim: int = 1028,
) -> torch.Tensor:
    """
    Tokenize and embed text data using a custom embedding layer.

    Args:
        text (List[str]): List of text input.
        tokenizer (nn.Embedding): The tokenizer that tokenizes the input into IDs.
        embedding_model (EmbeddingModel): The embedding model that contains text embedding logic.
        device (torch.device): Device on which the tensor should be loaded.

    Returns:
        torch.Tensor: A tensor containing the text embeddings.
    """
    logger.info(
        f"Tokenizing and embedding text data for {len(text)} samples."
    )

    # Tokenize the input text (assuming the tokenizer outputs integer token IDs)
    # text_input_ids = torch.tensor(
    #     [tokenizer.tokenize(t)["ids"] for t in text], dtype=torch.long
    # ).to(device)
    tokens = TikTokenizer().batch_encode(text)
    tokens_tensor = pad_sequence(
        [torch.tensor(t, dtype=torch.long) for t in tokens],
        batch_first=True,
    ).to(device)
    vocab_size = TikTokenizer().encoding.n_vocab
    print(tokens_tensor.shape)

    # Get text embeddings
    text_embeddings = TextEmbeddingModel(
        vocab_size_text=vocab_size, text_embedding_dim=embed_dim
    )(tokens_tensor)
    logger.info(f"Text embedding shape: {text_embeddings.shape}")

    logger.debug(f"Text embeddings shape: {text_embeddings.shape}")
    return text_embeddings


def embed_genomic(
    sequences: List[str],
    tokenizer: GenomeTokenizer,
    # embedding_model: EmbeddingModel,
    device: torch.device = torch.device("cpu"),
    dim: int = None,
) -> torch.Tensor:
    """
    Tokenize and embed genomic data using the GenomeTokenizer and a custom embedding layer.

    Args:
        sequences (List[str]): List of genomic sequences.
        tokenizer (GenomeTokenizer): An instance of the GenomeTokenizer to tokenize genomic sequences.
        embedding_model (EmbeddingModel): The embedding model that contains genomic embedding logic.
        device (torch.device): Device on which the tensor should be loaded.

    Returns:
        torch.Tensor: A tensor containing the genomic embeddings.
    """
    logger.info(
        f"Tokenizing and embedding genomic data for {len(sequences)} sequences."
    )

    # Tokenize the genomic sequences
    tokenized_data = tokenizer.tokenize_batch(sequences)

    # Convert the tokenized IDs into a tensor
    genomic_input_ids = [item["ids"] for item in tokenized_data]
    genomic_input_ids = torch.tensor(genomic_input_ids).to(device)
    # print(genomic_input_ids.shape)

    # Get genomic embeddings
    with torch.no_grad():
        genomic_embeddings = GenomicEmbeddingModel(
            tokenizer.vocab_size, dim
        )

    logger.debug(
        f"Genomic embeddings shape: {genomic_embeddings.shape}"
    )
    return genomic_embeddings


# # Define vocab sizes
# vocab_size_text = 10000  # Example vocab size for text
# vocab_size_genomic = 5000  # Example vocab size for genomic sequences

# Initialize the model and tokenizer
# embedding_model = EmbeddingModel(
#     vocab_size_text=vocab_size_text,
#     vocab_size_genomic=vocab_size_genomic,
# ).to(torch.device("cpu"))
# genome_tokenizer = GenomeTokenizer()

# # Sample data
# text_data = [
#     "I want a pink panda with glowing fur",
#     "An elephant-sized turtle",
# ]
# # genomic_data = ["ATCGTGAACG", "CGTTAACGTT"]

# # Get embeddings for text
# text_embeddings = embed_text(
#     text_data,
# )
# print(text_embeddings)

# # # Get embeddings for genomic data
# # genomic_embeddings = embed_genomic(
# #     genomic_data, genome_tokenizer, embedding_model
# )
