import random
import re
import string
from pathlib import Path
from typing import Callable

import tiktoken
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset

from llm_from_scratch.chapter_02.protocols import TokenizerProtocol


class ExpectedGPTDatasetV1(Dataset[tuple[Tensor, Tensor]]):
    def __init__(
        self,
        text: str,
        encode: Callable[[str], list[int]],
        context_size: int,
        stride: int,
    ) -> None:
        super().__init__()

        token_ids = encode(text)

        self.input_ids: list[Tensor] = [
            torch.tensor(_input_chunk(token_ids, i, context_size))
            for i in range(0, len(token_ids) - context_size, stride)
        ]

        self.target_ids: list[Tensor] = [
            torch.tensor(_target_chunk(token_ids, i, context_size))
            for i in range(0, len(token_ids) - context_size, stride)
        ]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, index: int | slice):
        return self.input_ids[index], self.target_ids[index]


def _input_chunk(token_ids: list[int], start_index: int, context_size: int):
    return token_ids[start_index : start_index + context_size]


def _target_chunk(token_ids: list[int], start_index: int, context_size: int):
    return token_ids[start_index + 1 : start_index + context_size + 1]


def test_document_loaded_correctly(document: str):
    expected = _load_document()

    assert document == expected, "Document not loaded correctly"


def test_document_tokenized_correctly(tokens: list[str]):
    expected = _tokenized_document(_load_document())

    assert tokens == expected, "Document not tokenized correctly"


def test_vocabulary_created_correctly(vocabulary: dict[str, int]):
    expected = _vocabulary_from_tokenized_document(
        _tokenized_document(_load_document())
    )

    assert vocabulary == expected, "Vocabulary not created correctly"


def test_encode_implemented_correctly(
    encode: Callable[[str, dict[str, int]], list[int]]
):
    document = _load_document()
    vocabulary = _vocabulary_from_tokenized_document(_tokenized_document(document))
    shuffled_document = _shuffled_document(document)

    assert encode(shuffled_document, vocabulary) == _expected_encode(
        shuffled_document, vocabulary
    ), "Encode not implemented correctly"


def test_decode_implemented_correctly(
    decode: Callable[[list[int], dict[str, int]], str]
):
    document = _load_document()
    vocabulary = _vocabulary_from_tokenized_document(_tokenized_document(document))
    shuffled_document = _shuffled_document(document)

    encoded_document = _expected_encode(shuffled_document, vocabulary)

    assert decode(encoded_document, vocabulary) == _expected_decode(
        encoded_document, vocabulary
    ), "Decode not implemented correctly"


def test_simple_tokenizer_v1_implemented_correctly(
    SimpleTokenizerV1: type[TokenizerProtocol],
):
    document = _load_document()
    vocabulary = _vocabulary_from_tokenized_document(_tokenized_document(document))
    shuffled_document = _shuffled_document(document)

    simple_tokenizer_v1 = SimpleTokenizerV1(vocabulary)

    encoded_document = simple_tokenizer_v1.encode(shuffled_document)

    assert encoded_document == _expected_encode(
        shuffled_document, vocabulary
    ), "SimpleTokenizerV1 not implemented correctly"

    assert simple_tokenizer_v1.decode(encoded_document) == _expected_decode(
        encoded_document, vocabulary
    ), "SimpleTokenizerV1 not implemented correctly"


def test_extended_vocabulary_created_correctly(extended_vocabulary: dict[str, int]):
    tokenized_document = _tokenized_document(_load_document())

    assert extended_vocabulary == _extended_vocabulary_from_tokenized_document(
        tokenized_document
    ), "Extended vocabulary not created correctly"


def test_simple_tokenizer_v2_implemented_correctly(
    SimpleTokenizerV2: type[TokenizerProtocol],
):
    document = _load_document()
    vocabulary = _extended_vocabulary_from_tokenized_document(
        _tokenized_document(document)
    )
    shuffled_document = _shuffled_document_with_random_strings(document)

    simple_tokenizer_v2 = SimpleTokenizerV2(vocabulary)

    encoded_document = simple_tokenizer_v2.encode(shuffled_document)

    assert encoded_document == _expected_encode_with_unk(
        shuffled_document, vocabulary
    ), "SimpleTokenizerV2 not implemented correctly"

    assert simple_tokenizer_v2.decode(encoded_document) == _expected_decode(
        encoded_document, vocabulary
    ), "SimpleTokenizerV2 not implemented correctly"


def test_input_chunk_and_target_chunk_created_correctly(
    input_chunk_and_target_chunk: Callable[
        [list[int], int, int], tuple[list[int], list[int]]
    ]
):
    token_ids = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

    assert input_chunk_and_target_chunk(token_ids, 0, 5) == (
        [0, 1, 2, 3, 4],
        [1, 2, 3, 4, 5],
    )
    assert input_chunk_and_target_chunk(token_ids, 2, 3) == ([2, 3, 4], [3, 4, 5])
    assert input_chunk_and_target_chunk(token_ids, 5, 4) == (
        [5, 6, 7, 8],
        [6, 7, 8, 9],
    )


def test_gpt_dataset_v1(
    GPTDatasetV1: type[Dataset[tuple[Tensor, Tensor]]],
):
    dataset = GPTDatasetV1(  # type: ignore
        text=_load_document(),
        encode=tiktoken.get_encoding("gpt2").encode,
        context_size=4,
        stride=1,
    )

    expected_dataset = ExpectedGPTDatasetV1(
        text=_load_document(),
        encode=tiktoken.get_encoding("gpt2").encode,
        context_size=4,
        stride=1,
    )

    assert len(dataset) == len(
        expected_dataset
    ), "GPTDatasetV1 not implemented correctly"

    assert all(
        torch.equal(
            dataset[i][0],
            expected_dataset[i][0],
        )
        and torch.equal(
            dataset[i][1],
            expected_dataset[i][1],
        )
        for i in range(len(expected_dataset))
    ), "GPTDatasetV1 not implemented correctly"


def test_simple_embedding(embedding: nn.Embedding):
    torch.manual_seed(42)
    expected = nn.Embedding(6, 3)

    assert torch.equal(
        embedding.weight, expected.weight
    ), "Simple embedding not implemented correctly"


def test_token_embedding(embedding: nn.Embedding):
    torch.manual_seed(42)
    expected = nn.Embedding(50257, 256)

    assert torch.equal(
        embedding.weight, expected.weight
    ), "Token embedding not implemented correctly"


def test_token_embeddings(token_embeddings: Tensor):
    torch.manual_seed(42)
    embedding = nn.Embedding(50257, 256)

    dataloader = _dataloader(batch_size=8, context_size=4, stride=4, shuffle=False)
    inputs, _ = next(iter(dataloader))

    expected = embedding(inputs)

    assert torch.equal(
        token_embeddings, expected
    ), "Token embeddings not implemented correctly"


def test_position_embedding(embedding: nn.Embedding):
    torch.manual_seed(42)
    expected = nn.Embedding(4, 256)

    assert torch.equal(
        embedding.weight, expected.weight
    ), "Position embedding not implemented correctly"


def test_position_embeddings(position_embeddings: Tensor):
    torch.manual_seed(42)
    embedding = nn.Embedding(4, 256)

    position_embeddings = embedding(torch.arange(4))

    assert torch.equal(
        position_embeddings, position_embeddings
    ), "Position embeddings not implemented correctly"


def test_input_embeddings(input_embeddings: Tensor):
    torch.manual_seed(42)
    token_embedding = nn.Embedding(50257, 256)

    torch.manual_seed(42)
    position_embedding = nn.Embedding(4, 256)

    dataloader = _dataloader(batch_size=8, context_size=4, stride=4, shuffle=False)
    inputs, _ = next(iter(dataloader))

    expected = token_embedding(inputs) + position_embedding(torch.arange(4))

    assert torch.equal(
        input_embeddings, expected
    ), "Input embeddings not implemented correctly"


def _expected_decode(token_ids: list[int], vocabulary: dict[str, int]) -> str:
    reversed_vocabulary = {integer: token for token, integer in vocabulary.items()}
    return " ".join([reversed_vocabulary[token_id] for token_id in token_ids])


def _expected_encode_with_unk(text: str, vocabulary: dict[str, int]) -> list[int]:
    tokenized = re.split(r'([,.:;?_!"()\']|--|\s)', text)
    return [
        vocabulary.get(token.strip(), vocabulary["<|unk|>"])
        for token in tokenized
        if token.strip()
    ]


def _expected_encode(text: str, vocabulary: dict[str, int]) -> list[int]:
    tokenized = re.split(r'([,.:;?_!"()\']|--|\s)', text)
    return [vocabulary[token.strip()] for token in tokenized if token.strip()]


def _extended_vocabulary_from_tokenized_document(tokenized_document: list[str]):
    all_tokens = [
        *sorted(list(set(tokenized_document))),
        "<|endoftext|>",
        "<|unk|>",
    ]

    return {token: integer for integer, token in enumerate(all_tokens)}


def _vocabulary_from_tokenized_document(tokenized_document: list[str]):
    return {
        token: integer for integer, token in enumerate(sorted(set(tokenized_document)))
    }


def _tokenized_document(document: str):
    splitted = re.split(r'([,.:;?_!"()\']|--|\s)', document)

    tokenized_document = [item.strip() for item in splitted if item.strip()]

    return tokenized_document


def _dataloader(
    batch_size: int = 4,
    context_size: int = 256,
    stride: int = 128,
    shuffle: bool = True,
    drop_last: bool = True,
):
    encode = tiktoken.get_encoding("gpt2").encode
    dataset = ExpectedGPTDatasetV1(
        _load_document(),
        encode,
        context_size,
        stride,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
    )

    return dataloader


def _load_document():
    return (Path(__file__).parent / "the-verdict.txt").read_text()


def _shuffled_document_with_random_strings(document: str):
    splitted = re.split(r'([,.:;?_!"()\']|--|\s)', document) + [
        _random_string(random.randint(1, 10)) for _ in range(100)
    ]
    random.shuffle(splitted)

    return " ".join(splitted)


def _random_string(length: int):
    return "".join(random.choices(string.ascii_letters, k=length))


def _shuffled_document(document: str):
    splitted = re.split(r'([,.:;?_!"()\']|--|\s)', document)
    random.shuffle(splitted)
    return " ".join(splitted)
