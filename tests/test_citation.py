from conftest import toy_sentences
import numpy as np
import omegaconf
import pytest
import torch

from decision_models import config
from decision_models.citation import masked
from decision_models.models import decoder
from decision_models.models import flash_attention


def test_token_owners_cover_the_sentences(trie_decoder, example) -> None:
    sents = toy_sentences(example)
    ids, owner, reader = masked.token_owners(
        trie_decoder.tokenizer, example, sents
    )
    assert len(ids) == len(owner)
    assert set(owner[owner >= 0]) == {0, 1, 2}
    assert (owner[reader:] == -1).all()
    assert ids == decoder.prompt_ids(example, trie_decoder.tokenizer)


def test_full_pack_shares_prefixes(trie_decoder, example) -> None:
    sents = toy_sentences(example)
    ids, owner, _ = masked.token_owners(trie_decoder.tokenizer, example, sents)
    keep = np.array([[1, 1, 1], [1, 1, 0], [1, 0, 1]], bool)
    nodes = list(decoder.candidates(trie_decoder.tokenizer, 3)[1])
    packed = masked.pack_full(ids, owner, keep, nodes, "cpu")
    assert packed.tree_tokens < packed.naive_tokens
    assert packed.at.shape == (3, len(nodes))


def test_reader_masking_keeping_everything_is_the_decision(
    trie_decoder, example
) -> None:
    sents = toy_sentences(example)
    tok = trie_decoder.tokenizer
    ids, owner, reader = masked.token_owners(tok, example, sents)
    expected = decoder.score_tree(trie_decoder.network, ids, tok, 3, "cpu")
    flash_attention.use_decoder(trie_decoder.network)
    try:
        keep = np.ones((2, len(sents)), bool)
        nodes = list(decoder.candidates(tok, 3)[1])
        packed = masked.pack_reader(ids, reader, len(keep), nodes)
        mask = masked.block_mask(packed, owner, keep, "cpu")
        scores = masked.score_reader(
            trie_decoder.network, packed, mask, tok, 3, "cpu"
        )
    finally:
        trie_decoder.network.set_attn_implementation("sdpa")
    for s in scores:
        torch.testing.assert_close(s, expected, atol=1e-3, rtol=1e-3)


def test_unknown_removal_fails_at_config_load() -> None:
    with pytest.raises(
        omegaconf.errors.ValidationError, match="expected one of"
    ):
        config.load(config.CiteConfig, None, ["removal=partial"])
