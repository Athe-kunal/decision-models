import numpy as np
import pytest
import torch

from decision_models import config
from decision_models.citation import masked
from decision_models.evidence import sentences as sentences_lib
from decision_models.models import decoder
from decision_models.models import flash_attention


@pytest.fixture(scope="module")
def model(decoder_dir) -> decoder.TrieDecoder:
    settings = config.DecoderModelConfig(id=str(decoder_dir))
    return decoder.TrieDecoder.for_inference(settings, "cpu")


def _sentences(example):
    # Fixed sentences, so the test does not need nltk's sentence model.
    text = example.parts[0].text
    out, start = [], 0
    for piece in (
        "The sky is blue today.",
        "Grass grows in the field.",
        "Cats sleep a lot.",
    ):
        begin = text.index(piece, start)
        out.append(
            sentences_lib.Sentence(
                0, begin, begin + len(piece), text[start:begin]
            )
        )
        start = begin + len(piece)
    return out


def test_token_owners_cover_the_sentences(model, example) -> None:
    sents = _sentences(example)
    ids, owner, reader = masked.token_owners(model.tokenizer, example, sents)
    assert len(ids) == len(owner)
    assert set(owner[owner >= 0]) == {0, 1, 2}
    assert (owner[reader:] == -1).all()
    assert ids == decoder.prompt_ids(example, model.tokenizer)


def test_full_pack_shares_prefixes(model, example) -> None:
    sents = _sentences(example)
    ids, owner, _ = masked.token_owners(model.tokenizer, example, sents)
    keep = np.array([[1, 1, 1], [1, 1, 0], [1, 0, 1]], bool)
    nodes = list(decoder.candidates(model.tokenizer, 3)[1])
    packed = masked.pack_full(ids, owner, keep, nodes, "cpu")
    assert packed.tree_tokens < packed.naive_tokens
    assert packed.at.shape == (3, len(nodes))


def test_reader_masking_keeping_everything_is_the_decision(
    model, example
) -> None:
    sents = _sentences(example)
    tok = model.tokenizer
    ids, owner, reader = masked.token_owners(tok, example, sents)
    expected = decoder.score_tree(model.network, ids, tok, 3, "cpu")
    flash_attention.use_decoder(model.network)
    try:
        keep = np.ones((2, len(sents)), bool)
        nodes = list(decoder.candidates(tok, 3)[1])
        packed = masked.pack_reader(ids, reader, len(keep), nodes)
        mask = masked.block_mask(packed, owner, keep, "cpu")
        scores = masked.score_reader(model.network, packed, mask, tok, 3, "cpu")
    finally:
        model.network.set_attn_implementation("sdpa")
    for s in scores:
        torch.testing.assert_close(s, expected, atol=1e-3, rtol=1e-3)


def test_unknown_removal(model, example) -> None:
    with pytest.raises(ValueError, match="removal"):
        masked.attribute(model, example, removal="partial")
