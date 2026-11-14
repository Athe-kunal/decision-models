"""One-pass ContextCite for decoders: all ablations in packed forwards.

`context_cite.attribute` deletes sentences and re-scores every ablated
prompt. Here the ablations are packed so they share work:

- removal="reader" (FlexAttention): the document runs once and each
  ablation is a copy of the reader (the text after the last part) that
  attends only to its kept sentences. Document tokens still see removed
  sentences, so their content can leak: fastest, not equal to deletion.
- removal="full" (FlashAttention-2 varlen): a dropped sentence is hidden
  from every later token, which equals deletion. The kept tokens of all
  ablations are packed, and copies sharing a prefix compute it once (a
  prefix tree over the ablation copies).

Copy 0 keeps every sentence and gives the attributed answer. The model must
run `flash_attention.use_decoder`; `decoder.fuse` makes it ~1.7x faster.

Usage:
  model = decoder.TrieDecoder.for_inference(settings)
  flash_attention.use_decoder(model.network)
  decoder.fuse(model.network)
  result = masked.attribute(model, example, removal="full")
"""

from collections.abc import Sequence
import dataclasses
from typing import Any

import numpy as np
import torch
from torch import nn

from decision_models.citation import context_cite
from decision_models.data import schema
from decision_models.evidence import sentences as sentences_lib
from decision_models.models import decoder

REMOVALS = ("reader", "full")
Node = tuple[int, ...]


def token_owners(
    tokenizer: Any,
    example: schema.Example,
    sentences: Sequence[sentences_lib.Sentence],
) -> tuple[list[int], np.ndarray, int]:
    """The prompt's ids, each token's sentence (-1 if none), the reader start.

    owner[t] is the first sentence whose characters (separator included)
    reach token t. Tokens outside every sentence (chat template, text
    around sentences, the reader) get -1. The reader is every token from
    the first one that starts after the last part.
    """
    order = range(len(example.options))
    text, where = decoder.build_prompt(example, order, tokenizer)
    enc = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    by_char = np.full(len(text) + 1, -1)
    for i, s in enumerate(sentences):
        offset = where[s.part][0]
        by_char[offset + s.start - len(s.sep) : offset + s.end] = i
    owner = np.array(
        [
            next((o for o in by_char[a:b] if o >= 0), -1)
            for a, b in enc["offset_mapping"]
        ]
    )
    starts = np.array([a for a, _ in enc["offset_mapping"]])
    reader = int(np.searchsorted(starts, where[-1][1]))
    return enc["input_ids"], owner, reader


def _ordered_nodes(nodes: Sequence[Node]) -> list[Node]:
    """The root first, then the inner trie nodes by depth."""
    return [(), *sorted((x for x in nodes if x), key=len)]


def _read(
    logits: torch.Tensor, nodes: Sequence[Node], tokenizer: Any, k: int
) -> list[torch.Tensor]:
    """S [K] per copy from logits [n, 1 + T, V] at each copy's trie nodes."""
    suffixes, trie = decoder.candidates(tokenizer, k)
    return [
        decoder.read_trie(
            dict(zip(nodes, torch.log_softmax(row.float(), -1), strict=True)),
            suffixes,
            trie,
        )
        for row in logits
    ]


# ---------------------------------------------------------------------------
# Reader-only masking (FlexAttention).
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class ReaderPack:
    """The document once, then n copies of reader + inner trie tokens.

    All copies take the full prompt's positions (tree-attention decoding).
    Costs D + n * (R + T) tokens instead of n * P.

    Attributes:
      input_ids: [D + n * C] token ids, C = R + T.
      position_ids: [D + n * C].
      branch: [D + n * C], -1 on the document, i on copy i.
      node: [D + n * C], j on inner trie node nodes[j]'s token, -1 elsewhere.
      at: [n, 1 + T] the copy-i token whose logits give nodes[j]'s
        next-token distribution.
      nodes: The trie nodes, root first.
    """

    input_ids: torch.Tensor
    position_ids: torch.Tensor
    branch: torch.Tensor
    node: torch.Tensor
    at: torch.Tensor
    nodes: list[Node]


def pack_reader(
    ids: Sequence[int], reader: int, n: int, nodes: Sequence[Node]
) -> ReaderPack:
    """Packs n reader copies after one copy of the document."""
    nodes = _ordered_nodes(nodes)
    length = len(ids)
    doc = torch.tensor(ids[:reader])
    copy = torch.tensor([*ids[reader:], *(x[-1] for x in nodes[1:])])
    pos = torch.tensor(
        [*range(reader, length), *(length - 1 + len(x) for x in nodes[1:])]
    )
    d, r, c = reader, length - reader, len(copy)
    start = d + torch.arange(n)[:, None] * c
    copy_nodes = torch.cat(
        [torch.full((r,), -1), torch.arange(1, c - r + 1)]
    ).repeat(n)
    return ReaderPack(
        input_ids=torch.cat([doc, copy.repeat(n)]),
        position_ids=torch.cat([torch.arange(d), pos.repeat(n)]),
        branch=torch.cat(
            [torch.full((d,), -1), torch.arange(n).repeat_interleave(c)]
        ),
        node=torch.cat([torch.full((d,), -1), copy_nodes]),
        at=start + torch.cat([torch.tensor([r - 1]), torch.arange(r, c)]),
        nodes=nodes,
    )


def block_mask(
    packed: ReaderPack,
    owner: np.ndarray,
    keep: np.ndarray,
    device: torch.device | str,
) -> Any:
    """The FlexAttention mask: copy i reads only what keep[i] keeps.

    A copy never sees another copy. Within a copy, a trie token is seen only
    by its own node's descendants (tree attention). keep [n, S] bool; an all
    True row gives the full prompt, i.e. the decision.
    """
    from torch.nn.attention.flex_attention import create_block_mask

    nodes = packed.nodes
    ancestor = torch.tensor(
        [[a[: len(b)] == b for b in nodes] for a in nodes], device=device
    )
    node = packed.node.to(device)
    branch = packed.branch.to(device)
    keep_t = torch.as_tensor(keep, device=device)  # [n, S]
    n_sentences = keep_t.shape[1]
    kept = torch.cat([keep_t, keep_t.new_ones(len(keep_t), 1)], 1)  # [n, S+1]
    column = torch.full_like(branch, n_sentences)  # Column S: no sentence.
    d = int((branch < 0).sum())
    doc = torch.as_tensor(owner[:d], device=device)
    column[:d] = torch.where(doc >= 0, doc, n_sentences)

    def mask_mod(b: Any, h: Any, q: Any, k: Any) -> Any:
        del b, h
        tree = (node[k] < 0) | (
            (node[q] >= 0)
            & ancestor[node[q].clamp(min=0), node[k].clamp(min=0)]
        )
        same = (branch[q] == branch[k]) & tree
        reads = (
            (branch[q] >= 0)
            & (branch[k] < 0)
            & kept[branch[q].clamp(min=0), column[k]]
        )
        return (k <= q) & (same | reads)

    n = len(branch)
    return create_block_mask(
        mask_mod,
        B=None,
        H=None,
        Q_LEN=n,
        KV_LEN=n,
        device=device,
        _compile=True,
    )


@torch.no_grad()
def score_reader(
    network: nn.Module,
    packed: ReaderPack,
    mask: Any,
    tokenizer: Any,
    k: int,
    device: torch.device | str,
) -> list[torch.Tensor]:
    """S [K] for every copy of a ReaderPack, from one forward."""
    out = network(
        input_ids=packed.input_ids[None].to(device),
        position_ids=packed.position_ids[None].to(device),
        attention_mask=mask,
        logits_to_keep=packed.at.flatten().to(device),
        use_cache=False,
    )
    logits = out.logits[0].view(*packed.at.shape, -1)
    return _read(logits, packed.nodes, tokenizer, k)


# ---------------------------------------------------------------------------
# Full masking (FlashAttention-2 varlen, shared prefixes).
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class FullPack:
    """Full-masking copies as FA2 varlen inputs.

    Attributes:
      input_ids: [N] the new tokens of every copy, then its trie tokens.
      position_ids: [N] the tokens' positions in the full prompt.
      at: [n, 1 + T] as in ReaderPack.
      nodes: The trie nodes, root first.
      fa2: The fa2_* keyword arguments of `decoder_attention`.
      naive_tokens: Tokens that deleting sentences would run.
      tree_tokens: Tokens run here (shared prefixes once).
    """

    input_ids: torch.Tensor
    position_ids: torch.Tensor
    at: torch.Tensor
    nodes: list[Node]
    fa2: dict[str, Any]
    naive_tokens: int
    tree_tokens: int


def _cumulative(
    lengths: torch.Tensor, device: torch.device | str
) -> torch.Tensor:
    zero = torch.zeros(1, dtype=torch.long)
    return torch.cat([zero, lengths.cumsum(0)]).to(device, torch.int32)


def pack_full(
    ids: Sequence[int],
    owner: np.ndarray,
    keep: np.ndarray,
    nodes: Sequence[Node],
    device: torch.device | str,
) -> FullPack:
    """Packs full-masking copies, computing shared prefixes once.

    Copy i is the prompt's kept tokens at their original positions. Copies
    are ordered by their keep rows, so each shares its longest common prefix
    with the previous one and only its new tokens are computed. Each trie
    token attends to its copy's path and its trie ancestors.
    """
    nodes = _ordered_nodes(nodes)
    inner, length = nodes[1:], len(ids)
    ancestors = [
        [a for a, y in enumerate(inner) if x[: len(y)] == y] for x in inner
    ]
    kept = [
        np.where((owner < 0) | keep[i][np.maximum(owner, 0)])[0]
        for i in range(len(keep))
    ]
    order = sorted(range(len(keep)), key=lambda i: tuple(keep[i]), reverse=True)
    path: list[np.ndarray] = [np.zeros(0, int)] * len(keep)
    new_positions, segments = [], []
    previous, n_new = None, 0
    for i in order:
        common = 0
        if previous is not None:
            m = min(len(kept[i]), len(kept[previous]))
            diff = np.nonzero(kept[i][:m] != kept[previous][:m])[0]
            common = int(diff[0]) if len(diff) else m
        new = kept[i][common:]
        index = np.arange(n_new, n_new + len(new))
        path[i] = (
            index
            if previous is None
            else np.concatenate([path[previous][:common], index])
        )
        new_positions.append(new)
        segments.append((len(new), len(path[i]), path[i]))
        n_new += len(new)
        previous = i
    main, t = n_new, len(inner)
    node_base = {i: main + r * t for r, i in enumerate(order)}
    for i in order:
        for j in range(t):
            keys = np.concatenate(
                [path[i], node_base[i] + np.array(ancestors[j], int)]
            )
            segments.append((1, len(keys), keys))
    q_lens = torch.tensor([q for q, _, _ in segments])
    k_lens = torch.tensor([k for _, k, _ in segments])
    node_tokens = torch.tensor([x[-1] for x in inner], dtype=torch.long)
    node_positions = torch.tensor([length - 1 + len(x) for x in inner])
    ids_t = torch.tensor(list(ids))
    last = torch.tensor([int(path[i][-1]) for i in range(len(keep))])
    if t:
        nodes_at = torch.stack(
            [node_base[i] + torch.arange(t) for i in range(len(keep))]
        )
    else:
        nodes_at = torch.zeros(len(keep), 0, dtype=torch.long)
    gather = np.concatenate([g for _, _, g in segments])
    return FullPack(
        input_ids=torch.cat(
            [ids_t[torch.from_numpy(x)] for x in new_positions]
            + [node_tokens.repeat(len(keep))]
        ),
        position_ids=torch.cat(
            [torch.from_numpy(x) for x in new_positions]
            + [node_positions.repeat(len(keep))]
        ),
        at=torch.cat([last[:, None], nodes_at], 1),
        nodes=nodes,
        fa2={
            "fa2_cu_q": _cumulative(q_lens, device),
            "fa2_cu_k": _cumulative(k_lens, device),
            "fa2_max_q": int(q_lens.max()),
            "fa2_max_k": int(k_lens.max()),
            "fa2_gather": torch.from_numpy(gather).to(device),
        },
        naive_tokens=int(sum(len(x) for x in kept)),
        tree_tokens=main,
    )


@torch.no_grad()
def score_full(
    network: nn.Module,
    packed: FullPack,
    tokenizer: Any,
    k: int,
    device: torch.device | str,
) -> list[torch.Tensor]:
    """S [K] for every copy of a FullPack, from one forward."""
    out = network(
        input_ids=packed.input_ids[None].to(device),
        position_ids=packed.position_ids[None].to(device),
        logits_to_keep=packed.at.flatten().to(device),
        use_cache=False,
        **packed.fa2,
    )
    logits = out.logits[0].view(*packed.at.shape, -1)
    return _read(logits, packed.nodes, tokenizer, k)


def score_keep(
    network: nn.Module,
    tokenizer: Any,
    ids: Sequence[int],
    owner: np.ndarray,
    keep: np.ndarray,
    k: int,
    device: torch.device | str,
    budget: int = 65536,
) -> list[torch.Tensor]:
    """S [K] for every keep row [n, S] under full masking.

    Rows go in FA2 passes of at most `budget` kept tokens.
    """
    nodes = list(decoder.candidates(tokenizer, k)[1])
    kept = ((owner[None] < 0) | keep[:, np.maximum(owner, 0)]).sum(1)
    scores: list[torch.Tensor] = []
    start = 0
    while start < len(keep):
        end = start + 1
        while end < len(keep) and kept[start : end + 1].sum() <= budget:
            end += 1
        packed = pack_full(ids, owner, keep[start:end], nodes, device)
        scores += score_full(network, packed, tokenizer, k, device)
        start = end
    return scores


# ---------------------------------------------------------------------------
# ContextCite.
# ---------------------------------------------------------------------------


def attribute(
    model: decoder.TrieDecoder,
    example: schema.Example,
    num_ablations: int | None = None,
    keep_prob: float = 0.5,
    alpha: float = 0.01,
    seed: int = 0,
    removal: str = "reader",
    budget: int = 65536,
) -> context_cite.Attribution:
    """`context_cite.attribute` with every forward packed.

    Uses the same ablation masks as `context_cite.attribute`, so
    removal="full" gives its result (up to bf16 rounding).

    Args:
      model: A decoder on `flash_attention.use_decoder`.
      example: The row.
      num_ablations: Ablations (None: `context_cite.ablation_count`).
      keep_prob: Probability that an ablation keeps a sentence.
      alpha: Lasso regularisation.
      seed: Seeds the ablation masks.
      removal: "reader" (FlexAttention, leaks) or "full" (FA2, exact).
      budget: Tokens per FA2 pass (full masking).

    Raises:
      ValueError: On an unknown removal.
    """
    if removal not in REMOVALS:
        raise ValueError(f"removal is one of {REMOVALS}, not {removal!r}")
    network, tokenizer, device = model.unwrapped, model.tokenizer, model.device
    sentences = sentences_lib.split(example)
    k = len(example.options)
    n = num_ablations or context_cite.ablation_count(max(len(sentences), 1))
    masks = sentences_lib.ablation_masks(n, len(sentences), keep_prob, seed)
    ids, owner, reader = token_owners(tokenizer, example, sentences)
    keep = np.vstack([np.ones((1, len(sentences)), bool), masks])  # [1 + n, S]
    if not sentences:
        keep = keep[:1]
    if removal == "reader" and sentences:
        nodes = list(decoder.candidates(tokenizer, k)[1])
        packed = pack_reader(ids, reader, len(keep), nodes)
        mask = block_mask(packed, owner, keep, device)
        scores = score_reader(network, packed, mask, tokenizer, k, device)
    else:
        scores = score_keep(
            network, tokenizer, ids, owner, keep, k, device, budget
        )
    answer = int(scores[0].argmax())
    q = torch.softmax(scores[0].float(), 0)[answer].item()
    if not sentences:
        return context_cite.Attribution(answer, q, [], np.zeros(0))
    y = np.array([context_cite.logit_q(s, answer) for s in scores[1:]])
    weights, bias = context_cite.fit_surrogate(masks, y, alpha)
    return context_cite.Attribution(answer, q, sentences, weights, bias)
