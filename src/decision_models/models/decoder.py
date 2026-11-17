"""A causal LM as a decision model: the prompt, the candidate trie, scoring.

Li & Wagle, LLM-as-Jev (arXiv 2610.02076v3, Sec. 3.2-3.4): the assistant
turn is prefilled with "Best answer: [" and option k is the token suffix
"k]". The suffixes of the K options form a trie; s_k sums the
log-probabilities along suffix k (Eq. 1). Training uses L_tree with KL
anchors to the starting model (decision_models.losses.trie).
"""

from collections.abc import Sequence
import copy
import dataclasses
import functools
import operator
import pathlib
import random
from typing import TYPE_CHECKING
from typing import Any
from typing import cast

import torch
from torch import nn

from decision_models import config
from decision_models.data import prompts
from decision_models.data import schema
from decision_models.models import base

if TYPE_CHECKING:
    from peft import PeftModel
    from transformers import PreTrainedModel
    from transformers._typing import GenerativePreTrainedModel

ANSWER_PREFIX = "Best answer: ["
# The assistant turn up to the answer, as the Qwen3 chat template renders it
# with thinking off.
_TAIL = "<|im_start|>assistant\n<think>\n\n</think>\n\n" + ANSWER_PREFIX
OFF_PATH = 3  # Off-gold-path trie nodes sampled per training row (App. I).
# Written by decision-models-merge next to merged weights: base + adapter.
MERGED_MARKER = "merged_adapter.json"

Trie = dict[tuple[int, ...], list[int]]


@dataclasses.dataclass(frozen=True)
class TrieEncoding:
    """One row, teacher-forced: the prompt followed by the gold suffix.

    Attributes:
      input_ids: Prompt + gold suffix token ids.
      gold: Index of the gold option in the shown order.
      n_options: K.
      path: (position, legal tokens, gold token) at each gold-path node.
      reply: Position whose logits predict the reply's first token.
      off_path: (node, legal tokens) of sampled off-gold-path trie nodes
        (training only).
    """

    input_ids: list[int]
    gold: int
    n_options: int
    path: list[tuple[int, list[int], int]]
    reply: int
    off_path: list[tuple[list[int], list[int]]]

    @property
    def prompt(self) -> list[int]:
        """The prompt ids, up to and including "Best answer: [".."""
        return self.input_ids[: self.path[0][0] + 1]


# ---------------------------------------------------------------------------
# Prompt and trie.
# ---------------------------------------------------------------------------


def build_prompt(
    example: schema.Example, order: Sequence[int], tokenizer: Any
) -> tuple[str, list[tuple[int, int]]]:
    """Prompt text up to "Best answer: [", and each part's character range.

    Raises:
      ValueError: If the chat template's assistant turn is not `_TAIL`.
    """
    content, where = prompts.render_example(example, order, "decision")
    text = (
        tokenizer.apply_chat_template(
            [{"role": "user", "content": content}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        + ANSWER_PREFIX
    )
    if not text.endswith(_TAIL):
        raise ValueError("the chat template's assistant turn is not _TAIL")
    start = text.index(content)
    return text, [(start + a, start + b) for a, b in where]


def suffix_ids(tokenizer: Any, k: int) -> list[list[int]]:
    """Token ids of the suffixes "1]" .. "k]" after the answer prefix."""
    head = tokenizer(_TAIL, add_special_tokens=False)["input_ids"]
    out: list[list[int]] = []
    for i in range(1, k + 1):
        full = tokenizer(_TAIL + f"{i}]", add_special_tokens=False)["input_ids"]
        if full[: len(head)] != head:
            raise ValueError(f"suffix {i}] changes the prompt's tokens")
        out.append(full[len(head) :])
    return out


def build_trie(suffixes: Sequence[Sequence[int]]) -> Trie:
    """Each trie node (a prefix of a suffix) -> its legal next tokens, sorted.

    The root is (). Complete suffixes are leaves, not nodes.

    Raises:
      ValueError: If one suffix is a prefix of another.
    """
    nodes: dict[tuple[int, ...], set[int]] = {}
    for s in suffixes:
        for t in range(len(s)):
            nodes.setdefault(tuple(s[:t]), set()).add(s[t])
    if any(tuple(s) in nodes for s in suffixes):
        raise ValueError("the suffixes are not prefix-free")
    return {n: sorted(v) for n, v in nodes.items()}


@functools.cache
def candidates(tokenizer: Any, k: int) -> tuple[list[list[int]], Trie]:
    """`suffix_ids` and their trie, once per tokenizer and option count."""
    suffixes = suffix_ids(tokenizer, k)
    return suffixes, build_trie(suffixes)


@functools.cache
def _prefix_length(tokenizer: Any) -> int:
    """Number of tokens of ANSWER_PREFIX at the end of `_TAIL`."""
    full = tokenizer(_TAIL, add_special_tokens=False)["input_ids"]
    bare = tokenizer(_TAIL[: -len(ANSWER_PREFIX)], add_special_tokens=False)
    if full[: len(bare["input_ids"])] != bare["input_ids"]:
        raise ValueError("ANSWER_PREFIX changes the tokens before it")
    return len(full) - len(bare["input_ids"])


def encode(
    example: schema.Example, tokenizer: Any, rng: random.Random | None = None
) -> TrieEncoding:
    """Encodes a row for training (with `rng`) or scoring (without).

    With `rng` the options are shuffled and OFF_PATH off-gold trie nodes are
    sampled; without it the options stay in release order.
    """
    order = list(range(len(example.options)))
    if rng is not None:
        rng.shuffle(order)
    text, _ = build_prompt(example, order, tokenizer)
    prompt = tokenizer(text, add_special_tokens=False)["input_ids"]
    suffixes, nodes = candidates(tokenizer, len(order))
    gold = order.index(example.gold)
    s = suffixes[gold]
    path = [
        (len(prompt) - 1 + t, nodes[tuple(s[:t])], s[t]) for t in range(len(s))
    ]
    off_path: list[tuple[list[int], list[int]]] = []
    if rng is not None:
        on = {tuple(s[:t]) for t in range(len(s))}
        off = [n for n in nodes if n not in on]
        off_path = [
            (list(n), nodes[n])
            for n in rng.sample(off, min(OFF_PATH, len(off)))
        ]
    return TrieEncoding(
        input_ids=prompt + s,
        gold=gold,
        n_options=len(order),
        path=path,
        reply=len(prompt) - _prefix_length(tokenizer) - 1,
        off_path=off_path,
    )


def prompt_ids(example: schema.Example, tokenizer: Any) -> list[int]:
    """The prompt token ids up to "Best answer: [", options in release order."""
    return encode(example, tokenizer).prompt


# ---------------------------------------------------------------------------
# Scoring.
# ---------------------------------------------------------------------------


def read_trie(
    logp: dict[tuple[int, ...], torch.Tensor],
    suffixes: Sequence[Sequence[int]],
    nodes: Trie,
) -> torch.Tensor:
    """S [K] from each trie node's log-probabilities logp[node] [V] (Eq. 1)."""
    return torch.stack(
        [
            functools.reduce(
                operator.add, (logp[tuple(x[:t])][x[t]] for t in range(len(x)))
            )
            for x in suffixes
        ]
    )


def length_batches(
    lengths: Sequence[int], budget: int, max_rows: int
) -> list[list[int]]:
    """Row indices sorted by length, packed so rows x longest <= budget."""
    out: list[list[int]] = []
    for i in sorted(range(len(lengths)), key=lengths.__getitem__):
        if (
            out
            and len(out[-1]) < max_rows
            and (len(out[-1]) + 1) * lengths[i] <= budget
        ):
            out[-1].append(i)
        else:
            out.append([i])
    return out


def _select_cache(cache: Any, rows: torch.Tensor) -> Any:
    """A new KV cache holding the given batch rows; `cache` is unchanged."""
    new = copy.copy(cache)
    new.layers = []
    for layer in cache.layers:
        layer = copy.copy(layer)
        for name, value in vars(layer).items():
            if isinstance(value, (dict, list)):
                setattr(layer, name, copy.copy(value))
        layer.reorder_cache(rows)
        new.layers.append(layer)
    return new


@torch.no_grad()
def score_batch(
    model: nn.Module,
    prompts_: Sequence[Sequence[int]],
    tokenizer: Any,
    ks: Sequence[int],
    device: torch.device | str,
    budget: int = 32768,
    max_rows: int = 64,
    return_mass: bool = False,
) -> list[Any]:
    """S [K] for several prompts (with return_mass: (s, root legal mass)).

    One left-padded prefill of all prompts, then the trie's deeper nodes run
    per depth on copies of their prompts' KV caches, in bounded chunks.
    """
    n = max(map(len, prompts_))
    ids = torch.full(
        (len(prompts_), n), tokenizer.pad_token_id, dtype=torch.long
    )
    mask = torch.zeros_like(ids)
    for i, p in enumerate(prompts_):
        ids[i, n - len(p) :] = torch.tensor(p)
        mask[i, n - len(p) :] = 1
    ids, mask = ids.to(device), mask.to(device)  # [B, n]
    out = model(
        input_ids=ids,
        attention_mask=mask,
        position_ids=(mask.cumsum(-1) - 1).clamp(min=0),
        use_cache=True,
        logits_to_keep=1,
    )
    root = torch.log_softmax(out.logits[:, -1].float(), -1)  # [B, V]
    logp: list[dict[tuple[int, ...], torch.Tensor]] = [{(): lp} for lp in root]
    tries = [candidates(tokenizer, k) for k in ks]
    depths: dict[int, list[tuple[int, tuple[int, ...]]]] = {}
    for i, (_, nodes) in enumerate(tries):
        for node in nodes:
            if node:
                depths.setdefault(len(node), []).append((i, node))
    length = mask.sum(-1)  # [B]
    step = max(1, min(max_rows, budget // n))
    for d, items in sorted(depths.items()):
        for c in range(0, len(items), step):
            chunk = items[c : c + step]
            rows = torch.tensor([i for i, _ in chunk], device=device)  # [M]
            o = model(
                input_ids=torch.tensor([x for _, x in chunk], device=device),
                attention_mask=torch.cat(
                    [mask[rows], mask.new_ones(len(chunk), d)], -1
                ),
                position_ids=length[rows, None]
                + torch.arange(d, device=device),
                past_key_values=_select_cache(out.past_key_values, rows),
                use_cache=True,
                logits_to_keep=1,
            )
            lps = torch.log_softmax(o.logits[:, -1].float(), -1)
            for (i, node), lp in zip(chunk, lps, strict=True):
                logp[i][node] = lp
    scores = [
        read_trie(lp, suffixes, nodes)
        for lp, (suffixes, nodes) in zip(logp, tries, strict=True)
    ]
    if not return_mass:
        return scores
    masses = [
        legal_mass(lp, nodes)
        for lp, (_, nodes) in zip(logp, tries, strict=True)
    ]
    return list(zip(scores, masses, strict=True))


def legal_mass(
    logp: dict[tuple[int, ...], torch.Tensor], nodes: Trie
) -> torch.Tensor:
    """Probability mass the root puts on the legal first tokens."""
    return logp[()][nodes[()]].exp().sum()


@functools.cache
def _tree_mask(
    tokenizer: Any, k: int
) -> tuple[list[tuple[int, ...]], torch.Tensor]:
    """The non-root trie nodes by depth, and their tree mask [q, q].

    Node i sees node j when j is a prefix of i, itself included.
    """
    _, nodes = candidates(tokenizer, k)
    inner = sorted((x for x in nodes if x), key=len)
    index = {x: i for i, x in enumerate(inner)}
    mask = torch.zeros(len(inner), len(inner), dtype=torch.bool)
    for i, x in enumerate(inner):
        for d in range(1, len(x) + 1):
            mask[i, index[x[:d]]] = True
    return inner, mask


@torch.no_grad()
def score_tree(
    model: nn.Module,
    prompt: Sequence[int],
    tokenizer: Any,
    k: int,
    device: torch.device | str,
) -> torch.Tensor:
    """S [K] for one prompt, with the whole trie read in one extra forward.

    The prompt is read once. Every non-root trie node then enters as one
    token at position len(prompt) + depth - 1, in a single forward on the
    prompt's KV cache, under a tree mask: a node attends to the prompt, its
    ancestors and itself. No cache is copied.
    """
    suffixes, nodes = candidates(tokenizer, k)
    n = len(prompt)
    out = model(
        input_ids=torch.tensor([list(prompt)], device=device),
        use_cache=True,
        logits_to_keep=1,
    )
    logp: dict[tuple[int, ...], torch.Tensor] = {
        (): torch.log_softmax(out.logits[0, -1].float(), -1)
    }
    inner, tree = _tree_mask(tokenizer, k)
    if inner:
        q = len(inner)
        mask = torch.cat([torch.ones(q, n, dtype=torch.bool), tree], 1)
        o = model(
            input_ids=torch.tensor([[x[-1] for x in inner]], device=device),
            attention_mask=mask.to(device)[None, None],
            position_ids=torch.tensor(
                [[n + len(x) - 1 for x in inner]], device=device
            ),
            past_key_values=out.past_key_values,
            use_cache=True,
        )
        rows = torch.log_softmax(o.logits[0].float(), -1)
        for x, row in zip(inner, rows, strict=True):
            logp[x] = row
    return read_trie(logp, suffixes, nodes)


def fuse(model: "PreTrainedModel | PeftModel") -> None:
    """Compiles each decoder layer (dynamic shapes) so its elementwise ops fuse.

    Eager RMSNorm, RoPE, SwiGLU and residual adds are about half of a packed
    ContextCite forward's GPU time (memory-bound kernels); fusing them makes
    it ~1.7x faster. Fused kernels round bf16 less often, so outputs change
    by no more than eager's own error against fp32. The first call compiles.
    """
    base_model = getattr(model, "model", model)
    for layer in cast(nn.ModuleList, base_model.layers):
        layer.compile(dynamic=True)


def batched_logits(
    model: nn.Module,
    seqs: Sequence[Sequence[int]],
    keep: Sequence[int],
    device: torch.device | str,
    pad: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Logits at positions `keep` of the first sequence, and every last logit.

    Sequences are left-padded. Returns [len(keep), V] and [S, V] in fp32.
    """
    n = max(len(x) for x in seqs)
    ids = torch.full((len(seqs), n), pad, dtype=torch.long)
    mask = torch.zeros((len(seqs), n), dtype=torch.long)
    for i, x in enumerate(seqs):
        ids[i, n - len(x) :] = torch.tensor(x)
        mask[i, n - len(x) :] = 1
    ids, mask = ids.to(device), mask.to(device)
    positions = (mask.cumsum(-1) - 1).clamp(min=0)
    at = torch.tensor(
        [n - len(seqs[0]) + p for p in keep] + [n - 1], device=device
    )
    logits = model(
        input_ids=ids,
        attention_mask=mask,
        position_ids=positions,
        use_cache=False,
        logits_to_keep=at,
    ).logits.float()
    return logits[0, :-1], logits[:, -1]


# ---------------------------------------------------------------------------
# Loading.
# ---------------------------------------------------------------------------


def load_base(
    model_id: str, dtype: torch.dtype = torch.bfloat16
) -> "PreTrainedModel":
    """The base LLM on the CPU (Qwen3.5 uses its multimodal class).

    Args:
      model_id: Hugging Face id or local path.
      dtype: bf16 by default; fp32 for full fine-tuning, since bf16 weights
        round away AdamW updates of lr ~ 1e-5.
    """
    import transformers

    cls = (
        transformers.Qwen3_5ForConditionalGeneration
        if "Qwen3.5" in model_id
        else transformers.AutoModelForCausalLM
    )
    return cls.from_pretrained(
        model_id, dtype=dtype, attn_implementation="sdpa"
    )


def add_lora(model: "PreTrainedModel", rank: int = 16) -> "PeftModel":
    """A fresh LoRA adapter on the decoder layers' linear modules."""
    import peft

    targets = [
        name
        for name, module in model.named_modules()
        if isinstance(module, nn.Linear)
        and ".layers." in name
        and "visual" not in name
    ]
    return peft.get_peft_model(
        model,
        peft.LoraConfig(
            r=rank,
            lora_alpha=2 * rank,
            lora_dropout=0.0,
            target_modules=targets,
            bias="none",
            task_type="CAUSAL_LM",
        ),
    )


class TrieDecoder(base.DecisionModel):
    """A causal LM that answers by the option-number trie.

    Attributes:
      reference: p0 for the KL anchors in full fine-tuning (a frozen copy of
        the base model); None for LoRA, whose p0 is the adapter disabled.
      settings: The model config (batching and fine-tuning mode).
      objective: The loss config (training only).
    """

    row_order = "decoder"

    def __init__(
        self,
        network: nn.Module,
        tokenizer: Any,
        device: str | torch.device,
        settings: config.DecoderModelConfig,
        objective: config.TrieObjectiveConfig | None = None,
        reference: nn.Module | None = None,
    ) -> None:
        super().__init__(network, tokenizer, device)
        self.settings = settings
        self.objective = objective or config.TrieObjectiveConfig()
        self.reference = reference

    @property
    def unwrapped(self) -> "PreTrainedModel | PeftModel":
        """The causal LM (with its LoRA adapter when training LoRA)."""
        return cast("PreTrainedModel | PeftModel", super().unwrapped)

    @classmethod
    def for_training(
        cls, cfg: config.DecoderTrainConfig, device: torch.device
    ) -> "TrieDecoder":
        """The base model prepared for LoRA or full fine-tuning."""
        import transformers

        settings, objective = cfg.model, cfg.objective
        tokenizer = transformers.AutoTokenizer.from_pretrained(settings.id)
        reference = None
        if settings.finetune == "full":
            network = load_base(settings.id, dtype=torch.float32)
            reference = load_base(settings.id).to(device).eval()
            reference.config.use_cache = False
            reference.requires_grad_(False)
        else:
            network = load_base(settings.id)
        network.config.use_cache = False
        if settings.finetune == "lora":
            network = add_lora(network, settings.lora_rank)
            network.enable_input_require_grads()
        if settings.gradient_checkpointing:
            network.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
        return cls(network, tokenizer, device, settings, objective, reference)

    @classmethod
    def for_inference(
        cls,
        path: str,
        adapter: str | None = None,
        device: str | torch.device | None = None,
        merge: bool = True,
    ) -> "TrieDecoder":
        """A model for scoring: `path`, with a LoRA `adapter` if given.

        Args:
          path: Hub id or local path of the (base or merged) model.
          adapter: A saved LoRA adapter on top of `path`.
          device: Where to put it (default: base.default_device()).
          merge: Merge the adapter into the weights.

        Raises:
          ValueError: If `path` already has an adapter merged in (written by
            decision-models-merge) and `adapter` is given again.
        """
        import peft
        import transformers

        marker = pathlib.Path(path) / MERGED_MARKER
        if adapter is not None and marker.exists():
            raise ValueError(
                f"{path} already has an adapter merged in ({marker}); drop "
                f"adapter={adapter} or load the base model"
            )
        device = device or base.default_device()
        tokenizer = transformers.AutoTokenizer.from_pretrained(path)
        network = load_base(path).to(device)
        if adapter is not None:
            network = peft.PeftModel.from_pretrained(network, adapter)
            if merge:
                network = network.merge_and_unload()
        settings = config.DecoderModelConfig(id=path)
        return cls(network.eval(), tokenizer, device, settings)

    @property
    def checkpoint_name(self) -> str:
        """The checkpoint folder: "adapter" (LoRA) or "model" (full)."""
        return "adapter" if self.settings.finetune == "lora" else "model"

    def encode(
        self, example: schema.Example, rng: random.Random | None = None
    ) -> TrieEncoding:
        """`encode` with this model's tokenizer."""
        return encode(example, self.tokenizer, rng)

    def num_tokens(self, example: schema.Example) -> int:
        """Prompt + gold suffix tokens."""
        return len(self.encode(example).input_ids)

    def score(self, examples: Sequence[schema.Example]) -> list[torch.Tensor]:
        """S [K] per example, in length-sorted padded batches."""
        return [s for s, _ in self.score_with_mass(examples)]

    def score_with_mass(
        self, examples: Sequence[schema.Example]
    ) -> list[tuple[torch.Tensor, float | None]]:
        """(s [K], root legal mass) per example.

        The legal mass is the probability the model puts on a valid first
        answer token; the KL anchor L_mass keeps it near the base model's.
        """
        model = self.unwrapped
        fp32 = next(model.parameters()).dtype == torch.float32
        autocast = torch.autocast(
            self.device.type, dtype=torch.bfloat16, enabled=fp32
        )
        prompts_ = [prompt_ids(e, self.tokenizer) for e in examples]
        out: list[tuple[torch.Tensor, float | None]] = [
            (torch.empty(0), 0.0)
        ] * len(examples)
        budget, rows = self.settings.batch_tokens, self.settings.batch_rows
        progress = base.Progress(len(examples))
        with autocast, torch.no_grad():
            for batch in length_batches(list(map(len, prompts_)), budget, rows):
                scored = score_batch(
                    model,
                    [prompts_[i] for i in batch],
                    self.tokenizer,
                    [len(examples[i].options) for i in batch],
                    self.device,
                    budget,
                    rows,
                    return_mass=True,
                )
                for i, (s, mass) in zip(batch, scored, strict=True):
                    out[i] = (s.cpu(), float(mass))
                progress.update(len(batch))
        return out

    @torch.no_grad()
    def generate_answer(
        self, example: schema.Example
    ) -> tuple[str, int | None]:
        """Unguided greedy text after "Best answer: [", and the option it names.

        Generation stops at "]" (or the longest suffix). Returns the text and
        the 0-based option it names, or None if it is not a valid "k]".
        """
        import re

        tok = self.tokenizer
        prompt = prompt_ids(example, tok)
        k = len(example.options)
        longest = max(map(len, candidates(tok, k)[0]))
        stop = [tok.convert_tokens_to_ids("<|im_end|>"), tok.eos_token_id]
        # Causal LMs generate (GenerationMixin); PreTrainedModel does not.
        lm = cast("GenerativePreTrainedModel", self.unwrapped)
        ids = lm.generate(
            torch.tensor([prompt], device=self.device),
            max_new_tokens=longest,
            do_sample=False,
            eos_token_id=[t for t in stop if t is not None],
            stop_strings=["]"],
            tokenizer=tok,
            pad_token_id=tok.pad_token_id,
            use_cache=True,
        )[0, len(prompt) :].tolist()
        text = tok.decode([t for t in ids if t not in stop])
        match = re.match(r"(\d+)\]", text)
        if match and 1 <= int(match.group(1)) <= k:
            return text, int(match.group(1)) - 1
        return text, None

    def loss(
        self, example: schema.Example, rng: random.Random
    ) -> base.LossOutput:
        """L_tree + lam * (chosen anchors); logs all four terms."""
        from decision_models.losses import trie

        terms = trie.compute(
            self.network,
            self.encode(example, rng),
            self.device,
            self.tokenizer.pad_token_id,
            self.reference,
        )
        total = terms["tree"] + self.objective.lam * sum(
            (terms[k] for k in self.objective.anchors),
            start=torch.zeros((), device=self.device),
        )
        return base.LossOutput(total, {k: v.detach() for k, v in terms.items()})

    def save(self, path: pathlib.Path, accelerator: Any) -> None:
        """The LoRA adapter, or (full fine-tuning) a bf16 model + tokenizer."""
        if self.settings.finetune == "lora":
            from torch.distributed.tensor import DTensor

            network = accelerator.unwrap_model(self.network)
            state = {
                k: (v.full_tensor() if isinstance(v, DTensor) else v).cpu()
                for k, v in network.state_dict().items()
                if "lora_" in k
            }
            network.save_pretrained(
                path,
                state_dict=state,
                is_main_process=accelerator.is_main_process,
            )
            return
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            network = accelerator.unwrap_model(self.network)
            # A bf16 copy: casting the model would round its fp32 weights.
            state = {
                k: v.detach().to(torch.bfloat16)
                for k, v in network.state_dict().items()
            }
            network.save_pretrained(path, state_dict=state)
            self.tokenizer.save_pretrained(path)
