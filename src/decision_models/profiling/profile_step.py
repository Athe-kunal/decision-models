"""Profiles training steps on one GPU: profiler traces, CUDA memory snapshots.

Usage (see `make torch-profiler` which explains
every output):
  python -m decision_models.profiling.profile_step model=Qwen/Qwen3-0.6B \
      out=runs/profile/Qwen3-0.6B

A step is the decoder training step (losses.trie) without Accelerate:
`batch_size` rows, each one forward with the adapter (p_theta), one without
(p0), the loss and a backward; then gradient clipping and AdamW. Steps run in
three stages:

  1. warmup (warmup_steps): torch.compile of the loss, cuBLAS handles and the
     caching allocator's first segments, which would distort the profile;
  2. memory (memory_steps): the CUDA caching allocator records every
     allocation and free with its Python stack -> memory_snapshot.pickle;
     each phase's allocated / peak memory -> memory_phases.json;
  3. profile (profile_steps): torch.profiler records CPU ops, CUDA kernels,
     shapes, memory and stacks -> trace.json.gz, op tables, memory timeline.

The memory stage runs before the warmup when cold_memory is set, so the
snapshot also shows the first step (AdamW creating its state, compile).
"""

from collections.abc import Callable
from collections.abc import Generator
import contextlib
import dataclasses
import json
import pathlib
import pickle
import random
import time
from typing import Any

from loguru import logger
from omegaconf import MISSING
import torch
from torch import nn
from torch.autograd import DeviceType
from torch.profiler import ProfilerActivity
from torch.profiler import record_function

from decision_models import config
from decision_models.data import schema
from decision_models.data import storage
from decision_models.losses import trie
from decision_models.models import decoder

MIB = 2**20
# record_function names, in step order: this file's phases, with the ones in
# losses.trie.compute nested inside forward_and_loss.
PHASES = (
    "forward_and_loss",
    "p_theta_forward",
    "p0_forward",
    "loss_terms",
    "backward",
    "clip_grad_norm",
    "optimizer_step",
    "zero_grad",
)


@dataclasses.dataclass
class ProfileConfig:
    """What to profile.

    Attributes:
      model: Decoder Hub id or path (a fresh LoRA adapter is trained).
      data: Data folder or Hub id (the train split is read).
      out: Output folder.
      batch_size: Rows per optimizer step.
      max_tokens: Rows over this many tokens are skipped.
      warmup_steps: Untraced steps first (compile, allocator).
      memory_steps: Steps recorded by the CUDA allocator.
      profile_steps: Steps recorded by torch.profiler.
      cold_memory: Record memory from the first step, before the warmup.
      no_checkpointing: Keep every activation (compare peak memory).
      lora_rank: LoRA rank.
      lr: AdamW learning rate.
      lam: KL anchor weight.
      max_grad_norm: Gradient clipping norm.
      table_rows: Rows per profiler table.
      seed: Picks and shuffles the rows.
    """

    model: str = "Qwen/Qwen3-0.6B"
    data: str = config.DEFAULT_DATA
    out: str = MISSING
    batch_size: int = 4
    max_tokens: int = 4096
    warmup_steps: int = 3
    memory_steps: int = 1
    profile_steps: int = 1
    cold_memory: bool = False
    no_checkpointing: bool = False
    lora_rank: int = 16
    lr: float = 1e-4
    lam: float = 1.0
    max_grad_norm: float = 1.0
    table_rows: int = 40
    seed: int = config.SEED

    @property
    def steps_total(self) -> int:
        """Every step run (+ 1: the profiler's discarded warmup step)."""
        return self.warmup_steps + self.memory_steps + 1 + self.profile_steps


def pick_rows(
    train: list[schema.Example], tok: Any, args: ProfileConfig
) -> list[decoder.TrieEncoding]:
    """Encoded training rows of at most `max_tokens` tokens, seeded.

    Raises:
      ValueError: If too few rows fit.
    """
    rng = random.Random(args.seed)
    rows: list[decoder.TrieEncoding] = []
    for i in rng.sample(range(len(train)), len(train)):
        ex = decoder.encode(train[i], tok, random.Random(f"{args.seed}:{i}"))
        if len(ex.input_ids) <= args.max_tokens:
            rows.append(ex)
        if len(rows) == args.batch_size * args.steps_total:
            return rows
    raise ValueError(
        f"only {len(rows)} rows of at most {args.max_tokens} tokens"
    )


class Phases:
    """Names each phase for the profiler and, when on, measures its memory.

    torch.cuda.memory_allocated is the caching allocator's host-side count of
    live tensor bytes, so it needs no synchronize; max_memory_allocated is the
    high-water mark since the last reset_peak_memory_stats.
    """

    def __init__(self) -> None:
        self.measure = False
        self.rows: list[dict[str, Any]] = []

    @contextlib.contextmanager
    def __call__(self, name: str, row: int | None = None) -> Generator[None]:
        """Names (and when measuring, records the memory of) one phase."""
        if not self.measure:
            with record_function(name):
                yield
            return
        torch.cuda.reset_peak_memory_stats()
        before = torch.cuda.memory_allocated()
        with record_function(name):
            yield
        self.rows.append(
            {
                "phase": name,
                "row": row,
                "allocated_before_mib": before / MIB,
                "allocated_after_mib": torch.cuda.memory_allocated() / MIB,
                "peak_mib": torch.cuda.max_memory_allocated() / MIB,
                "peak_above_before_mib": (
                    torch.cuda.max_memory_allocated() - before
                )
                / MIB,
                "reserved_mib": torch.cuda.memory_reserved() / MIB,
            }
        )


def train_step(
    model: nn.Module,
    opt: torch.optim.Optimizer,
    batch: list[decoder.TrieEncoding],
    device: torch.device,
    pad: int,
    phase: Phases,
    args: ProfileConfig,
) -> float:
    """One optimizer step over `batch`; returns the mean loss."""
    total = 0.0
    for j, ex in enumerate(batch):
        with phase("forward_and_loss", j):
            terms = trie.compute(model, ex, device, pad)
            anchors = terms["mass"] + terms["out"] + terms["pos"]
            loss = terms["tree"] + args.lam * anchors
        with phase("backward", j):
            (loss / len(batch)).backward()
        total += loss.item() / len(batch)
    with phase("clip_grad_norm"):
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
    with phase("optimizer_step"):
        opt.step()
    with phase("zero_grad"):
        opt.zero_grad(set_to_none=True)
    return total


def save_tables(prof: Any, out: pathlib.Path, rows: int) -> None:
    """The profiler's aggregate tables, as text."""
    averages = prof.key_averages()
    tables = {
        "ops_by_cuda_time.txt": averages.table(
            sort_by="self_cuda_time_total", row_limit=rows
        ),
        "ops_by_cpu_time.txt": averages.table(
            sort_by="self_cpu_time_total", row_limit=rows
        ),
        "ops_by_cuda_memory.txt": averages.table(
            sort_by="self_cuda_memory_usage", row_limit=rows
        ),
        "ops_by_input_shape.txt": prof.key_averages(
            group_by_input_shape=True
        ).table(sort_by="self_cuda_time_total", row_limit=rows),
        "ops_by_stack.txt": prof.key_averages(group_by_stack_n=8).table(
            sort_by="self_cuda_time_total", row_limit=rows
        ),
    }
    for name, table in tables.items():
        (out / name).write_text(table)
    # Each record_function appears twice: the CPU range (host time from enter
    # to exit) and a GPU range (first to last kernel launched inside it). The
    # GPU range of backward is empty: autograd runs it on its own thread.
    cpu = {e.key: e for e in averages if e.device_type == DeviceType.CPU}
    gpu = {e.key: e for e in averages if e.device_type != DeviceType.CPU}
    lines = [
        f"{'phase':<20}{'calls':>7}{'cpu range ms':>14}{'gpu range ms':>14}"
    ]
    for name in ("ProfilerStep*",) + PHASES:
        if name in cpu:
            g = gpu[name].device_time_total / 1e3 if name in gpu else 0.0
            lines.append(
                f"{name:<20}{cpu[name].count:>7}"
                f"{cpu[name].cpu_time_total / 1e3:>14.1f}{g:>14.1f}"
            )
    (out / "phases.txt").write_text("\n".join(lines) + "\n")


def save_snapshot(out: pathlib.Path) -> None:
    """The allocator's history as a pickle and two offline HTML views."""
    from torch.cuda import _memory_viz

    snapshot = torch.cuda.memory._snapshot()
    with open(out / "memory_snapshot.pickle", "wb") as f:
        pickle.dump(snapshot, f)
    (out / "memory_trace.html").write_text(_memory_viz.trace_plot(snapshot))
    (out / "memory_segments.html").write_text(
        _memory_viz.segment_plot(snapshot)
    )


def main() -> None:
    """Runs the three stages and writes every output to `out`."""
    args, raw = config.from_command_line(ProfileConfig, __doc__)

    from transformers import AutoTokenizer

    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    logger.add(out / "profile.log", mode="w")
    config.save(raw, out / "config.yaml")
    device = torch.device("cuda")
    torch.manual_seed(args.seed)

    tok = AutoTokenizer.from_pretrained(args.model)
    pad = None if tok is None else tok.pad_token_id
    if tok is None or pad is None:
        raise ValueError(f"{args.model}'s tokenizer has no pad token")
    rows = pick_rows(storage.load_examples(args.data, "train"), tok, args)
    lengths = [len(ex.input_ids) for ex in rows]
    logger.info(
        "{} rows, tokens per row: min {} mean {:.0f} max {}",
        len(rows),
        min(lengths),
        sum(lengths) / len(lengths),
        max(lengths),
    )

    model = decoder.add_lora(decoder.load_base(args.model), args.lora_rank)
    model.config.use_cache = False
    if not args.no_checkpointing:
        model.enable_input_require_grads()
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
    model.to(device).train()
    weights = torch.cuda.memory_allocated()
    lora = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(lora, lr=args.lr)
    logger.info(
        "weights on GPU: {:.0f} MiB; LoRA parameters: {:,} ({:.1f} MiB)",
        weights / MIB,
        sum(p.numel() for p in lora),
        sum(p.numel() * p.element_size() for p in lora) / MIB,
    )

    phase = Phases()
    batches = iter(
        [
            rows[i : i + args.batch_size]
            for i in range(0, len(rows), args.batch_size)
        ]
    )

    def run(
        stage: str, n: int, after_step: Callable[[], Any] | None = None
    ) -> None:
        for k in range(n):
            t0 = time.time()
            loss = train_step(
                model, opt, next(batches), device, pad, phase, args
            )
            torch.cuda.synchronize()
            logger.info(
                "{} step {}/{}: loss {:.4f}, {:.2f} s, peak {:.0f} MiB",
                stage,
                k + 1,
                n,
                loss,
                time.time() - t0,
                torch.cuda.max_memory_allocated() / MIB,
            )
            if after_step:
                after_step()

    def memory_stage() -> None:
        # stacks="python": Python frames only; "all" adds C++ frames (slower).
        torch.cuda.memory._record_memory_history(
            max_entries=500_000, stacks="python"
        )
        phase.measure = True
        run("memory", args.memory_steps)
        phase.measure = False
        save_snapshot(out)
        torch.cuda.memory._record_memory_history(enabled=None)
        (out / "memory_phases.json").write_text(
            json.dumps(
                {"weights_mib": weights / MIB, "phases": phase.rows}, indent=2
            )
        )
        stats = torch.cuda.memory_stats()
        (out / "memory_stats.json").write_text(json.dumps(stats, indent=2))
        (out / "memory_summary.txt").write_text(torch.cuda.memory_summary())

    if args.cold_memory:
        memory_stage()
        run("warmup", args.warmup_steps)
    else:
        run("warmup", args.warmup_steps)
        memory_stage()

    def export(prof: Any) -> None:
        """Called by the profiler when the active steps end."""
        logger.info("writing the profiler outputs")
        prof.export_chrome_trace(str(out / "trace.json.gz"))
        save_tables(prof, out, args.table_rows)
        prof.export_stacks(str(out / "stacks_cuda.txt"), "self_cuda_time_total")
        logger.info(
            "writing memory_timeline.html (slow: replays every allocation)"
        )
        prof.export_memory_timeline(
            str(out / "memory_timeline.html"), device="cuda:0"
        )

    torch.cuda.reset_peak_memory_stats()
    with torch.profiler.profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        # One traced but discarded step (the profiler's own start-up cost),
        # then profile_steps recorded steps, once (repeat=1).
        schedule=torch.profiler.schedule(
            wait=0, warmup=1, active=args.profile_steps, repeat=1
        ),
        on_trace_ready=export,
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
        with_flops=True,
        # verbose: keep the Python stacks for ops_by_stack and export_stacks.
        experimental_config=torch.profiler._ExperimentalConfig(verbose=True),
    ) as prof:
        run("profile", 1 + args.profile_steps, after_step=prof.step)
    logger.info("done: {}", out)


if __name__ == "__main__":
    main()
