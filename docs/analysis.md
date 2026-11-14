# Analyses

Paper-side measurements live in `analysis/` at the repository root. They are not part of the installed package. Run them from the root:

```bash
python -m analysis.latency  --config configs/analysis/latency.yaml     # ms + GPU memory per decision -> runs/profile.json
python -m analysis.coupling --config configs/analysis/coupling.yaml    # option dependence -> <run>/coupling_dev.json
python -m analysis.at2      --config configs/analysis/at2.yaml         # AT2 theta for an encoder -> runs/at2_*/theta.pt
python -m analysis.citation --config configs/analysis/citation.yaml    # AT2 vs average attention vs ContextCite
```

Training profiles (torch.profiler, memory snapshots): [profiling.md](profiling.md).
