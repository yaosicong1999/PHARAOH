"""Shared code for benchmark_new: configuration, manifests, coordinate contracts,
pyramid I/O and resource guarding. Imported by the runner, the wrappers (each in its
own method environment) and the evaluation scripts, so it depends only on the standard
library plus numpy/tifffile, imported lazily where needed."""
