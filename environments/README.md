# Environment provenance

`molmoact-gpu1-pip-freeze.txt` was exported from GPU1's working Python 3.11
environment on November 2, 2026. Editable and local `file://` entries were
removed; `setup_molmoact_env.sh` installs the corresponding pinned source trees
explicitly. No credentials, environment variable dumps, or model files are
included.

The setup script uses `--no-deps` to restore this observed package set. This is
a runtime snapshot, not a guarantee that upstream dependency metadata can solve
to the same set. Use the GPU smoke step to validate your installation. The
original GPU1 runs used Ubuntu 22.04 and RTX 4090 GPUs with 24 GB VRAM.
