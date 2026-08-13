# syntax=docker/dockerfile:1
#
# Two build targets:
#   base (default) -- core simulation + tests, matches pyproject.toml's core
#     `dependencies` + `dev` extra (numpy/scipy/matplotlib/pyyaml/pytest). Small,
#     builds fast.
#   rl -- also installs the optional `rl` extra (gymnasium + stable-baselines3, which
#     pulls in torch, a multi-GB dependency) -- only needed for core/rl/train.py and
#     the RL comparison scripts. Build with `--target rl` to get it; skip it entirely
#     if you only care about parking/highway mode, same "opt-in, not core" choice
#     pyproject.toml itself makes (see DESIGN.md section 10's "learned parking
#     policy" entry).
#
# Python 3.12, not the 3.14 this was developed against -- pyproject.toml only
# requires >=3.10, and 3.12 is a safer bet here for broad, well-established prebuilt-
# wheel availability (numpy/scipy/matplotlib/torch all have mature 3.12 wheels) than
# pinning to whatever the newest interpreter happens to be.
FROM python:3.12-slim AS base

WORKDIR /app

# matplotlib needs a backend that doesn't require a display -- Agg (raster, file-
# output only) is the standard headless choice. Every core/demo.py and
# core/validation/*.py script already supports --save/--plot for file output, so
# nothing here actually needs a window; see the Quickstart examples in README.md.
ENV MPLBACKEND=Agg \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

COPY . .
RUN pip install -e ".[dev]"

# Runs the full test suite by default -- the most useful "does this image actually
# work" check, and the same command CI would run. Override for anything else, e.g.:
#   docker run --rm -v "$PWD/out:/app/out" auto-park \
#     python -m core.demo perpendicular_open --save out/demo.gif
CMD ["pytest", "-q"]

FROM base AS rl
RUN pip install -e ".[dev,rl]"
