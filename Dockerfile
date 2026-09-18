# syntax=docker/dockerfile:1
# Two targets: base (core sim + tests) and rl (adds the optional `rl` extra, which pulls
# in torch). Build with --target rl only if you need core/rl/train.py.
FROM python:3.12-slim AS base

WORKDIR /app

# matplotlib needs a headless backend -- Agg (raster, file-output only). Scripts already
# support --save/--plot for file output; see README.md's Quickstart.
ENV MPLBACKEND=Agg \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

COPY . .
RUN pip install -e ".[dev]"

# Runs the full test suite by default. Override for anything else, e.g.:
#   docker run --rm -v "$PWD/out:/app/out" auto-park python -m core.demo perpendicular_open --save out/demo.gif
CMD ["pytest", "-q"]

FROM base AS rl
RUN pip install -e ".[dev,rl]"
