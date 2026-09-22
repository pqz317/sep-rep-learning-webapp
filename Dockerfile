FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends git && rm -rf /var/lib/apt/lists/*

# Core ML dependencies
RUN pip install --no-cache-dir "jax[cpu]" flax distrax optax
RUN pip install --no-cache-dir tqdm matplotlib seaborn imageio aiofiles

# Web app dependencies
# nicewebrl pinned to e30aca8 (2026-01-14): the next upstream commit (cdc6822, 2026-08-31)
# has an inverted None-check in stages.save_stage_state that raises
# "TypeError: 'NoneType' object is not callable" and silently drops stage-state saves.
RUN pip install --no-cache-dir nicegui "tortoise-orm<1.0" "nicewebrl @ git+https://github.com/KempnerInstitute/nicewebrl.git@e30aca8c6b25bfde55a338779fd0181db72d10e6"

# Overcooked environments (overcooked_v2 and bug fixes not yet in mainline crossEnvCooperation)
RUN git clone --depth=1 https://github.com/pqz317/crossEnvCooperation.git /src/crossEnvCooperation && \
    pip install --no-cache-dir --no-deps -e /src/crossEnvCooperation

# Install app source
COPY . /app
WORKDIR /app
RUN pip install --no-cache-dir -e .

# JAX persistent compilation cache — baked into the image at build time.
# The same path is used at runtime so compiled artifacts are reused.
ENV JAX_COMPILATION_CACHE_DIR=/app/.jax_cache

# Set to "" (empty string) to skip precompilation: docker build --build-arg PRECOMPILE=
ARG PRECOMPILE=""

RUN if [ -n "$PRECOMPILE" ]; then python scripts/precompile.py; fi

ENV HOST=0.0.0.0
ENV PORT=8080
ENV DATA_DIR=/app/data

EXPOSE 8080

CMD ["python", "-m", "web_app.web_app"]
