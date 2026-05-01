FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends git && rm -rf /var/lib/apt/lists/*

# Core ML dependencies
RUN pip install --no-cache-dir "jax[cpu]" flax distrax optax
RUN pip install --no-cache-dir tqdm wandb matplotlib seaborn imageio aiofiles

# Web app dependencies
RUN pip install --no-cache-dir nicegui "tortoise-orm<1.0" "nicewebrl @ git+https://github.com/KempnerInstitute/nicewebrl.git"

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

# W&B credentials needed only during the precompile build step.
# Pass at build time: docker build --build-arg WANDB_API_KEY_BUILD=... --build-arg WANDB_ENTITY_BUILD=...
# These ARGs are not promoted to ENV, so they are not stored in the final image config.
ARG WANDB_API_KEY_BUILD=""
ARG WANDB_ENTITY_BUILD=""

RUN WANDB_API_KEY=${WANDB_API_KEY_BUILD} \
    WANDB_ENTITY=${WANDB_ENTITY_BUILD} \
    python scripts/precompile.py

ENV HOST=0.0.0.0
ENV PORT=8080
ENV DATA_DIR=/data

EXPOSE 8080

CMD ["python", "-m", "web_app.web_app"]
