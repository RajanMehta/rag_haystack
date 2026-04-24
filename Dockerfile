FROM python:3.11-slim@sha256:9c85d1d49df54abca1c5db3b4016400e198e9e9bb699f32f1ef8e5c0c2149ccf

# Pinned uv version for reproducible builds
COPY --from=ghcr.io/astral-sh/uv:0.11.3 /uv /uvx /usr/local/bin/

# This prevents Python from writing out pyc files
ENV PYTHONDONTWRITEBYTECODE=1
# Install into the system interpreter; no project venv inside the image
ENV UV_PROJECT_ENVIRONMENT=/usr/local
ENV UV_LINK_MODE=copy
ENV UV_COMPILE_BYTECODE=1
ENV UV_NO_CACHE=1

# Build-time toggle: pass `--build-arg INSTALL_DEV=true` to include dev deps
ARG INSTALL_DEV=false

RUN apt-get -y update \
    && apt-get install -y locales locales-all gnupg sudo curl wget vim \
    unzip rsyslog gettext patch gcc g++ build-essential \
    pkg-config libxml2-dev libxmlsec1-dev libxmlsec1-openssl xpdf \
    && apt-get dist-upgrade -y \
    && rm -rf /var/lib/apt/lists

WORKDIR /app

COPY pyproject.toml uv.lock ./

RUN if [ "$INSTALL_DEV" = "true" ]; then \
        uv sync --locked --no-install-project --group dev; \
    else \
        uv sync --locked --no-install-project --no-default-groups; \
    fi

RUN python -m nltk.downloader --dir=/usr/local/lib/nltk_data punkt_tab punkt

# Bake the embedder into the image so first /query doesn't race on a runtime download.
# Uses HF_HOME as the system-wide cache location; chowned to uid 1000 so the runtime
# user can read it.
ENV HF_HOME=/opt/hf_cache
RUN mkdir -p $HF_HOME \
    && python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-mpnet-base-v2')" \
    && chown -R 1000:1000 $HF_HOME

COPY . .

RUN if [ "$INSTALL_DEV" = "true" ]; then \
        uv sync --locked --group dev; \
    else \
        uv sync --locked --no-default-groups; \
    fi

COPY scripts/haystack-entry.sh /usr/local/bin/haystack-entry.sh
COPY scripts/haystack-worker-entry.sh /usr/local/bin/haystack-worker-entry.sh
RUN chmod -R ag=u /usr/local/bin/haystack-entry.sh /usr/local/bin/haystack-worker-entry.sh tests/samples/faq

ENV PORT=31415
EXPOSE 31415

# Ensure uid 1000 resolves via getpwuid (newer torch/transformers call getpass.getuser())
RUN useradd -u 1000 -m -s /bin/bash appuser

CMD /usr/local/bin/haystack-entry.sh

USER 1000:1000
