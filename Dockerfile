# CPU-only image for portability -- tools/qat.py auto-detects a GPU when one
# is actually present (AUTOCIM_QAT_DEVICE), but a container has no direct
# access to the host's GPU without extra runtime flags (--gpus, nvidia-container-toolkit)
# this image doesn't assume. Not a multi-stage/slim-runtime build -- keeping
# it simple until there's an actual deployment target to optimize for.
FROM python:3.13-slim

WORKDIR /app

# Which torch/torchvision wheel (CPU-only vs CUDA, multi-GB) you get depends
# entirely on which index you install from -- never rely on a bare `pip
# install` to pick the right one. Installed first, explicitly from the
# CPU-only index (this container can't use a GPU anyway -- see the image
# comment above) at the exact pinned version from requirements.txt, so the
# later full install is a no-op for these two packages instead of pulling a
# different, unwanted wheel.
COPY requirements.txt .
RUN pip install --no-cache-dir torch==2.13.0 torchvision==0.28.0 --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt

COPY . .

# llm.py defaults AUTOCIM_PLANNER_MODEL to a local Ollama model
# ("ollama:qwen2.5:7b"), which this container can't reach at its default
# localhost:11434 -- either point it at the host's Ollama (-e
# OLLAMA_HOST=http://host.docker.internal:11434) or override
# AUTOCIM_PLANNER_MODEL to a cloud provider spec plus its matching API key
# (ANTHROPIC_API_KEY / OPENAI_API_KEY / GROQ_API_KEY / GOOGLE_API_KEY -- see
# README.md's verified-provider table). Deliberately not set here -- pass
# these at `docker run` time (-e), never bake credentials into an image.
# .dockerignore excludes .env/.env.local, so COPY . . above can't leak them
# in as a file either.
ENTRYPOINT ["python", "main.py"]
