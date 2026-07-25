# CPU-only image for portability -- tools/qat.py auto-detects a GPU when one
# is actually present (AUTOCIM_QAT_DEVICE), but a container has no direct
# access to the host's GPU without extra runtime flags (--gpus, nvidia-container-toolkit)
# this image doesn't assume. Not a multi-stage/slim-runtime build -- keeping
# it simple until there's an actual deployment target to optimize for.
FROM python:3.13-slim

WORKDIR /app

# torch/torchvision from PyPI's default index pull in CUDA wheels
# (multi-GB, unneeded here) -- installed first from the CPU-only index at
# the exact pinned version from requirements.txt so the later full install
# is a no-op for these two packages instead of re-downloading the CUDA build.
COPY requirements.txt .
RUN pip install --no-cache-dir torch==2.13.0 torchvision==0.28.0 --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt

COPY . .

# AUTOCIM_PLANNER_MODEL (required by llm.py) and the matching provider API
# key (ANTHROPIC_API_KEY / OPENAI_API_KEY / GROQ_API_KEY / GOOGLE_API_KEY --
# see README.md's verified-provider table) are deliberately not set here --
# pass them at `docker run` time (-e), never bake credentials into an image.
# .dockerignore excludes .env/.env.local, so COPY . . above can't leak them
# in as a file either.
ENTRYPOINT ["python", "main.py"]
