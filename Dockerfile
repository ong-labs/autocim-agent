# CPU-only, matching this project's documented scope (tools/qat.py: "no GPU
# in this environment"). Not a multi-stage/slim-runtime build -- keeping it
# simple until there's an actual deployment target to optimize for.
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

# AUTOCIM_PLANNER_MODEL (required by llm.py) and provider API keys
# (ANTHROPIC_API_KEY / OPENAI_API_KEY) are deliberately not set here --
# pass them at `docker run` time, never bake credentials into an image.
ENTRYPOINT ["python", "main.py"]
