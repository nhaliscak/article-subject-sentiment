FROM python:3.12-slim

WORKDIR /app

# build-essential is kept defensively: spaCy/numpy/blis publish prebuilt
# manylinux wheels for x86_64 reliably, but aarch64 wheel coverage varies by
# release. Untested on real ARM64 hardware from this build environment - see
# README "ARM64 status" section before relying on this in the RK1 cluster.
RUN apt-get update && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    && python -m spacy download en_core_web_md

COPY app ./app

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
