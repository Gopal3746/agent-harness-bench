FROM python:3.12-slim

RUN python -m pip install --no-cache-dir \
    "httpx>=0.28,<1" \
    "matplotlib>=3.9,<4" \
    "pydantic>=2.11,<3"

WORKDIR /workspace

COPY app.py /workspace/app.py
