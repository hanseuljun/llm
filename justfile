run:
    uv run ruff check . --fix
    uv run main.py

bigram:
    uv run bigram.py

bow2:
    uv run bow_v2.py

bow4:
    uv run bow_v4.py
