run:
    uv run ruff check . --fix
    uv run main.py

bigram:
    uv run bigram.py

bow:
    uv run bow.py
