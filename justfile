set positional-arguments

run *ARGS:
    uv run ruff check . --fix
    uv run main.py "$@"

bigram1:
    uv run bigram_v1.py

bow2:
    uv run bow_v2.py

bow4:
    uv run bow_v4.py

selfattention5:
    uv run self_attention_v5.py

transformer5 *ARGS:
    uv run transformer_v5.py "$@"
