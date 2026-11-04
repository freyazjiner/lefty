"""Build text embedding + Chroma indexes for the product catalog."""

from app.shop.chroma_index import build_chroma_index
from app.shop.embed_index import build_index


def main() -> None:
    path = build_index(force=True)
    print(f"[done] numpy index → {path}")
    n = build_chroma_index(force=True)
    print(f"[done] chroma catalog docs → {n}")


if __name__ == "__main__":
    main()
