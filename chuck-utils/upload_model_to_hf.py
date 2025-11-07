"""Upload a local HF-compatible model folder to the Hugging Face Hub.

Usage:
  python upload_model_to_hf.py /path/to/model_dir org_or_user/repo-name \
    --private \
    --branch main \
    --commit-message "Add merged SFT weights"

Auth:
  - Prefer `huggingface-cli login` beforehand, or set HF_TOKEN in env.
"""

from __future__ import annotations

import os
from argparse import ArgumentParser
from pathlib import Path

from huggingface_hub import HfApi


def parse_args():
    p = ArgumentParser(description="Upload a model folder to the Hugging Face Hub.")
    p.add_argument("model_dir", type=Path, help="Local folder containing HF model files (config, tokenizer, weights)")
    p.add_argument("repo_id", help="Target Hub repo id, e.g. org_or_user/repo-name")
    p.add_argument("--repo-type", default="model", choices=["model", "dataset", "space"], help="Hub repo type")
    # Privacy toggles: default private for safety
    p.add_argument("--private", dest="private", action="store_true", help="Create/ensure repo is private")
    p.add_argument("--public", dest="private", action="store_false", help="Create/ensure repo is public")
    p.set_defaults(private=True)
    p.add_argument("--branch", default="main", help="Target branch (revision) to push to")
    p.add_argument("--commit-message", default="Add model snapshot", help="Commit message")
    p.add_argument(
        "--token",
        default=os.environ.get("HF_TOKEN"),
        help="HF token (optional). If omitted, uses cached login if available.",
    )
    return p.parse_args()


def main():
    args = parse_args()

    if not args.model_dir.is_dir():
        raise SystemExit(f"model_dir not found or not a directory: {args.model_dir}")

    api = HfApi(token=args.token) if args.token else HfApi()

    # Create repo if needed; no-op if it exists
    api.create_repo(
        repo_id=args.repo_id,
        repo_type=args.repo_type,
        private=bool(args.private),
        exist_ok=True,
    )

    # Upload folder contents at repo root
    api.upload_folder(
        repo_id=args.repo_id,
        repo_type=args.repo_type,
        folder_path=str(args.model_dir),
        path_in_repo=".",
        commit_message=args.commit_message,
        revision=args.branch,
    )

    print(f"Uploaded {args.model_dir} -> https://huggingface.co/{args.repo_id} (branch: {args.branch})")


if __name__ == "__main__":
    main()


