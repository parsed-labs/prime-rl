#!/usr/bin/env python3
"""Upload synthetic datasets to Hugging Face Hub as proper dataset repos with chat template formatting."""

import argparse
import os
from typing import Optional, Dict, Any

import yaml
from datasets import Dataset, DatasetDict
from huggingface_hub import HfApi
from transformers import AutoTokenizer


def _load_token(config_path: Optional[str]) -> Optional[str]:
    """Load HuggingFace token from config file.
    
    Args:
        config_path: Path to YAML config file
        
    Returns:
        HuggingFace token or None if not found
    """
    path = config_path or "config.yaml"
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    for key in ("huggingface", "huggingface_token", "hf_token"):
        value = data.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _load_raw_dataset(file_path: str, data_format: Optional[str]) -> Dataset:
    """Load dataset from CSV or JSONL file.
    
    Args:
        file_path: Path to dataset file
        data_format: Optional explicit format (csv, jsonl, json)
        
    Returns:
        Dataset object
    """
    ext = data_format or os.path.splitext(file_path)[1].lower().lstrip(".")
    if ext in {"csv"}:
        dataset = Dataset.from_csv(file_path)
    elif ext in {"jsonl", "json"}:
        dataset = Dataset.from_json(file_path)
    else:
        raise ValueError(f"Unsupported data format for {file_path}. Use CSV or JSONL.")
    return dataset


def _format_with_chat_template(
    example: Dict[str, Any],
    tokenizer: AutoTokenizer,
    has_messages: bool,
) -> Dict[str, str]:
    """Format a single example using the model's chat template.
    
    Args:
        example: Dataset example (either with 'text' column or chat messages)
        tokenizer: Tokenizer with chat template
        has_messages: Whether the example has chat message format
        
    Returns:
        Dictionary with 'text' key containing formatted conversation
    """
    if has_messages:
        # JSONL format with system/user/assistant messages
        messages = []
        
        # Build messages list from the example
        if "system" in example and example["system"]:
            messages.append({"role": "system", "content": example["system"]})
        
        if "user" in example and example["user"]:
            messages.append({"role": "user", "content": example["user"]})
        
        if "assistant" in example and example["assistant"]:
            messages.append({"role": "assistant", "content": example["assistant"]})
        
        # Handle alternative format: list of messages
        if "messages" in example:
            messages = example["messages"]
        
        # Apply chat template
        formatted_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )
    else:
        # CSV format with just "text" column - treat as a user message with assistant response
        # We'll create a simple user message format
        text_content = example.get("text", "")
        messages = [{"role": "user", "content": text_content}]
        
        formatted_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,  # Add generation prompt for single user messages
        )
    
    return {"text": formatted_text}


def _apply_chat_template(
    dataset: Dataset,
    model_name: str,
    token: Optional[str],
) -> Dataset:
    """Apply chat template to entire dataset.
    
    Args:
        dataset: Raw dataset
        model_name: Model name to load tokenizer from
        token: HuggingFace token for private models
        
    Returns:
        Dataset with 'text' column containing formatted conversations
    """
    # Load tokenizer with chat template
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=token, trust_remote_code=True)
    
    # Check if dataset has chat message format or just text
    first_example = dataset[0]
    has_messages = any(key in first_example for key in ["messages", "user", "assistant", "system"])
    
    # Apply chat template to all examples
    formatted_dataset = dataset.map(
        lambda x: _format_with_chat_template(x, tokenizer, has_messages),
        remove_columns=dataset.column_names,
        desc="Applying chat template",
    )
    
    return formatted_dataset


def _format_prompt_completion(example: Dict[str, Any]) -> Dict[str, Any]:
    """Build prompt/completion columns from chat-style fields.
    
    Produces two columns:
      - prompt: list of messages with roles 'system' and 'user'
      - completion: list of messages with role 'assistant'
    """
    # messages list takes precedence if present
    if "messages" in example and isinstance(example["messages"], list):
        msgs = example["messages"]
        prompt = [m for m in msgs if m.get("role") in {"system", "user"}]
        completion = [m for m in msgs if m.get("role") == "assistant"]
        return {"prompt": prompt, "completion": completion}

    # otherwise, use system/user/assistant scalar fields if present
    prompt: list = []
    if example.get("system"):
        prompt.append({"role": "system", "content": example["system"]})
    if example.get("user"):
        prompt.append({"role": "user", "content": example["user"]})

    completion: list = []
    if example.get("assistant"):
        completion.append({"role": "assistant", "content": example["assistant"]})

    # fallback: plain text datasets (CSV with 'text') → treat as user-only prompt
    if not prompt and "text" in example:
        prompt = [{"role": "user", "content": example.get("text", "")}]

    return {"prompt": prompt, "completion": completion}


def upload_dataset(
    file_path: str,
    repo_id: str,
    commit_message: str,
    config_path: Optional[str],
    token: Optional[str],
    data_format: Optional[str],
    split: str,
    private: bool,
    model_name: str,
    prompt_completion: bool,
) -> None:
    """Upload dataset to HuggingFace Hub with chat template formatting.
    
    Args:
        file_path: Path to local dataset file
        repo_id: HuggingFace repo ID (e.g., parsed/dataset-name)
        commit_message: Git commit message
        config_path: Path to config file with HF token
        token: Explicit HF token (overrides config)
        data_format: Explicit data format (csv, jsonl, json)
        split: Dataset split name (default: train)
        private: Whether to make repo private
        model_name: Model name to use for chat template
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    # Resolve token
    resolved_token = token or _load_token(config_path)
    if not resolved_token:
        raise ValueError(
            "No HuggingFace token found. Provide via --token or in config file "
            "(keys: huggingface_token, hf_token, or huggingface)"
        )

    # Load raw dataset
    print(f"Loading dataset from {file_path}...")
    raw_dataset = _load_raw_dataset(file_path, data_format)
    print(f"Loaded {len(raw_dataset)} examples")
    
    if prompt_completion:
        print("Building prompt/completion columns...")
        formatted_dataset = raw_dataset.map(
            _format_prompt_completion,
            remove_columns=raw_dataset.column_names,
            desc="Creating prompt/completion",
        )
        print(f"Prepared {len(formatted_dataset)} examples with prompt/completion")
    else:
        # Apply chat template to a single 'text' column
        print(f"Applying chat template from {model_name}...")
        formatted_dataset = _apply_chat_template(raw_dataset, model_name, resolved_token)
        print(f"Formatted {len(formatted_dataset)} examples")
    
    # Wrap in DatasetDict
    dataset_dict = DatasetDict({split: formatted_dataset})

    # Create repo if it doesn't exist
    print(f"Creating/updating repo {repo_id}...")
    api = HfApi(token=resolved_token)
    api.create_repo(repo_id=repo_id, repo_type="dataset", exist_ok=True, private=private)

    # Push to hub
    print(f"Pushing dataset to {repo_id}...")
    dataset_dict.push_to_hub(
        repo_id=repo_id,
        token=resolved_token,
        private=private,
        commit_message=commit_message,
    )
    
    print(f"✓ Successfully uploaded dataset to {repo_id} (split: {split})")
    print(f"  - Examples: {len(formatted_dataset)}")
    print(f"  - Private: {private}")
    if prompt_completion:
        print("  - Columns: prompt, completion")
    else:
        print(f"  - Model template: {model_name}")


def main() -> None:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Upload CSV/JSONL datasets to Hugging Face Hub with chat template formatting"
    )
    parser.add_argument(
        "file",
        help="Local path to the dataset file (CSV or JSONL)",
    )
    parser.add_argument(
        "repo",
        help="Destination Hugging Face dataset repo id (e.g., parsed/openx-synth-text)",
    )
    parser.add_argument(
        "--commit-message",
        default="Add synthetic dataset",
        help="Commit message to use for the dataset upload",
    )
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Path to YAML config containing huggingface_token (default: config.yaml)",
    )
    parser.add_argument(
        "--token",
        default=None,
        help="Explicit Hugging Face access token (overrides config file)",
    )
    parser.add_argument(
        "--format",
        default=None,
        choices=["csv", "jsonl", "json"],
        help="Optional explicit data format; inferred from file extension if omitted",
    )
    parser.add_argument(
        "--split",
        default="train",
        help="Split name to assign within the dataset repository (default: train)",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Upload the dataset to a private Hugging Face repo",
    )
    parser.add_argument(
        "--model",
        default="Qwen/Qwen3-32B",
        help="Model to use for chat template (default: Qwen/Qwen2.5-32B-Instruct)",
    )
    parser.add_argument(
        "--prompt-completion",
        action="store_true",
        help="If set, upload two columns: 'prompt' (system+user) and 'completion' (assistant)",
    )

    args = parser.parse_args()

    upload_dataset(
        file_path=args.file,
        repo_id=args.repo,
        commit_message=args.commit_message,
        config_path=args.config,
        token=args.token,
        data_format=args.format,
        split=args.split,
        private=args.private,
        model_name=args.model,
        prompt_completion=args.prompt_completion,
    )


if __name__ == "__main__":
    main()

