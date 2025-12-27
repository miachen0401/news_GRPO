"""
Prepare news classification dataset for VERL GRPO training.

This script processes the raw news CSV data and creates train/val/test splits
in the format required by VERL GRPO training pipeline.

Output format follows the same schema as GSM8K:
- data_source: str
- prompt: List[Dict] (chat messages)
- ability: str
- reward_model: Dict
- extra_info: Dict
"""

import csv
import json
import logging
import random
import sys
from pathlib import Path
from typing import Dict, List, Any

from datasets import Dataset

# Add parent directory to path to import prompts
sys.path.insert(0, str(Path(__file__).parent.parent))
from GRPO_news.prompts import get_news_classification_prompt

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_news_item(row: Dict[str, str]) -> Dict[str, Any]:
    """
    Parse a single news item from CSV row.

    Args:
        row: Dictionary containing CSV row data

    Returns:
        Parsed news item with title, summary, and label
    """
    # Parse raw_json field
    try:
        news_data = json.loads(row['raw_json'])
    except (json.JSONDecodeError, KeyError):
        logger.warning(f"Failed to parse raw_json for row {row.get('id', 'unknown')}")
        return None

    # Extract label from error_log
    error_log = row.get('error_log', '').strip()
    is_event_based = not bool(error_log and 'Filtered out' in error_log)

    # Extract news content
    title = news_data.get('title', '').strip()
    summary = news_data.get('summary', '').strip()

    if not title:
        logger.warning(f"Empty title for row {row.get('id', 'unknown')}")
        return None

    # Combine title and summary for the news text
    news_text = f"{title}"
    if summary:
        news_text += f"\n\n{summary}"

    return {
        'id': row.get('id', ''),
        'symbol': row.get('symbol', ''),
        'text': news_text,
        'title': title,
        'summary': summary,
        'is_event_based': is_event_based,
        'published_at': news_data.get('published_at', ''),
        'source': news_data.get('source', '') or news_data.get('publisher', ''),
    }


def format_news_example(news_item: Dict[str, Any], index: int, split: str) -> Dict[str, Any]:
    """
    Format a single news item into GRPO training format.

    Args:
        news_item: Parsed news item
        index: Example index
        split: Data split (train/val/test)

    Returns:
        Formatted example in GRPO schema
    """
    # Get system prompt from prompts module
    system_prompt = get_news_classification_prompt()

    # Create the user message with the news content
    user_message = f"News: {news_item['text']}"

    # Create chat-style prompt
    prompt = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message}
    ]

    # Ground truth answer
    ground_truth = "true" if news_item['is_event_based'] else "false"

    # Create formatted example
    example = {
        "data_source": "news_classification",
        "prompt": prompt,
        "ability": "news_classification",
        "reward_model": {
            "style": "rule",
            "ground_truth": ground_truth
        },
        "extra_info": {
            "split": split,
            "index": index,
            "news_id": news_item['id'],
            "symbol": news_item['symbol'],
            "source": news_item['source'],
            "is_event_based": news_item['is_event_based'],
            "news_text": news_item['text']  # Add news content for reward logging
        }
    }

    return example


def load_and_parse_csv(csv_path: Path) -> tuple[List[Dict], List[Dict]]:
    """
    Load and parse the news CSV file.

    Args:
        csv_path: Path to the CSV file

    Returns:
        Tuple of (positive_examples, negative_examples)
    """
    logger.info(f"Loading CSV from {csv_path}")

    positive_examples = []
    negative_examples = []

    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)

        for row in reader:
            news_item = parse_news_item(row)

            if news_item is None:
                continue

            if news_item['is_event_based']:
                positive_examples.append(news_item)
            else:
                negative_examples.append(news_item)

    logger.info(f"Loaded {len(positive_examples)} positive examples")
    logger.info(f"Loaded {len(negative_examples)} negative examples")

    return positive_examples, negative_examples


def balance_dataset(
    positive_examples: List[Dict],
    negative_examples: List[Dict],
    ratio: float = 3.0,
    seed: int = 42
) -> List[Dict]:
    """
    Balance the dataset to achieve desired positive:negative ratio.

    Args:
        positive_examples: List of event-based news
        negative_examples: List of non-event-based news
        ratio: Desired negative to positive ratio
        seed: Random seed for sampling

    Returns:
        Balanced list of all examples
    """
    random.seed(seed)

    n_positive = len(positive_examples)
    n_negative_target = int(n_positive * ratio)

    if n_negative_target > len(negative_examples):
        logger.warning(
            f"Requested {n_negative_target} negative examples but only "
            f"{len(negative_examples)} available. Using all negative examples."
        )
        sampled_negative = negative_examples
    else:
        sampled_negative = random.sample(negative_examples, n_negative_target)

    # Combine and shuffle
    all_examples = positive_examples + sampled_negative
    random.shuffle(all_examples)

    logger.info(f"Balanced dataset: {len(positive_examples)} positive, {len(sampled_negative)} negative")
    logger.info(f"Actual ratio (positive:negative): 1:{len(sampled_negative)/len(positive_examples):.2f}")

    return all_examples


def split_dataset(
    examples: List[Dict],
    val_ratio: float = 0.1,
    seed: int = 42
) -> tuple[List[Dict], List[Dict]]:
    """
    Split dataset into train and validation sets.

    Args:
        examples: List of all examples
        val_ratio: Fraction of data to use for validation
        seed: Random seed for splitting

    Returns:
        Tuple of (train_examples, val_examples)
    """
    random.seed(seed)
    random.shuffle(examples)

    n_val = int(len(examples) * val_ratio)
    val_examples = examples[:n_val]
    train_examples = examples[n_val:]

    logger.info(f"Split: {len(train_examples)} train, {len(val_examples)} val")

    return train_examples, val_examples


def save_to_parquet(examples: List[Dict], output_path: Path, split: str):
    """
    Save examples to parquet file using HuggingFace datasets library.
    This preserves nested structures (dicts, lists) properly for VERL.

    Args:
        examples: List of formatted examples
        output_path: Output parquet file path
        split: Data split name (train/val/test)
    """
    logger.info(f"Formatting {len(examples)} examples for {split} split")

    # Format all examples
    formatted_examples = [
        format_news_example(ex, idx, split)
        for idx, ex in enumerate(examples)
    ]

    # Create HuggingFace Dataset (preserves nested structures)
    dataset = Dataset.from_list(formatted_examples)

    # Write to parquet (HuggingFace handles nested structures correctly)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_parquet(output_path)

    logger.info(f"Saved {len(formatted_examples)} examples to {output_path}")


def prepare_news_classification_dataset(
    csv_path: str = "data/news_classification/stock_news_raw_rows.csv",
    output_dir: str = "data/news_classification",
    ratio: float = 3.0,
    val_ratio: float = 0.1,
    seed: int = 42
):
    """
    Main function to prepare news classification dataset.

    Args:
        csv_path: Path to raw CSV file
        output_dir: Directory to save output parquet files
        ratio: Desired negative to positive ratio
        val_ratio: Fraction of data for validation
        seed: Random seed for reproducibility
    """
    logger.info("Starting news classification dataset preparation")
    logger.info(f"CSV path: {csv_path}")
    logger.info(f"Output dir: {output_dir}")
    logger.info(f"Target ratio (positive:negative): 1:{ratio}")
    logger.info(f"Validation ratio: {val_ratio}")

    # Load and parse CSV
    csv_path = Path(csv_path)
    positive_examples, negative_examples = load_and_parse_csv(csv_path)

    # Balance dataset
    balanced_examples = balance_dataset(positive_examples, negative_examples, ratio=ratio, seed=seed)

    # Split into train/val
    train_examples, val_examples = split_dataset(balanced_examples, val_ratio=val_ratio, seed=seed)

    # Save to parquet
    output_dir = Path(output_dir)
    save_to_parquet(train_examples, output_dir / "train.parquet", "train")
    save_to_parquet(val_examples, output_dir / "val.parquet", "val")

    # Print statistics
    logger.info("=" * 50)
    logger.info("Dataset preparation complete!")
    logger.info(f"Train: {len(train_examples)} examples")
    logger.info(f"Val: {len(val_examples)} examples")
    logger.info(f"Total: {len(balanced_examples)} examples")

    # Count labels in train/val
    train_positive = sum(1 for ex in train_examples if ex['is_event_based'])
    train_negative = len(train_examples) - train_positive
    val_positive = sum(1 for ex in val_examples if ex['is_event_based'])
    val_negative = len(val_examples) - val_positive

    logger.info(f"Train split - Positive: {train_positive}, Negative: {train_negative}")
    logger.info(f"Val split - Positive: {val_positive}, Negative: {val_negative}")
    logger.info("=" * 50)

if __name__ == "__main__":
    prepare_news_classification_dataset()
