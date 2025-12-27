"""
Custom reward function for news classification task.

This reward function evaluates binary classification responses (true/false) for
determining whether a news article is event-based or not.

Expected response format:
<think>
reasoning process
</think>
<answer>
true/false
</answer>

Reward components:
1. correctness_reward: 2.0 for exact match (true/false)
2. format_reward: 0.2 total (0.1 for <think> tags + 0.1 for <answer> tags)

Total possible reward: 2.2 (correctness + format)
"""

from __future__ import annotations

import re
import os
import logging
import pathlib
from typing import Any, Dict, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Log file path
_log_file_path = str(pathlib.Path.cwd() / "reward_samples_news.log")

# Sampling configuration
_LOG_SAMPLE_RATE = int(os.getenv("REWARD_LOG_SAMPLE_RATE", "25"))
_ENABLE_LOGGING = os.getenv("REWARD_ENABLE_LOGGING", "true").lower() == "true"
_ENABLE_CONSOLE_OUTPUT = os.getenv("REWARD_ENABLE_CONSOLE_OUTPUT", "false").lower() == "true"
_call_counter = 0


def extract_answer(text: str) -> str:
    """
    Extract answer from <answer></answer> tags.

    Parameters
    ----------
    text :
        The full model output

    Returns
    -------
    str
        Extracted answer or empty string if not found
    """
    if "<answer>" not in text or "</answer>" not in text:
        return ""

    answer = text.split("<answer>")[-1]
    answer = answer.split("</answer>")[0]
    return answer.strip().lower()


def extract_thinking(text: str) -> str:
    """
    Extract reasoning from <think></think> tags.

    Parameters
    ----------
    text :
        The full model output

    Returns
    -------
    str
        Extracted reasoning or empty string if not found
    """
    if "<think>" not in text or "</think>" not in text:
        return ""

    thinking = text.split("<think>")[-1]
    thinking = thinking.split("</think>")[0]
    return thinking.strip()


def correctness_reward_component(response: str, ground_truth: str) -> float:
    """
    Compute correctness reward for binary classification.

    Returns 2.0 for exact match (true/false), 0.0 otherwise.

    Parameters
    ----------
    response :
        Model-generated response
    ground_truth :
        Ground truth answer ("true" or "false")

    Returns
    -------
    float
        Reward in [0.0, 2.0]
    """
    extracted_response = extract_answer(response)
    ground_truth_clean = ground_truth.strip().lower()

    # Exact match
    if extracted_response == ground_truth_clean:
        return 2.0

    # Check for common variations
    if ground_truth_clean == "true":
        if extracted_response in ["yes", "1", "event-based", "event based", "true."]:
            return 2.0
    elif ground_truth_clean == "false":
        if extracted_response in ["no", "0", "not event-based", "not event based", "false."]:
            return 2.0

    return 0.0


def format_reward_component(response: str) -> float:
    """
    Reward proper XML format structure.

    Simplified format reward:
    - 0.1 for having both <think> and </think> tags
    - 0.1 for having both <answer> and </answer> tags

    Parameters
    ----------
    response :
        Model-generated response

    Returns
    -------
    float
        Reward in [0.0, 0.2]
    """
    reward = 0.0

    # 0.1 for having think tags
    if "<think>" in response and "</think>" in response:
        reward += 0.1

    # 0.1 for having answer tags
    if "<answer>" in response and "</answer>" in response:
        reward += 0.1

    return reward


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: Optional[Dict[str, Any]] = None,
) -> float:
    """
    Compute simplified reward for news classification.

    Reward components:
    - correctness_reward: 2.0 max (exact match for true/false)
    - format_reward: 0.2 max (0.1 for think tags + 0.1 for answer tags)

    Total possible reward: 2.2

    Parameters
    ----------
    data_source :
        Dataset name (should be "news_classification")
    solution_str :
        Model-generated solution text
    ground_truth :
        Ground truth answer ("true" or "false")
    extra_info :
        Optional extra fields

    Returns
    -------
    float
        Combined reward score (0-2.2)
    """
    global _call_counter
    _call_counter += 1

    # Debug counter
    try:
        with open(_log_file_path + ".counter", "a") as f:
            f.write(f"Call #{_call_counter}\n")
            f.flush()
    except:
        pass

    _ = data_source

    # Compute reward components
    correctness = correctness_reward_component(solution_str, ground_truth)
    format_reward = format_reward_component(solution_str)

    # Combine rewards
    total_reward = correctness + format_reward

    # Extract for logging
    extracted_answer = extract_answer(solution_str)
    ground_truth_clean = ground_truth.strip().lower()
    is_correct = 1.0 if correctness >= 2.0 else 0.0

    # Extract question/news text from extra_info if available
    question_text = "N/A"
    if extra_info is not None:
        # Extract news_text directly from extra_info
        question_text = extra_info.get("news_text", "N/A")
    else:
        # Debug: log if extra_info is None (first few calls only)
        if _call_counter <= 5:
            logger.debug(f"Call #{_call_counter}: extra_info is None")

    # Truncate question for display if too long
    display_question = question_text[:300] + "..." if len(question_text) > 300 else question_text
    display_output = solution_str[:500] + "..." if len(solution_str) > 500 else solution_str

    # Log sample outputs periodically
    if _ENABLE_LOGGING and _call_counter % _LOG_SAMPLE_RATE == 0:
        log_msg = f"""
{"=" * 100}
📊 NEWS CLASSIFICATION SAMPLE #{_call_counter // _LOG_SAMPLE_RATE} (Call #{_call_counter})
{"=" * 100}

📰 NEWS ARTICLE:
{display_question}

✓ GROUND TRUTH: {ground_truth_clean} (event-based: {ground_truth_clean == 'true'})

🤖 MODEL ANSWER: {extracted_answer}  {'✓ CORRECT' if is_correct == 1.0 else '✗ INCORRECT'}

{"-" * 100}
📈 REWARD BREAKDOWN:
   • Correctness:  {correctness:5.2f} / 2.00  {'✓' if correctness >= 2.0 else '✗'}
   • Format:       {format_reward:5.2f} / 0.20  {'✓' if format_reward >= 0.2 else '○'}
   • TOTAL REWARD: {total_reward:5.2f} / 2.20
   • ACCURACY:     {is_correct:5.2f} (Binary)
{"-" * 100}
📝 FULL MODEL OUTPUT:
{display_output}
{"=" * 100}

"""

        # Write to file
        try:
            with open(_log_file_path, "a") as f:
                f.write(log_msg)
                f.flush()
        except Exception as e:
            if _ENABLE_CONSOLE_OUTPUT or logger.isEnabledFor(logging.DEBUG):
                try:
                    import sys
                    print(f"[REWARD LOG ERROR] {e}", file=sys.stderr, flush=True)
                except:
                    pass

        # Console output if enabled
        if _ENABLE_CONSOLE_OUTPUT:
            print(log_msg, flush=True)
        elif logger.isEnabledFor(logging.DEBUG):
            logger.debug(log_msg)

    return total_reward
