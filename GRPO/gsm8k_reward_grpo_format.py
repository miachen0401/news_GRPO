"""
Custom GSM8K reward function matching the original GRPO implementation.

This reward function implements the same multi-component reward system as the
original grpo/reward.py file, which expects responses in the format:
<think>
reasoning process
</think>
<answer>
final answer
</answer>

Reward components:
1. correctness_reward: 2.0 for exact match, partial credit for close answers
2. digit_reward: Up to 0.5 for presence of numbers
3. hard_format_reward: Up to 0.6 for proper XML structure
4. mark_reward: Up to 0.5 for proper XML tags

Total possible reward: ~3.6 (correctness + format rewards)
"""

from __future__ import annotations

import re
import os
import logging
from typing import Any, Dict, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Sampling configuration - log every Nth question to avoid spam
_LOG_SAMPLE_RATE = int(os.getenv("REWARD_LOG_SAMPLE_RATE", "25"))  # Default: log every 25 questions
_ENABLE_LOGGING = os.getenv("REWARD_ENABLE_LOGGING", "true").lower() == "true"
_call_counter = 0
_question_cache = {}  # Cache to group samples by question


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
    return answer.strip()


def correctness_reward_component(response: str, answer: str) -> float:
    """
    Compute correctness reward with partial credit.

    Returns 2.0 for exact match, partial credit for close numeric answers.

    Parameters
    ----------
    response :
        Model-generated response
    answer :
        Ground truth answer

    Returns
    -------
    float
        Reward in [0.0, 2.0]
    """
    extracted_response = extract_answer(response)

    if extracted_response == str(answer):
        return 2.0

    # Try to extract numeric value and give partial credit
    try:
        # Extract numbers from response
        numbers = re.findall(r'\d+', extracted_response)
        if numbers:
            response_num = int(numbers[0])
            ans_num = int(answer)
            # Give partial credit based on how close the answer is
            distance = abs(response_num - ans_num)
            if distance == 0:
                return 2.0
            elif distance <= 5:
                return 1.0
            elif distance <= 10:
                return 0.5
            else:
                return 0.0
        else:
            return 0.0
    except:
        return 0.0


def digit_reward_component(response: str) -> float:
    """
    Reward responses containing numbers (shows reasoning).

    Parameters
    ----------
    response :
        Model-generated response

    Returns
    -------
    float
        Reward in [0.0, 0.5]
    """
    extracted_response = extract_answer(response)

    # Check if response contains numbers
    numbers = re.findall(r'\d+', extracted_response)
    if numbers:
        # Give more reward for more numbers (shows more reasoning)
        reward = min(0.5, len(numbers) * 0.1)
        return reward
    else:
        return 0.0


def hard_format_reward_component(response: str) -> float:
    """
    Reward proper XML structure.

    Checks for:
    - Presence of <think> and </think> tags
    - Presence of <answer> and </answer> tags
    - Exactly one of each tag

    Parameters
    ----------
    response :
        Model-generated response

    Returns
    -------
    float
        Reward in [0.0, 0.6]
    """
    reward = 0.0

    # Check for think section
    if "<think>" in response and "</think>" in response:
        reward += 0.2
    # Check for answer section
    if "<answer>" in response and "</answer>" in response:
        reward += 0.2
    # Check for proper structure (exactly one of each)
    if response.count("<think>") == 1 and response.count("</think>") == 1:
        reward += 0.1
    if response.count("<answer>") == 1 and response.count("</answer>") == 1:
        reward += 0.1

    return reward


def mark_reward_component(response: str) -> float:
    """
    Reward presence of proper XML tags.

    Parameters
    ----------
    response :
        Model-generated response

    Returns
    -------
    float
        Reward in [0.0, 0.5]
    """
    reward = 0.0

    # Check for proper XML tags
    if "<think>" in response:
        reward += 0.1
    if "</think>" in response:
        reward += 0.1
    if "<answer>" in response:
        reward += 0.1
    if "</answer>" in response:
        reward += 0.1

    # Bonus for proper structure
    if response.count("<think>") == 1 and response.count("</think>") == 1:
        reward += 0.05
    if response.count("<answer>") == 1 and response.count("</answer>") == 1:
        reward += 0.05

    return reward


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: Optional[Dict[str, Any]] = None,
) -> float:
    """
    Compute multi-component reward matching original GRPO implementation.

    This combines all four reward components with equal weighting (weights=1.0 each):
    - correctness_reward: 2.0 max (main signal)
    - digit_reward: 0.5 max
    - hard_format_reward: 0.6 max
    - mark_reward: 0.5 max

    Total possible reward: ~3.6

    Parameters
    ----------
    data_source :
        Dataset name (unused, for API compatibility)
    solution_str :
        Model-generated solution text
    ground_truth :
        Ground truth answer (e.g., "72")
    extra_info :
        Optional extra fields (unused)

    Returns
    -------
    float
        Combined reward score
    """
    global _call_counter
    _call_counter += 1

    _ = data_source

    # Compute all reward components
    correctness = correctness_reward_component(solution_str, ground_truth)
    digit = digit_reward_component(solution_str)
    hard_format = hard_format_reward_component(solution_str)
    mark = mark_reward_component(solution_str)

    # Combine with equal weights (all weights = 1.0 in original implementation)
    total_reward = correctness + digit + hard_format + mark

    # Log sample outputs periodically for monitoring (every 25 questions)
    if _ENABLE_LOGGING and _call_counter % _LOG_SAMPLE_RATE == 0:
        extracted = extract_answer(solution_str)

        # Extract question from extra_info if available
        question = ""
        if extra_info and isinstance(extra_info, dict):
            question = extra_info.get('question', '')

        # Truncate long outputs for logging
        display_output = solution_str[:500] + "..." if len(solution_str) > 500 else solution_str
        display_question = question[:200] + "..." if len(question) > 200 else question

        logger.info("=" * 100)
        logger.info(f"📊 ROLLOUT SAMPLE #{_call_counter // _LOG_SAMPLE_RATE} (Call #{_call_counter})")
        logger.info("=" * 100)
        logger.info("")
        logger.info(f"❓ QUESTION:")
        logger.info(f"   {display_question}")
        logger.info("")
        logger.info(f"✓ GROUND TRUTH:")
        logger.info(f"   {ground_truth}")
        logger.info("")
        logger.info(f"🤖 MODEL ANSWER:")
        logger.info(f"   {extracted}")
        logger.info(f"   {'✓ CORRECT' if extracted == str(ground_truth) else '✗ INCORRECT'}")
        logger.info("")
        logger.info("-" * 100)
        logger.info(f"📈 REWARD BREAKDOWN:")
        logger.info(f"   • Correctness:  {correctness:5.2f} / 2.00  {'✓' if correctness >= 2.0 else '✗'}")
        logger.info(f"   • Digit:        {digit:5.2f} / 0.50  {'✓' if digit >= 0.3 else '○'}")
        logger.info(f"   • Format:       {hard_format:5.2f} / 0.60  {'✓' if hard_format >= 0.5 else '○'}")
        logger.info(f"   • Mark:         {mark:5.2f} / 0.50  {'✓' if mark >= 0.4 else '○'}")
        logger.info(f"   • TOTAL REWARD: {total_reward:5.2f} / 3.60")
        logger.info("-" * 100)
        logger.info(f"📝 FULL MODEL OUTPUT:")
        logger.info(display_output)
        logger.info("=" * 100)
        logger.info("")

        # Also try to import wandb and log there if available
        try:
            import wandb
            if wandb.run is not None:
                # Create a formatted table for wandb
                output_table = f"""
                <div style='font-family: monospace; padding: 10px;'>
                <h3>Sample #{_call_counter // _LOG_SAMPLE_RATE}</h3>
                <p><strong>Question:</strong><br/>{question}</p>
                <p><strong>Ground Truth:</strong> {ground_truth}</p>
                <p><strong>Model Answer:</strong> {extracted}</p>
                <p><strong>Correct:</strong> {'✓ YES' if extracted == str(ground_truth) else '✗ NO'}</p>
                <hr/>
                <p><strong>Full Output:</strong></p>
                <pre>{display_output}</pre>
                </div>
                """
                
                wandb.log({
                    "sample/reward_total": total_reward,
                    "sample/reward_correctness": correctness,
                    "sample/reward_digit": digit,
                    "sample/reward_format": hard_format,
                    "sample/reward_mark": mark,
                    "sample/question": question,
                    "sample/ground_truth": str(ground_truth),
                    "sample/extracted_answer": extracted,
                    "sample/is_correct": 1.0 if extracted == str(ground_truth) else 0.0,
                    "sample/output_html": wandb.Html(output_table),
                    "sample/step": _call_counter,
                })
        except (ImportError, Exception):
            pass  # Wandb not available or not initialized

    return total_reward
