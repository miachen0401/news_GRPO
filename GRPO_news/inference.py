"""
Inference Script for GRPO-trained Models

This script loads a trained model and performs inference on GSM8K test examples
or custom math problems.
"""

import logging
from pathlib import Path
from typing import Optional, Union
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MathSolver:
    """Inference wrapper for math problem solving"""

    def __init__(
        self,
        model_path: str,
        device: Optional[str] = None,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ):
        """
        Initialize the math solver.

        Args:
            model_path: Path to trained model (checkpoint or HuggingFace path)
            device: Device to run inference on (auto-detected if None)
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling probability
        """
        self.model_path = model_path
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p

        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        logger.info(f"Loading model from {model_path}")
        logger.info(f"Using device: {self.device}")

        self._load_model()

    def _load_model(self) -> None:
        """Load the model and tokenizer"""
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path,
                trust_remote_code=True
            )

            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto" if self.device == "cuda" else None,
                trust_remote_code=True
            )

            if self.device == "cpu":
                self.model = self.model.to(self.device)

            self.model.eval()
            logger.info("Model loaded successfully")

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

    def solve(self, question: str, return_full_output: bool = False) -> Union[str, dict]:
        """
        Solve a math problem.

        Args:
            question: Math problem to solve
            return_full_output: If True, return dict with answer and reasoning

        Returns:
            Answer string or dict with full output
        """
        prompt = (
            f"{question}\n"
            "Answer the above math problem. "
            "Think step by step. Output the final answer after ####."
        )

        messages = [{"role": "user", "content": prompt}]

        try:
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )

            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                padding=True,
                truncation=True
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )

            response = self.tokenizer.decode(
                outputs[0][inputs.input_ids.shape[1]:],
                skip_special_tokens=True
            )

            if return_full_output:
                answer = self._extract_answer(response)
                return {
                    "question": question,
                    "reasoning": response,
                    "answer": answer
                }
            else:
                return self._extract_answer(response)

        except Exception as e:
            logger.error(f"Inference failed: {e}")
            return "Error: Failed to generate answer"

    def _extract_answer(self, response: str) -> str:
        """
        Extract the final answer from model response.

        Args:
            response: Full model response

        Returns:
            Extracted answer or full response if pattern not found
        """
        if "####" in response:
            answer = response.split("####")[-1].strip()
            return answer
        return response.strip()

    def batch_solve(
        self,
        questions: list[str],
        return_full_output: bool = False
    ) -> list[Union[str, dict]]:
        """
        Solve multiple math problems.

        Args:
            questions: List of math problems
            return_full_output: If True, return dicts with full output

        Returns:
            List of answers or dicts
        """
        results = []
        for i, question in enumerate(questions, 1):
            logger.info(f"Solving problem {i}/{len(questions)}")
            result = self.solve(question, return_full_output)
            results.append(result)
        return results


def main():
    """Main inference entrypoint"""
    import argparse

    parser = argparse.ArgumentParser(
        description="Run inference with GRPO-trained model"
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to trained model checkpoint or HuggingFace model"
    )
    parser.add_argument(
        "--question",
        type=str,
        help="Math question to solve"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (cuda/cpu, auto-detected if not specified)"
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=512,
        help="Maximum tokens to generate"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature"
    )

    args = parser.parse_args()

    solver = MathSolver(
        model_path=args.model,
        device=args.device,
        max_new_tokens=args.max_tokens,
        temperature=args.temperature,
    )

    if args.question:
        logger.info(f"Question: {args.question}")
        result = solver.solve(args.question, return_full_output=True)
        logger.info(f"Reasoning: {result['reasoning']}")
        logger.info(f"Answer: {result['answer']}")
    else:
        logger.info("Interactive mode - enter questions (type 'quit' to exit)")
        while True:
            try:
                question = input("\nQuestion: ").strip()
                if question.lower() in ['quit', 'exit', 'q']:
                    break
                if question:
                    result = solver.solve(question, return_full_output=True)
                    print(f"\nReasoning: {result['reasoning']}")
                    print(f"\nAnswer: {result['answer']}")
            except KeyboardInterrupt:
                break


if __name__ == "__main__":
    main()
