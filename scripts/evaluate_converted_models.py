#!/usr/bin/env python3
"""
Evaluate all converted HuggingFace checkpoints on benchmarks.
"""
import argparse
import json
import os
import sys
from pathlib import Path
from datetime import datetime

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from evaluation.loaders import load_benchmark


class HuggingFaceEvaluator:
    """Evaluator for HuggingFace models on MCQ benchmarks."""

    def __init__(self, model_name, model_path, device="cuda"):
        """
        Initialize evaluator with a HuggingFace model.

        Args:
            model_name: Short name for logging
            model_path: Path to HuggingFace model directory
            device: Device to run on (cuda/cpu)
        """
        self.model_name = model_name
        self.model_path = model_path
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        print(f"Loading model: {model_name}")
        print(f"Path: {model_path}")
        print(f"Device: {self.device}")

        # Load model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)

        # Set pad token if not set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
        )
        self.model.eval()

        # Get the device of the first model parameter for proper tensor placement
        self.input_device = next(self.model.parameters()).device

        print(f"✓ Model loaded (input device: {self.input_device})")

    def compute_logprob(self, prefix, continuation):
        """Compute log probability of continuation given prefix."""
        prefix = str(prefix)
        continuation = str(continuation)
        full_text = prefix + " " + continuation

        # Tokenize
        prefix_tokens = self.tokenizer.encode(prefix, add_special_tokens=True)
        full_tokens = self.tokenizer.encode(full_text, add_special_tokens=True)

        # Get continuation tokens
        continuation_tokens = full_tokens[len(prefix_tokens):]

        if len(continuation_tokens) == 0:
            return -100.0

        # Convert to tensors
        input_ids = torch.tensor([full_tokens]).to(self.input_device)

        # Get logits
        with torch.no_grad():
            outputs = self.model(input_ids)
            logits = outputs.logits

        # Get log probabilities for continuation tokens
        log_probs = F.log_softmax(logits[0], dim=-1)

        # Sum log probabilities for the continuation tokens
        total_log_prob = 0.0
        for i, token_id in enumerate(continuation_tokens):
            pos = len(prefix_tokens) + i - 1
            if pos >= 0 and pos < log_probs.shape[0]:
                total_log_prob += log_probs[pos, token_id].item()

        return total_log_prob

    def evaluate_mcq(self, prefix, ground_truth, false_options):
        """Evaluate a single MCQ question."""
        all_options = [ground_truth] + false_options
        logprobs = []

        for option in all_options:
            logprob = self.compute_logprob(prefix, option)
            logprobs.append(logprob)

        return torch.tensor(logprobs)

    def evaluate_benchmark(self, benchmark_name, max_samples=None):
        """Evaluate on a benchmark."""
        print(f"\nEvaluating on {benchmark_name}...")

        # Load benchmark
        try:
            benchmark_loader = load_benchmark(benchmark_name)
        except Exception as e:
            print(f"Error loading benchmark {benchmark_name}: {e}")
            return None

        # Collect benchmark items
        benchmark_items = []
        for i, item in enumerate(benchmark_loader):
            benchmark_items.append(item)
            if max_samples and i >= max_samples - 1:
                break

        if len(benchmark_items) == 0:
            print(f"No samples loaded for {benchmark_name}")
            return None

        # Evaluate MCQ format
        confidences = []
        correct_count = 0
        total_count = 0
        detailed_results = []

        for item_data in tqdm(benchmark_items, desc=benchmark_name):
            prefix, ground_truth, false_options, sample_id = item_data

            # Evaluate
            logprobs = self.evaluate_mcq(prefix, ground_truth, false_options)
            confidences.append(logprobs)

            # Check if correct
            predicted_idx = torch.argmax(logprobs).item()
            is_correct = predicted_idx == 0
            if is_correct:
                correct_count += 1
            total_count += 1

            # Store detailed result
            all_options = [ground_truth] + false_options
            detailed_result = {
                'id': sample_id,
                'question': prefix,
                'ground_truth': ground_truth,
                'options': all_options,
                'predicted_idx': predicted_idx,
                'predicted_answer': all_options[predicted_idx] if predicted_idx < len(all_options) else None,
                'is_correct': is_correct,
                'logprobs': logprobs.tolist(),
            }
            detailed_results.append(detailed_result)

        if total_count == 0:
            print(f"No samples evaluated for {benchmark_name}")
            return None

        # Pad confidences to same length
        max_length = max([len(c) for c in confidences])
        padded_confidences = []
        for c in confidences:
            padded = F.pad(c, (0, max_length - len(c)), value=-1e10)
            padded_confidences.append(padded)

        confidences_tensor = torch.stack(padded_confidences)

        # Calculate metrics
        results = self.calculate_metrics(confidences_tensor)
        results['num_samples'] = total_count
        results['format'] = 'mcq'
        results['detailed_results'] = detailed_results

        return results

    def calculate_metrics(self, confidences):
        """Calculate evaluation metrics."""
        # Predictions
        _, predicted = torch.max(confidences, 1)

        # Accuracy
        accuracy = (predicted == 0).float().mean().item()

        # F1, Precision, Recall
        tp = (predicted == 0).float().sum().item()
        fn = (predicted != 0).float().sum().item()
        fp = 0

        precision = tp / (tp + fp + 1e-10)
        recall = tp / (tp + fn + 1e-10)
        f1 = 2 * (precision * recall) / (precision + recall + 1e-10)

        # Path confidence
        softmaxed = F.softmax(confidences, dim=-1)
        path_confidence = softmaxed[:, 0].mean().item()

        # Normalized accuracy
        num_options = confidences.shape[1]
        normalized_accuracy = (accuracy * num_options - 1) / (num_options - 1)

        return {
            'accuracy': accuracy,
            'f1_score': f1,
            'precision': precision,
            'recall': recall,
            'path_confidence': path_confidence,
            'normalized_accuracy': normalized_accuracy,
            'num_options': num_options,
        }


def main():
    parser = argparse.ArgumentParser(description="Evaluate converted checkpoints on benchmarks")
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to converted HuggingFace model"
    )
    parser.add_argument(
        "--model-name",
        type=str,
        required=True,
        help="Short name for the model (e.g., vanilla-step4999)"
    )
    parser.add_argument(
        "--benchmarks",
        nargs="+",
        default=["pacute", "hierarchical", "cute", "langgame", "multi-digit-addition"],
        help="Benchmarks to evaluate on"
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum samples per benchmark (None = all)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/converted_models",
        help="Output directory for results"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device to run on"
    )

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Timestamp for this run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    print(f"\n{'='*80}")
    print(f"Evaluating Model: {args.model_name}")
    print(f"{'='*80}")

    try:
        # Initialize evaluator
        evaluator = HuggingFaceEvaluator(
            model_name=args.model_name,
            model_path=args.model_path,
            device=args.device
        )

        # Evaluate on each benchmark
        model_results = {}
        for benchmark_name in args.benchmarks:
            results = evaluator.evaluate_benchmark(
                benchmark_name=benchmark_name,
                max_samples=args.max_samples
            )

            if results:
                # Save detailed inference results
                detailed_results = results.pop('detailed_results', None)
                if detailed_results:
                    inference_dir = os.path.join(args.output_dir, args.model_name, "inference")
                    os.makedirs(inference_dir, exist_ok=True)
                    inference_file = os.path.join(
                        inference_dir,
                        f"{benchmark_name}_{timestamp}.jsonl"
                    )
                    with open(inference_file, 'w') as f:
                        for result in detailed_results:
                            f.write(json.dumps(result) + '\n')
                    print(f"Saved detailed results: {inference_file}")

                model_results[benchmark_name] = results

                # Print results
                print(f"\n{benchmark_name} Results:")
                print(f"  Samples: {results['num_samples']}")
                print(f"  Accuracy: {results['accuracy']:.4f}")
                print(f"  F1 Score: {results['f1_score']:.4f}")
                print(f"  Precision: {results['precision']:.4f}")
                print(f"  Recall: {results['recall']:.4f}")
                print(f"  Path Confidence: {results['path_confidence']:.4f}")
                print(f"  Normalized Accuracy: {results['normalized_accuracy']:.4f}")

        # Save results
        model_output_dir = os.path.join(args.output_dir, args.model_name)
        os.makedirs(model_output_dir, exist_ok=True)
        model_output_file = os.path.join(
            model_output_dir,
            f"evaluation_results_{timestamp}.json"
        )

        model_result = {
            'model_name': args.model_name,
            'model_path': args.model_path,
            'benchmarks': model_results,
            'timestamp': timestamp
        }

        with open(model_output_file, 'w') as f:
            json.dump(model_result, f, indent=2)

        print(f"\n{'='*80}")
        print(f"✓ Results saved to: {model_output_file}")
        print(f"{'='*80}")

        # Clean up GPU memory
        del evaluator
        torch.cuda.empty_cache()

    except Exception as e:
        print(f"Error evaluating {args.model_name}: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
