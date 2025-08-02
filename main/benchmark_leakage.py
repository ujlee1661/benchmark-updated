import numpy as np
import torch
import torch.nn.functional as F
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass
from transformers import AutoTokenizer, AutoModelForCausalLM
import json
import re
from collections import defaultdict
from sklearn.metrics import accuracy_score
import logging
from tqdm import tqdm
from pathlib import Path
import pandas as pd
import os
import random

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class LeakageDetectionResult:
    """Results from leakage detection analysis"""
    perplexity_original: float
    perplexity_paraphrased: float
    perplexity_ratio: float
    ngram_accuracy_original: Dict[int, float]
    ngram_accuracy_paraphrased: Dict[int, float]
    ngram_accuracy_diff: Dict[int, float]
    suspicious_samples: List[Dict[str, Any]]
    leakage_probability: float
    confidence_score: float

@dataclass
class BenchmarkResult:
    """Standard benchmark evaluation results"""
    accuracy: float
    f1_score: float
    exact_match: float
    rouge_l: float
    perplexity: float
    predictions: List[str]
    ground_truth: List[str]

class LeakageDetector:
    """
    Core leakage detection system based on BenBench methodology
    Uses perplexity and n-gram accuracy to detect potential data leakage
    """
    
    def __init__(self, model_name: str, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        self.model_name = model_name
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)
        self.model.to(device)
        self.model.eval()
        
        # Add padding token if missing
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def calculate_perplexity(self, texts: List[str]) -> float:
        """Calculate perplexity for a list of texts"""
        total_loss = 0
        total_tokens = 0
        
        for text in tqdm(texts, desc="Calculating perplexity"):
            inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model(**inputs, labels=inputs["input_ids"])
                loss = outputs.loss
                total_loss += loss.item() * inputs["input_ids"].size(1)
                total_tokens += inputs["input_ids"].size(1)
        
        avg_loss = total_loss / total_tokens
        return torch.exp(torch.tensor(avg_loss)).item()
    
    def calculate_ngram_accuracy(self, texts: List[str], n_values: List[int] = [3, 4, 5]) -> Dict[int, float]:
        """Calculate n-gram prediction accuracy"""
        ngram_accuracies = {n: [] for n in n_values}
        
        for text in tqdm(texts, desc="Calculating n-gram accuracy"):
            tokens = self.tokenizer.encode(text, add_special_tokens=False)
            
            for n in n_values:
                if len(tokens) < n + 1:
                    continue
                
                correct_predictions = 0
                total_predictions = 0
                
                for i in range(len(tokens) - n):
                    context = tokens[i:i+n]
                    target = tokens[i+n]
                    
                    # Get model prediction
                    context_tensor = torch.tensor([context]).to(self.device)
                    
                    with torch.no_grad():
                        outputs = self.model(context_tensor)
                        logits = outputs.logits[0, -1, :]
                        predicted_token = torch.argmax(logits).item()
                    
                    if predicted_token == target:
                        correct_predictions += 1
                    total_predictions += 1
                
                if total_predictions > 0:
                    accuracy = correct_predictions / total_predictions
                    ngram_accuracies[n].append(accuracy)
        
        # Calculate average accuracy for each n-gram size
        return {n: np.mean(accuracies) if accuracies else 0.0 
                for n, accuracies in ngram_accuracies.items()}
    
    def detect_leakage(self, 
                      original_dataset: List[str], 
                      paraphrased_dataset: List[str],
                      threshold_perplexity: float = 1.2,
                      threshold_ngram: float = 0.1) -> LeakageDetectionResult:
        """
        Main leakage detection method
        Compares original vs paraphrased versions to detect suspicious patterns
        """
        
        logger.info("Starting leakage detection analysis...")
        
        # Calculate perplexity for both datasets
        ppl_original = self.calculate_perplexity(original_dataset)
        ppl_paraphrased = self.calculate_perplexity(paraphrased_dataset)
        ppl_ratio = ppl_paraphrased / ppl_original if ppl_original > 0 else float('inf')
        
        # Calculate n-gram accuracy for both datasets
        ngram_original = self.calculate_ngram_accuracy(original_dataset)
        ngram_paraphrased = self.calculate_ngram_accuracy(paraphrased_dataset)
        
        # Calculate differences
        ngram_diff = {n: ngram_original[n] - ngram_paraphrased[n] 
                     for n in ngram_original.keys()}
        
        # Find suspicious samples
        suspicious_samples = self._find_suspicious_samples(
            original_dataset, paraphrased_dataset, ngram_original, ngram_paraphrased
        )
        
        # Calculate leakage probability
        leakage_prob = self._calculate_leakage_probability(
            ppl_ratio, ngram_diff, threshold_perplexity, threshold_ngram
        )
        
        # Calculate confidence score
        confidence = self._calculate_confidence_score(ppl_ratio, ngram_diff)
        
        return LeakageDetectionResult(
            perplexity_original=ppl_original,
            perplexity_paraphrased=ppl_paraphrased,
            perplexity_ratio=ppl_ratio,
            ngram_accuracy_original=ngram_original,
            ngram_accuracy_paraphrased=ngram_paraphrased,
            ngram_accuracy_diff=ngram_diff,
            suspicious_samples=suspicious_samples,
            leakage_probability=leakage_prob,
            confidence_score=confidence
        )
    
    def _find_suspicious_samples(self, 
                                original: List[str], 
                                paraphrased: List[str],
                                ngram_orig: Dict[int, float],
                                ngram_para: Dict[int, float]) -> List[Dict[str, Any]]:
        """Identify samples with suspicious leakage patterns"""
        suspicious = []
        
        for i, (orig_text, para_text) in enumerate(zip(original, paraphrased)):
            # Calculate per-sample metrics
            orig_ppl = self.calculate_perplexity([orig_text])
            para_ppl = self.calculate_perplexity([para_text])
            
            # Flag if original has much lower perplexity
            if para_ppl / orig_ppl > 2.0:
                suspicious.append({
                    'index': i,
                    'original_text': orig_text,
                    'paraphrased_text': para_text,
                    'original_perplexity': orig_ppl,
                    'paraphrased_perplexity': para_ppl,
                    'perplexity_ratio': para_ppl / orig_ppl,
                    'reason': 'High perplexity ratio'
                })
        
        return suspicious
    
    def _calculate_leakage_probability(self, 
                                     ppl_ratio: float,
                                     ngram_diff: Dict[int, float],
                                     threshold_ppl: float,
                                     threshold_ngram: float) -> float:
        """Calculate probability of data leakage"""
        
        # Perplexity-based evidence
        ppl_evidence = max(0, threshold_ppl - ppl_ratio) / threshold_ppl
        
        # N-gram accuracy evidence
        ngram_evidence = np.mean([max(0, diff - threshold_ngram) / threshold_ngram 
                                 for diff in ngram_diff.values()])
        
        # Combine evidence
        combined_evidence = (ppl_evidence + ngram_evidence) / 2
        
        # Convert to probability using sigmoid
        return 1 / (1 + np.exp(-5 * (combined_evidence - 0.5)))
    
    def _calculate_confidence_score(self, ppl_ratio: float, ngram_diff: Dict[int, float]) -> float:
        """Calculate confidence in leakage detection"""
        
        # Higher confidence when metrics are more extreme
        ppl_confidence = min(1.0, abs(ppl_ratio - 1.0))
        ngram_confidence = min(1.0, np.mean([abs(diff) for diff in ngram_diff.values()]))
        
        return (ppl_confidence + ngram_confidence) / 2

class DatasetParaphraser:
    """
    Paraphrases datasets to create reference versions for leakage detection
    """
    
    def __init__(self, paraphrase_model: str = "microsoft/DialoGPT-medium"):
        self.paraphrase_model = paraphrase_model
        # In practice, you'd use a dedicated paraphrasing model
        # For now, we'll use simple text transformations
    
    def paraphrase_dataset(self, texts: List[str]) -> List[str]:
        """Create paraphrased versions of input texts"""
        paraphrased = []
        
        for text in texts:
            # Simple paraphrasing strategies
            paraphrased_text = self._simple_paraphrase(text)
            paraphrased.append(paraphrased_text)
        
        return paraphrased
    
    def _simple_paraphrase(self, text: str) -> str:
        """Simple rule-based paraphrasing"""
        # Replace common words with synonyms
        replacements = {
            "the": "a",
            "and": "plus",
            "or": "alternatively",
            "but": "however",
            "because": "since",
            "therefore": "thus",
            "answer": "solution",
            "question": "problem",
            "find": "determine",
            "calculate": "compute"
        }
        
        result = text
        for old, new in replacements.items():
            result = re.sub(r'\b' + old + r'\b', new, result, flags=re.IGNORECASE)
        
        # Add minor structural changes
        if "?" in result:
            result = result.replace("?", ".")
            if not result.startswith("Find"):
                result = "Find " + result.lower()
        
        return result

class EnhancedLLMEvaluator:
    """
    Enhanced LLM evaluation framework with benchmarking and leakage detection
    """
    
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.leakage_detector = LeakageDetector(model_name)
        self.paraphraser = DatasetParaphraser()
        self.benchmarks = {}
        
    def add_benchmark(self, name: str, dataset: Dict[str, List[str]]):
        """Add a benchmark dataset"""
        self.benchmarks[name] = dataset
        
    def evaluate_with_leakage_detection(self, 
                                      benchmark_name: str,
                                      detect_leakage: bool = True) -> Dict[str, Any]:
        """
        Comprehensive evaluation including leakage detection
        """
        
        if benchmark_name not in self.benchmarks:
            raise ValueError(f"Benchmark {benchmark_name} not found")
        
        dataset = self.benchmarks[benchmark_name]
        results = {}
        
        # Standard benchmark evaluation
        benchmark_result = self._evaluate_benchmark(dataset)
        results['benchmark'] = benchmark_result
        
        # Leakage detection
        if detect_leakage:
            logger.info(f"Performing leakage detection for {benchmark_name}...")
            
            # Create paraphrased version
            original_texts = dataset.get('texts', [])
            paraphrased_texts = self.paraphraser.paraphrase_dataset(original_texts)
            
            # Detect leakage
            leakage_result = self.leakage_detector.detect_leakage(
                original_texts, paraphrased_texts
            )
            results['leakage_detection'] = leakage_result
            
            # Generate report
            report = self._generate_leakage_report(leakage_result)
            results['leakage_report'] = report
        
        return results
    
    def _evaluate_benchmark(self, dataset: Dict[str, List[str]]) -> BenchmarkResult:
        """Standard benchmark evaluation"""
        # Placeholder for actual benchmark evaluation
        # In practice, you'd implement specific evaluation logic for each benchmark
        
        return BenchmarkResult(
            accuracy=0.85,
            f1_score=0.82,
            exact_match=0.78,
            rouge_l=0.88,
            perplexity=15.2,
            predictions=["pred1", "pred2"],
            ground_truth=["truth1", "truth2"]
        )
    
    def _generate_leakage_report(self, leakage_result: LeakageDetectionResult) -> Dict[str, Any]:
        """Generate comprehensive leakage detection report"""
        
        report = {
            'summary': {
                'leakage_probability': leakage_result.leakage_probability,
                'confidence_score': leakage_result.confidence_score,
                'risk_level': self._classify_risk_level(leakage_result.leakage_probability)
            },
            'metrics': {
                'perplexity_analysis': {
                    'original': leakage_result.perplexity_original,
                    'paraphrased': leakage_result.perplexity_paraphrased,
                    'ratio': leakage_result.perplexity_ratio,
                    'interpretation': self._interpret_perplexity_ratio(leakage_result.perplexity_ratio)
                },
                'ngram_analysis': {
                    'original': leakage_result.ngram_accuracy_original,
                    'paraphrased': leakage_result.ngram_accuracy_paraphrased,
                    'differences': leakage_result.ngram_accuracy_diff,
                    'interpretation': self._interpret_ngram_differences(leakage_result.ngram_accuracy_diff)
                }
            },
            'suspicious_samples': {
                'count': len(leakage_result.suspicious_samples),
                'samples': leakage_result.suspicious_samples[:5],  # Top 5
                'patterns': self._analyze_suspicious_patterns(leakage_result.suspicious_samples)
            },
            'recommendations': self._generate_recommendations(leakage_result)
        }
        
        return report
    
    def _classify_risk_level(self, probability: float) -> str:
        """Classify leakage risk level"""
        if probability >= 0.8:
            return "HIGH"
        elif probability >= 0.5:
            return "MEDIUM"
        elif probability >= 0.2:
            return "LOW"
        else:
            return "MINIMAL"
    
    def _interpret_perplexity_ratio(self, ratio: float) -> str:
        """Interpret perplexity ratio"""
        if ratio > 2.0:
            return "Strong evidence of leakage - much higher perplexity on paraphrased text"
        elif ratio > 1.5:
            return "Moderate evidence of leakage - higher perplexity on paraphrased text"
        elif ratio > 1.2:
            return "Weak evidence of leakage - slightly higher perplexity on paraphrased text"
        else:
            return "No evidence of leakage from perplexity analysis"
    
    def _interpret_ngram_differences(self, differences: Dict[int, float]) -> str:
        """Interpret n-gram accuracy differences"""
        max_diff = max(differences.values())
        if max_diff > 0.3:
            return "Strong evidence of leakage - much higher n-gram accuracy on original text"
        elif max_diff > 0.2:
            return "Moderate evidence of leakage - higher n-gram accuracy on original text"
        elif max_diff > 0.1:
            return "Weak evidence of leakage - slightly higher n-gram accuracy on original text"
        else:
            return "No evidence of leakage from n-gram analysis"
    
    def _analyze_suspicious_patterns(self, samples: List[Dict[str, Any]]) -> List[str]:
        """Analyze patterns in suspicious samples"""
        patterns = []
        
        if samples:
            avg_ratio = np.mean([s['perplexity_ratio'] for s in samples])
            patterns.append(f"Average perplexity ratio: {avg_ratio:.2f}")
            
            reasons = [s['reason'] for s in samples]
            reason_counts = {reason: reasons.count(reason) for reason in set(reasons)}
            patterns.append(f"Most common pattern: {max(reason_counts, key=reason_counts.get)}")
        
        return patterns
    
    def _generate_recommendations(self, leakage_result: LeakageDetectionResult) -> List[str]:
        """Generate recommendations based on leakage detection results"""
        recommendations = []
        
        if leakage_result.leakage_probability > 0.7:
            recommendations.append("High risk of data leakage detected. Consider excluding this model from fair comparisons.")
            recommendations.append("Investigate training data sources and procedures.")
            recommendations.append("Consider using alternative evaluation methods.")
        
        elif leakage_result.leakage_probability > 0.4:
            recommendations.append("Moderate risk of data leakage. Use results with caution.")
            recommendations.append("Consider additional validation with unseen data.")
            recommendations.append("Document potential leakage in model evaluation reports.")
        
        else:
            recommendations.append("Low risk of data leakage detected.")
            recommendations.append("Results appear reliable for fair comparison.")
        
        if leakage_result.suspicious_samples:
            recommendations.append(f"Investigate {len(leakage_result.suspicious_samples)} suspicious samples manually.")
        
        return recommendations

def run_benchmark_and_leakage_detection(model_name: str, dataset_texts: List[str]) -> Dict[str, Any]:
    """
    Runs benchmark and leakage detection for a given model and dataset.
    """
    logger.info(f"Starting benchmark and leakage detection for model: {model_name}")
    evaluator = EnhancedLLMEvaluator(model_name)
    
    # Prepare the dataset for the evaluator
    prepared_dataset = {'texts': dataset_texts}
    evaluator.add_benchmark("mmlu_dataset", prepared_dataset)
    
    # Run evaluation with leakage detection
    results = evaluator.evaluate_with_leakage_detection("mmlu_dataset", detect_leakage=True)
    
    logger.info(f"Finished benchmark and leakage detection for model: {model_name}")
    return results

class MMLUDatasetProcessor:
    def __init__(self):
        self.dataset_path = None
        self.subjects = []
        self.data = {}
        
    def load_mmlu_data(self, config_path='config.json'):
        """Load MMLU dataset from CSV files based on a configuration file."""
        # Load configuration from JSON file
        with open(config_path, 'r') as f:
            config = json.load(f)['data_loader']

        directory = config.get('directory', 'raw_data')
        self.output_directory = Path(config.get('output_directory', 'processed_data'))
        file_pattern = config.get('file_pattern', '*.csv')
        has_header = config.get('has_header', False)
        column_config = config['column_config']
        columns = column_config['with_header'] if has_header else column_config['no_header']

        encoding = config.get('encoding', 'utf-8')

        script_dir = Path(__file__).parent
        self.dataset_path = script_dir.parent / directory
        
        # Find all files matching the pattern
        for subject_file in self.dataset_path.glob(file_pattern):
            subject_name = subject_file.stem
            
            # Read CSV with or without header
            if has_header:
                df = pd.read_csv(subject_file, encoding=encoding)
            else:
                df = pd.read_csv(subject_file, header=None, encoding=encoding)

            self.data[subject_name] = []
            for _, row in df.iterrows():
                try:
                    # Decode data to UTF-8 to ensure it is human-readable
                    question_data = {
                        'question': str(row[columns['question']]),
                        'choices': {
                            '1': str(row[columns['choice_1']]),
                            '2': str(row[columns['choice_2']]),
                            '3': str(row[columns['choice_3']]),
                            '4': str(row[columns['choice_4']]),
                        },
                        'answer': str(row[columns['answer']]),
                        'subject': subject_name
                    }
                    self.data[subject_name].append(question_data)
                except (IndexError, KeyError) as e:
                    print(f"Skipping row in {subject_file.name} due to missing data: {e}")

        self.subjects = list(self.data.keys())
        if self.subjects:
            print(f"Loaded {len(self.subjects)} subjects")
            # Print first 10 samples from the first loaded subject for verification
            first_subject = self.subjects[0]
            print(f"Data Samples from '{first_subject}': {self.data[first_subject][:10]}")
        else:
            print("No subjects loaded. Please check the configuration and data files.")

    def format_question_for_api(self, question_data: Dict, few_shot_examples: List[Dict] = None) -> str:
        """Format question for API call with optional few-shot examples"""
        prompt = ""
        
        # Add few-shot examples if provided
        if few_shot_examples:
            prompt += "Here are some example questions:\n\n"
            for example in few_shot_examples:
                prompt += self._format_single_question(example)
                prompt += f"Answer: {example['answer']}\n\n"
        
        # Add the actual question
        prompt += "Question:\n"
        prompt += self._format_single_question(question_data)
        prompt += "Answer:"
        
        return prompt
    
    def _format_single_question(self, question_data: Dict) -> str:
        """Format a single question"""
        formatted = f"{question_data['question']}\n"
        for letter, choice in question_data['choices'].items():
            formatted += f"{letter}) {choice}\n"
        return formatted
    
    def get_few_shot_examples(self, subject: str, n_examples: int = 5) -> List[Dict]:
        """Get few-shot examples from training data"""
        if subject in self.data and len(self.data[subject]) > n_examples:
            return random.sample(self.data[subject], n_examples)
        return []
    
    def save_results(self, results: Dict[str, Any], output_path: str):
        """Save evaluation results to JSON file"""
        # Ensure the output directory exists
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"Results saved to {output_path}")


if __name__ == "__main__":
    # Load configuration from JSON file
    script_dir = os.path.dirname(__file__)
    config_path = os.path.join(script_dir, '..', 'config.json')
    with open(config_path, 'r') as f:
        config = json.load(f)

    hf_models_config = config.get('huggingface_models', {})
    
    if not hf_models_config:
        print("No HuggingFace models configured in config.json. Exiting.")
        exit()

    print("Available HuggingFace models for benchmark and leakage detection:")
    for i, (model_name, model_config) in enumerate(hf_models_config.items()):
        print(f"{i+1}. {model_name} (Base: {model_config['base_model_id']})")

    while True:
        try:
            choice = input("Select a HuggingFace model to evaluate (number): ").strip()
            model_index = int(choice) - 1
            if 0 <= model_index < len(hf_models_config):
                selected_model_name = list(hf_models_config.keys())[model_index]
                selected_model_config = hf_models_config[selected_model_name]
                base_model_id = selected_model_config['base_model_id']
                break
            else:
                print("Invalid choice. Please enter a valid number.")
        except ValueError:
            print("Invalid input. Please enter a number.")

    # Initialize dataset processor and load MMLU data
    dataset_processor = MMLUDatasetProcessor()
    dataset_processor.load_mmlu_data(config_path)

    # Prepare dataset texts for benchmark and leakage detection
    mmlu_questions = []
    for subject_data in dataset_processor.data.values():
        for q_data in subject_data:
            mmlu_questions.append(q_data['question'])

    # Run benchmark and leakage detection
    try:
        print(f"\n--- Running Benchmark and Leakage Detection for {selected_model_name} ---")
        benchmark_leakage_results = run_benchmark_and_leakage_detection(base_model_id, mmlu_questions)
        print("Benchmark and Leakage Detection Results:")
        if benchmark_leakage_results and 'leakage_detection' in benchmark_leakage_results:
            print(f"  Leakage Probability: {benchmark_leakage_results['leakage_detection'].leakage_probability:.3f}")
            print(f"  Confidence Score: {benchmark_leakage_results['leakage_detection'].confidence_score:.3f}")
            print(f"  Risk Level: {benchmark_leakage_results['leakage_report']['summary']['risk_level']}")
            print("  Recommendations:")
            for rec in benchmark_leakage_results['leakage_report']['recommendations']:
                print(f"    - {rec}")
        else:
            print("  Leakage detection results not available.")
    except Exception as e:
        print(f"Error during benchmark and leakage detection for {selected_model_name}: {e}")