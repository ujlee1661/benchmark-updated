import pandas as pd
import json
import requests
import time
import random
from typing import List, Dict, Any
import os
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()
from mmlu_evaluator import MMLUEvaluator
from azure_evaluator import AzureLMAPIEvaluator
from hf_evaluator import HuggingFaceEvaluator
from compare_results import main as compare_results_main
from benchmark_leakage import run_benchmark_and_leakage_detection

class MMLUDatasetProcessor:
    def __init__(self):
        self.dataset_path = None
        self.subjects = []
        self.data = {}
        
    def load_mmlu_data(self, config_path='config.json'):
        """Load MMLU dataset from CSV files based on a configuration file."""
        # Load configuration from JSON file
        with open(config_path, 'r', encoding='utf-8') as f:  # UTF-8 인코딩 지정
            config = json.load(f)['data_loader']

        directory = config.get('directory', 'raw_data')
        self.output_directory = Path(config.get('output_directory', 'processed_data'))
        file_pattern = config.get('file_pattern', '*.csv')
        has_header = config.get('has_header', False)
        column_config = config['column_config']
        columns = column_config['with_header'] if has_header else column_config['no_header']

        encoding = config.get('encoding', 'utf-8')

        self.dataset_path = Path(directory)
        
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
            first_subject = self.subjects[0]
            print(f"Data Samples from '{first_subject}': {self.data[first_subject][:10]}")
        else:
            print("No subjects loaded. Please check the configuration and data files.")
        
    def format_question_for_api(self, question_data: Dict, few_shot_examples: List[Dict] = None) -> str:
        """Format question for API call with optional few-shot examples"""
        prompt = ""
        if few_shot_examples:
            prompt += "Here are some example questions:\n\n"
            for example in few_shot_examples:
                prompt += self._format_single_question(example)
                prompt += f"Answer: {example['answer']}\n\n"
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
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"Results saved to {output_path}")

def evaluate_with_huggingface_model(config: Dict):
    hf_models_config = config.get('huggingface_models', {})
    if not hf_models_config:
        print("No HuggingFace models configured in config.json. Skipping HuggingFace evaluation.")
        return

    print("Available HuggingFace models:")
    for i, (model_name, model_config) in enumerate(hf_models_config.items()):
        print(f"{i+1}. {model_name} (Base: {model_config['base_model_id']}, Adapter: {model_config['adapter_model_id']})")

    while True:
        try:
            choice = input("Select a HuggingFace model to evaluate (number): ").strip()
            model_index = int(choice) - 1
            if 0 <= model_index < len(hf_models_config):
                selected_model_name = list(hf_models_config.keys())[model_index]
                selected_model_config = hf_models_config[selected_model_name]
                base_model_id = selected_model_config['base_model_id']
                adapter_model_id = selected_model_config['adapter_model_id']
                break
            else:
                print("Invalid choice. Please enter a valid number.")
        except ValueError:
            print("Invalid input. Please enter a number.")

    dataset_processor = MMLUDatasetProcessor()
    dataset_processor.load_mmlu_data('config.json')
    
    print(f"Initializing HuggingFace model: {selected_model_name}...")
    hf_evaluator = HuggingFaceEvaluator(
        base_model_id=base_model_id,
        adapter_model_id=adapter_model_id,
        max_new_tokens=10,
        temperature=0.1,
        use_4bit=True
    )
    
    try:
        mmlu_eval_hf = MMLUEvaluator(dataset_processor, hf_evaluator)
        print("Starting MMLU evaluation with HuggingFace model...")
        all_results = {}
        for subject in dataset_processor.subjects:
            print(f"--- Evaluating subject: {subject} ---")
            results = mmlu_eval_hf.evaluate_subject(
                subject,
                few_shot=True,
                max_questions=100
            )
            all_results[subject] = results
            if 'accuracy' in results:
                print(f"Accuracy for {subject}: {results['accuracy']:.2%}")
            if 'total_questions' in results:
                print(f"Total questions evaluated for {subject}: {results['total_questions']}")

        output_file = dataset_processor.output_directory / f'mmlu_evaluation_{selected_model_name.replace("/", "_")}_results.json'
        dataset_processor.save_results(all_results, output_file)
            
    except Exception as e:
        print(f"Error during evaluation: {e}")
    finally:
        hf_evaluator.cleanup()

def evaluate_with_azure_model():
    API_KEY = os.getenv('AZURE_API_KEY')
    API_URL = os.getenv('AZURE_ENDPOINT')
    MODEL_NAME = os.getenv('AZURE_DEPLOYMENT_NAME')
    
    dataset_processor = MMLUDatasetProcessor()
    dataset_processor.load_mmlu_data('config.json')
    
    azure_evaluator = AzureLMAPIEvaluator(
        api_key=API_KEY,
        azure_endpoint=API_URL,
        deployment_name=MODEL_NAME
    )
    
    mmlu_eval_azure = MMLUEvaluator(dataset_processor, azure_evaluator)
    
    print("Starting MMLU evaluation with Azure model...")
    all_results = {}
    for subject in dataset_processor.subjects:
        print(f"--- Evaluating subject: {subject} ---")
        results = mmlu_eval_azure.evaluate_subject(
            subject, 
            few_shot=True, 
            max_questions=100
        )
        all_results[subject] = results
        if 'accuracy' in results:
            print(f"Accuracy for {subject}: {results['accuracy']:.2%}")
        if 'total_questions' in results:
            print(f"Total questions evaluated for {subject}: {results['total_questions']}")

    output_file = dataset_processor.output_directory / 'mmlu_evaluation_openai_4o_results.json'
    dataset_processor.save_results(all_results, output_file)
    compare_results_main()

def main():
    with open('config.json', 'r', encoding='utf-8') as f:  # UTF-8 인코딩 지정
        config = json.load(f)

    print("Choose evaluation method:")
    print("1. HuggingFace Models (Local)")
    print("2. Cloud Models (API)")
    print("3. Both")
    print("4. Run Comparison Table")
    print("5. Run Benchmark and Leakage Detection")
    
    choice = input("Enter your choice: ").strip()
    
    if choice == "1":
        evaluate_with_huggingface_model(config)
    elif choice == "2":
        evaluate_with_azure_model()
    elif choice == "3":
        print("Running both evaluations...")
        evaluate_with_huggingface_model(config)
        print("\n" + "="*50 + "\n")
        evaluate_with_azure_model()
    elif choice == "4":
        compare_results_main()
    elif choice == "5":
        with open('config.json', 'r', encoding='utf-8') as f:  # UTF-8 인코딩 지정
            config = json.load(f)

        hf_models_config = config.get('huggingface_models', {})
        
        if not hf_models_config:
            print("No HuggingFace models configured in config.json. Exiting.")
            return

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

        dataset_processor = MMLUDatasetProcessor()
        dataset_processor.load_mmlu_data('config.json')

        mmlu_questions = []
        for subject_data in dataset_processor.data.values():
            for q_data in subject_data:
                mmlu_questions.append(q_data['question'])

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
    else:
        print("Invalid choice. Running HuggingFace evaluation by default.")
        evaluate_with_huggingface_model(config)

if __name__ == "__main__":
    main()
1