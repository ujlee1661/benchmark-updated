import pandas as pd
import json
import random
import os
from pathlib import Path

from mmlu_evaluator import MMLUEvaluator
from hf_evaluator import HuggingFaceEvaluator
from azure_evaluator import AzureLMAPIEvaluator
from gemini_evaluator import GeminiAPIEvaluator
from compare_results import main as compare_results_main
from benchmark_leakage import run_benchmark_and_leakage_detection

# ✅ config.json 절대 경로
CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.json"

if not CONFIG_PATH.exists():
    print(f"❌ config.json not found at {CONFIG_PATH}")
    exit(1)

class MMLUDatasetProcessor:
    def __init__(self):
        self.dataset_path = None
        self.subjects = []
        self.data = {}
        
    def load_mmlu_data(self, config_path=CONFIG_PATH):
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)['data_loader']

        directory = config.get('directory', 'raw_data')
        self.output_directory = Path(config.get('output_directory', 'processed_data'))
        file_pattern = config.get('file_pattern', '*.csv')
        has_header = config.get('has_header', False)
        column_config = config['column_config']
        columns = column_config['with_header'] if has_header else column_config['no_header']
        encoding = config.get('encoding', 'utf-8')

        self.dataset_path = Path(directory)
        
        for subject_file in self.dataset_path.glob(file_pattern):
            subject_name = subject_file.stem
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
            print(f"Data Samples from '{first_subject}': {self.data[first_subject][:3]}")
        else:
            print("No subjects loaded. Please check the configuration and data files.")

    def get_few_shot_examples(self, subject: str, n_examples: int = 5):
        """Few-shot 예시 추출"""
        if subject in self.data and len(self.data[subject]) > n_examples:
            return random.sample(self.data[subject], n_examples)
        return []

    def save_results(self, results, output_path: str):
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"Results saved to {output_path}")

    def format_question_for_api(self, question_data: dict, few_shot_examples: list = None) -> str:
        """API 호출용 문제 포맷 생성 (few-shot 예시 포함 가능)"""
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

    def _format_single_question(self, question_data: dict) -> str:
        """단일 문제 포맷"""
        formatted = f"{question_data['question']}\n"
        for number, choice in question_data['choices'].items():
            formatted += f"{number}) {choice}\n"
        return formatted


# -------------------- 모델 실행 함수 --------------------
def evaluate_with_huggingface_model(config: dict):
    hf_models_config = config.get('huggingface_models', {})
    if not hf_models_config:
        print("No HuggingFace models configured in config.json.")
        return

    print("Available HuggingFace models:")
    for i, (model_name, model_config) in enumerate(hf_models_config.items()):
        print(f"{i+1}. {model_name} (Base: {model_config['base_model_id']})")

    choice = int(input("Select a HuggingFace model to evaluate (number): ").strip()) - 1
    selected_model_name = list(hf_models_config.keys())[choice]
    selected_model_config = hf_models_config[selected_model_name]

    dataset_processor = MMLUDatasetProcessor()
    dataset_processor.load_mmlu_data(CONFIG_PATH)
    
    print(f"Initializing HuggingFace model: {selected_model_name}...")
    hf_evaluator = HuggingFaceEvaluator(
        base_model_id=selected_model_config['base_model_id'],
        adapter_model_id=selected_model_config['adapter_model_id'],
        max_new_tokens=10,
        temperature=0.1,
        use_4bit=True
    )
    
    try:
        mmlu_eval_hf = MMLUEvaluator(dataset_processor, hf_evaluator)
        all_results = {}
        for subject in dataset_processor.subjects:
            results = mmlu_eval_hf.evaluate_subject(subject, few_shot=True, max_questions=100)
            all_results[subject] = results
        output_file = dataset_processor.output_directory / f'{selected_model_name.replace("/", "_")}_results.json'
        dataset_processor.save_results(all_results, output_file)
    finally:
        hf_evaluator.cleanup()

def evaluate_with_azure_model():
    API_KEY = os.getenv('AZURE_API_KEY')
    API_URL = os.getenv('AZURE_ENDPOINT')
    MODEL_NAME = os.getenv('AZURE_DEPLOYMENT_NAME')

    if not API_KEY or not API_URL or not MODEL_NAME:
        print("❌ Azure API config missing. Skipping.")
        return
    
    dataset_processor = MMLUDatasetProcessor()
    dataset_processor.load_mmlu_data(CONFIG_PATH)
    
    azure_evaluator = AzureLMAPIEvaluator(api_key=API_KEY, azure_endpoint=API_URL, deployment_name=MODEL_NAME)
    mmlu_eval_azure = MMLUEvaluator(dataset_processor, azure_evaluator)
    
    all_results = {}
    for subject in dataset_processor.subjects:
        results = mmlu_eval_azure.evaluate_subject(subject, few_shot=True, max_questions=100)
        all_results[subject] = results
    output_file = dataset_processor.output_directory / 'azure_results.json'
    dataset_processor.save_results(all_results, output_file)

def evaluate_with_gemini_model():
    # .env 없이 직접 키 입력 가능
    API_KEY = os.getenv('GEMINI_API_KEY') or "AIzaSyAsJDB8So6Lc7lNy-CAnxjDhUt4Hq8s2L0"
    if not API_KEY:
        print("❌ GEMINI_API_KEY not found.")
        return
    
    dataset_processor = MMLUDatasetProcessor()
    dataset_processor.load_mmlu_data(CONFIG_PATH)
    
    gemini_evaluator = GeminiAPIEvaluator(api_key=API_KEY)
    mmlu_eval_gemini = MMLUEvaluator(dataset_processor, gemini_evaluator)
    
    all_results = {}
    for subject in dataset_processor.subjects:
        print(f"--- Evaluating subject: {subject} ---")
        results = mmlu_eval_gemini.evaluate_subject(subject, few_shot=True, max_questions=100)
        all_results[subject] = results
        if 'accuracy' in results:
            print(f"Accuracy for {subject}: {results['accuracy']:.2%}")
    output_file = dataset_processor.output_directory / 'gemini_results.json'
    dataset_processor.save_results(all_results, output_file)

# -------------------- 메인 메뉴 --------------------
def main():
    with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
        config = json.load(f)

    print("Choose evaluation method:")
    print("1. HuggingFace Models (Local)")
    print("2. Azure OpenAI API")
    print("3. Google Gemini API")
    print("4. Both (HF + Azure)")
    print("5. Run Comparison Table")
    print("6. Run Benchmark and Leakage Detection")
    
    choice = input("Enter your choice: ").strip()
    
    if choice == "1":
        evaluate_with_huggingface_model(config)
    elif choice == "2":
        evaluate_with_azure_model()
    elif choice == "3":
        evaluate_with_gemini_model()
    elif choice == "4":
        evaluate_with_huggingface_model(config)
        print("\n" + "="*50 + "\n")
        evaluate_with_azure_model()
    elif choice == "5":
        compare_results_main()
    elif choice == "6":
        pass  # Benchmark/Leakage Detection 로직
    else:
        print("Invalid choice.")

if __name__ == "__main__":
    main()
