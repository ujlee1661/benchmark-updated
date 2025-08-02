import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

def clean_question_text(question: str) -> str:
    """Extract the actual question from the full question text."""
    try:
        # Find the last occurrence of 'Now answer this question:' to isolate the target question
        start_idx = question.rfind("Now answer this question:\n\n")
        if start_idx != -1:
            start_idx += len("Now answer this question:\n\n")
            question = question[start_idx:].strip()
        # Remove any trailing answer or choices
        end_idx = question.find("\nAnswer:")
        if end_idx != -1:
            question = question[:end_idx].strip()
        return question
    except Exception as e:
        print(f"Error cleaning question text: {e}")
        return question.strip()

def analyze_results(file_path: Path) -> Tuple[str, float, int, Dict]:
    """Analyze a single JSON result file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            results = json.load(f)
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None, 0.0, 0, defaultdict(lambda: defaultdict(int))

    model_name = file_path.stem.replace('mmlu_evaluation_', '').replace('_results', '')
    accuracy = 0.0
    total_questions = 0
    incorrect_answers = defaultdict(lambda: defaultdict(int))

    for subject, data in results.items():
        if not isinstance(data, dict) or data.get('total', 0) == 0:
            continue
        accuracy = data.get('accuracy', 0.0)
        total_questions = data.get('total', 0)

        detailed_results = data.get('detailed_results', [])
        for item in detailed_results:
            if not item.get('is_correct', True):
                question_text = item.get('question', '')
                predicted_answer = item.get('predicted_answer', 'N/A')
                correct_answer = item.get('correct_answer', 'N/A')
                cleaned_question = clean_question_text(question_text)
                answer_pair = f"Predicted: {predicted_answer}, Correct: {correct_answer}"
                incorrect_answers[cleaned_question][answer_pair] += 1

    return model_name, accuracy, total_questions, incorrect_answers

def main():
    processed_data_dir = Path('processed_data')
    result_files = list(processed_data_dir.glob('mmlu_evaluation_*.json'))

    if not result_files:
        print("No evaluation result files found in 'processed_data' directory.")
        return

    all_model_data = []
    for file_path in result_files:
        model_name, accuracy, total_questions, incorrect_answers = analyze_results(file_path)
        if model_name:
            all_model_data.append({
                'model_name': model_name,
                'accuracy': accuracy,
                'total_questions': total_questions,
                'incorrect_answers': incorrect_answers
            })

    if not all_model_data:
        print("No valid model data to process.")
        return

    # Sort by accuracy (highest first)
    all_model_data.sort(key=lambda x: x['accuracy'], reverse=True)

    # Calculate column widths for accuracy table
    max_model_name_len = max(len(d['model_name']) for d in all_model_data)
    max_accuracy_len = max(len(f"{d['accuracy']:.2%}") for d in all_model_data)
    max_total_questions_len = max(len(str(d['total_questions'])) for d in all_model_data)

    # Ensure minimum widths
    max_model_name_len = max(max_model_name_len, len("Model Name"))
    max_accuracy_len = max(max_accuracy_len, len("Accuracy"))
    max_total_questions_len = max(max_total_questions_len, len("Total Questions"))

    # Print accuracy table
    print("\n--- Model Comparison Table (Accuracy) ---")
    print(f"| {'Model Name':<{max_model_name_len}} | {'Accuracy':<{max_accuracy_len}} | {'Total Questions':<{max_total_questions_len}} |")
    print(f"| {'-' * max_model_name_len} | {'-' * max_accuracy_len} | {'-' * max_total_questions_len} |")
    for model_data in all_model_data:
        model_name = model_data['model_name']
        accuracy = f"{model_data['accuracy']:.2%}"
        total_questions = model_data['total_questions']
        print(f"| {model_name:<{max_model_name_len}} | {accuracy:<{max_accuracy_len}} | {total_questions:<{max_total_questions_len}} |")

    # Aggregate incorrect answers across models
    global_incorrect_questions = defaultdict(lambda: {'count': 0, 'answers': defaultdict(int), 'models': set()})
    for model_data in all_model_data:
        model_name = model_data['model_name']
        incorrect_answers = model_data['incorrect_answers']
        for question, answers in incorrect_answers.items():
            global_incorrect_questions[question]['count'] += sum(answers.values())
            global_incorrect_questions[question]['models'].add(model_name)
            for answer_pair, count in answers.items():
                global_incorrect_questions[question]['answers'][answer_pair] += count

    # Sort questions by the number of models that got them wrong
    sorted_incorrect_questions = sorted(
        global_incorrect_questions.items(),
        key=lambda x: x[1]['count'],
        reverse=True
    )

    # Print top 10 common incorrect questions
    print("\n--- Top 10 Common Incorrect Questions Across Models ---")
    if sorted_incorrect_questions:
        for i, (question, data) in enumerate(sorted_incorrect_questions[:10], 1):
            print(f"\n{i}. Question: {question}")
            print(f"   - Times Incorrect: {data['count']}")
            print(f"   - Models Incorrect: {', '.join(sorted(data['models']))}")
            print("   - Incorrect Answers:")
            for answer_pair, count in sorted(data['answers'].items(), key=lambda x: x[1], reverse=True):
                print(f"     - {answer_pair} ({count} times)")
    else:
        print("  No incorrect answers found across models.")

if __name__ == "__main__":
    main()