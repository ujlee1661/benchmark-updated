import time
from typing import Dict, Any, Union

class MMLUEvaluator:
    def __init__(self, dataset_processor, evaluator: Union['LMAPIEvaluator', 'AzureLMAPIEvaluator', 'HuggingFaceEvaluator']):
        """
        Initialize MMLU Evaluator
        
        Args:
            dataset_processor: MMLUDatasetProcessor instance
            evaluator: LMAPIEvaluator, AzureLMAPIEvaluator, or HuggingFaceEvaluator instance
        """
        self.dataset_processor = dataset_processor
        self.evaluator = evaluator
        self.results = {}
        
        # Detect evaluator type for any specific handling if needed
        self.evaluator_type = type(evaluator).__name__
        
        # Check if evaluator has the new interface (HuggingFaceEvaluator)
        self.is_hf_evaluator = hasattr(evaluator, 'evaluate_question')
    
    def evaluate_subject(self, subject: str, few_shot: bool = True, n_examples: int = 5,
                        max_questions: int = None) -> Dict[str, Any]:
        """Evaluate model on a specific subject"""
        if subject not in self.dataset_processor.data:
            raise ValueError(f"Subject {subject} not found in dataset")
        
        questions = self.dataset_processor.data[subject]
        if max_questions:
            questions = questions[:max_questions]
        
        # Get few-shot examples if needed
        few_shot_examples = []
        if few_shot:
            few_shot_examples = self.dataset_processor.get_few_shot_examples(subject, n_examples)
        
        correct = 0
        total = 0
        detailed_results = []
        
        print(f"Evaluating {subject} with {len(questions)} questions using {self.evaluator_type}...")
        
        for i, question_data in enumerate(questions):
            if self.is_hf_evaluator:
                # Use HuggingFace evaluator interface
                result = self._evaluate_question_hf(question_data, few_shot_examples if few_shot else None)
            else:
                # Use API evaluator interface (OpenAI/Azure)
                result = self._evaluate_question_api(question_data, few_shot_examples if few_shot else None)
            
            is_correct = result['is_correct']
            if is_correct:
                correct += 1
            total += 1
            
            # Add question index for tracking
            result['question_id'] = i
            detailed_results.append(result)
            
            # Rate limiting based on evaluator type
            if self.evaluator_type == 'AzureLMAPIEvaluator':
                time.sleep(0.2)  # More conservative for Azure
            elif self.evaluator_type == 'LMAPIEvaluator':
                time.sleep(0.1)  # Standard for OpenAI API
            elif self.evaluator_type == 'HuggingFaceEvaluator':
                time.sleep(0.05)  # Minimal delay for local models
            
            if (i + 1) % 10 == 0:
                print(f"Progress: {i + 1}/{len(questions)} ({correct}/{total} correct, {correct/total:.1%})")
        
        accuracy = correct / total if total > 0 else 0
        
        results = {
            'subject': subject,
            'accuracy': accuracy,
            'correct': correct,
            'total': total,
            'few_shot': few_shot,
            'n_examples': n_examples if few_shot else 0,
            'evaluator_type': self.evaluator_type,
            'detailed_results': detailed_results
        }
        
        print(f"Final Results for {subject}: {accuracy:.1%} accuracy ({correct}/{total})")
        return results
    
    def _evaluate_question_hf(self, question_data: Dict[str, Any], few_shot_examples: list = None) -> Dict[str, Any]:
        """Evaluate single question using HuggingFace evaluator"""
        # For HuggingFace evaluator, we can add few-shot examples to the question data if needed
        if few_shot_examples:
            # Modify the question to include few-shot examples
            modified_question = self._add_few_shot_to_question(question_data, few_shot_examples)
            result = self.evaluator.evaluate_question(modified_question)
        else:
            result = self.evaluator.evaluate_question(question_data)
        
        # Ensure consistent format with API evaluators
        return {
            'question': result['question'],
            'correct_answer': str(result['correct_answer']),
            'predicted_answer': str(result['predicted_answer']),
            'is_correct': result['is_correct'],
            'raw_response': result.get('raw_response', ''),
            'choices': result.get('choices', question_data.get('choices', {}))
        }
    
    def _evaluate_question_api(self, question_data: Dict[str, Any], few_shot_examples: list = None) -> Dict[str, Any]:
        """Evaluate single question using API evaluator (OpenAI/Azure)"""
        # Format prompt for API
        prompt = self.dataset_processor.format_question_for_api(
            question_data, few_shot_examples
        )
        
        # Query model
        response = self.evaluator.query_model(prompt)
        predicted_answer = self.evaluator.extract_answer(response)
        correct_answer = str(question_data['answer'])
        
        is_correct = str(predicted_answer) == correct_answer
        
        return {
            'question': question_data['question'],
            'correct_answer': correct_answer,
            'predicted_answer': str(predicted_answer),
            'is_correct': is_correct,
            'raw_response': response,
            'choices': question_data.get('choices', {})
        }
    
    def _add_few_shot_to_question(self, question_data: Dict[str, Any], few_shot_examples: list) -> Dict[str, Any]:
        """Add few-shot examples to the question for HuggingFace evaluator"""
        # Create a modified question that includes few-shot examples in the question text
        few_shot_text = "Here are some example questions:\n\n"
        
        for example in few_shot_examples:
            few_shot_text += f"Question: {example['question']}\n"
            for key, choice in example['choices'].items():
                few_shot_text += f"{key}) {choice}\n"
            few_shot_text += f"Answer: {example['answer']}\n\n"
        
        few_shot_text += "Now answer this question:\n\n"
        
        # Create modified question data
        modified_question = question_data.copy()
        modified_question['question'] = few_shot_text + question_data['question']
        
        return modified_question
    
    def evaluate_multiple_subjects(self, subjects: list, few_shot: bool = True, 
                                 n_examples: int = 5, max_questions: int = None) -> Dict[str, Any]:
        """Evaluate model on multiple subjects"""
        all_results = {}
        total_correct = 0
        total_questions = 0
        
        for subject in subjects:
            print(f"\n{'='*50}")
            print(f"Starting evaluation for subject: {subject}")
            print(f"{'='*50}")
            
            subject_results = self.evaluate_subject(
                subject, few_shot, n_examples, max_questions
            )
            all_results[subject] = subject_results
            total_correct += subject_results['correct']
            total_questions += subject_results['total']
            
            print(f"Completed {subject}: {subject_results['accuracy']:.1%} accuracy")
        
        overall_accuracy = total_correct / total_questions if total_questions > 0 else 0
        
        print(f"\n{'='*50}")
        print(f"OVERALL RESULTS")
        print(f"{'='*50}")
        print(f"Overall Accuracy: {overall_accuracy:.1%}")
        print(f"Total Correct: {total_correct}/{total_questions}")
        print(f"Subjects Evaluated: {len(subjects)}")
        print(f"Evaluator: {self.evaluator_type}")
        
        summary = {
            'overall_accuracy': overall_accuracy,
            'total_correct': total_correct,
            'total_questions': total_questions,
            'subjects_evaluated': len(subjects),
            'evaluator_type': self.evaluator_type,
            'subject_results': all_results
        }
        
        return summary
    


# Example usage with all three evaluator types:
"""
# Using with OpenAI API
openai_evaluator = LMAPIEvaluator(
    api_key="your-openai-key",
    api_url="https://api.openai.com/v1/chat/completions",
    model_name="gpt-3.5-turbo"
)

# Using with Azure OpenAI
azure_evaluator = AzureLMAPIEvaluator(
    api_key="your-azure-key",
    azure_endpoint="https://your-resource.openai.azure.com/",
    deployment_name="your-deployment-name"
)

# Using with HuggingFace model
hf_evaluator = HuggingFaceEvaluator(
    base_model_id="unsloth/qwen2-7b-bnb-4bit",
    adapter_model_id="gabi0215/kmmlu-Qwen2-7B"
)

# All can be used with the same MMLUEvaluator
mmlu_eval_openai = MMLUEvaluator(dataset_processor, openai_evaluator)
mmlu_eval_azure = MMLUEvaluator(dataset_processor, azure_evaluator)
mmlu_eval_hf = MMLUEvaluator(dataset_processor, hf_evaluator)

# Run evaluations
results_openai = mmlu_eval_openai.evaluate_subject("computer_science")
results_azure = mmlu_eval_azure.evaluate_subject("computer_science")
results_hf = mmlu_eval_hf.evaluate_subject("computer_science")
"""