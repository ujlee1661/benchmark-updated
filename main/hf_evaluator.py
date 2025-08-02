import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
import re
from typing import Dict, Any, Optional
import time

class HuggingFaceEvaluator:
    """Evaluator for Hugging Face models with LoRA adapters"""
    
    def __init__(self, 
                 base_model_id: str,
                 adapter_model_id: Optional[str] = None,
                 max_new_tokens: int = 10,
                 temperature: float = 0.1,
                 use_4bit: bool = True):
        """
        Initialize the Hugging Face model evaluator
        
        Args:
            base_model_id: Base model identifier from Hugging Face
            adapter_model_id: LoRA adapter model identifier (optional)
            max_new_tokens: Maximum number of new tokens to generate
            temperature: Sampling temperature
            use_4bit: Whether to use 4-bit quantization
        """
        self.base_model_id = base_model_id
        self.adapter_model_id = adapter_model_id
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.use_4bit = use_4bit
        
        # Load model and tokenizer
        self._load_model()
    
    def _load_model(self):
        """Load the model and tokenizer"""
        print(f"Loading tokenizer from {self.base_model_id}...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.base_model_id, 
            trust_remote_code=True
        )
        
        # Set pad token if not exists
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Setup quantization config if using 4-bit
        if self.use_4bit:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
        else:
            bnb_config = None
        
        print(f"Loading base model {self.base_model_id}...")
        self.model = AutoModelForCausalLM.from_pretrained(
            self.base_model_id,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.float16 if not self.use_4bit else None
        )
        
        # Load LoRA adapter if specified
        if self.adapter_model_id:
            print(f"Loading LoRA adapter {self.adapter_model_id}...")
            self.model = PeftModel.from_pretrained(self.model, self.adapter_model_id)
        
        # Set to evaluation mode
        self.model.eval()
        print("Model loaded successfully!")
    
    def generate_response(self, prompt: str) -> str:
        """
        Generate response from the model
        
        Args:
            prompt: Input prompt
            
        Returns:
            Generated response text
        """
        try:
            # Tokenize input
            inputs = self.tokenizer(
                prompt, 
                return_tensors="pt", 
                truncation=True, 
                max_length=2048
            ).to(self.model.device)
            
            # Generate response
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    temperature=self.temperature,
                    do_sample=True if self.temperature > 0 else False,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    repetition_penalty=1.1
                )
            
            # Decode only the new tokens (exclude input prompt)
            input_length = inputs['input_ids'].shape[1]
            generated_tokens = outputs[0][input_length:]
            response = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
            
            return response.strip()
            
        except Exception as e:
            print(f"Error generating response: {e}")
            return ""
    
    def evaluate_question(self, question_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate a single question
        
        Args:
            question_data: Dictionary containing question, choices, and answer
            
        Returns:
            Dictionary with prediction and correctness
        """
        # Format the prompt for multiple choice
        prompt = self._format_mcq_prompt(question_data)
        
        # Generate response
        response = self.generate_response(prompt)
        
        # Extract the predicted answer
        predicted_answer = self._extract_answer(response)
        
        # Check if correct
        correct_answer = str(question_data['answer'])
        is_correct = predicted_answer == correct_answer
        
        return {
            'question': question_data['question'],
            'correct_answer': correct_answer,
            'predicted_answer': predicted_answer,
            'raw_response': response,
            'is_correct': is_correct,
            'choices': question_data['choices']
        }
    
    def _format_mcq_prompt(self, question_data: Dict[str, Any]) -> str:
        """Format question as multiple choice prompt"""
        prompt = f"Question: {question_data['question']}\n\n"
        
        for key, choice in question_data['choices'].items():
            prompt += f"{key}) {choice}\n"
        
        prompt += "\nAnswer with only the number (1, 2, 3, or 4): "
        return prompt
    
    def _extract_answer(self, response: str) -> str:
        """Extract answer choice from model response"""
        # Clean the response
        response = response.strip().lower()
        
        # Look for patterns like "1)", "1.", "1", "(1)", etc.
        patterns = [
            r'\b([1-4])\)',  # 1), 2), etc.
            r'\b([1-4])\.',  # 1., 2., etc.
            r'\(([1-4])\)',  # (1), (2), etc.
            r'\b([1-4])\b',  # Just the number
        ]
        
        for pattern in patterns:
            match = re.search(pattern, response)
            if match:
                return match.group(1)
        
        # If no clear pattern found, try to find any digit 1-4
        digits = re.findall(r'[1-4]', response)
        if digits:
            return digits[0]
        
        # Default return
        return "1"  # Default to choice 1 if no answer found
    
    def batch_evaluate(self, questions: list) -> list:
        """
        Evaluate multiple questions
        
        Args:
            questions: List of question dictionaries
            
        Returns:
            List of evaluation results
        """
        results = []
        total_questions = len(questions)
        
        for i, question in enumerate(questions):
            print(f"Evaluating question {i+1}/{total_questions}")
            result = self.evaluate_question(question)
            results.append(result)
            
            # Optional: Add small delay to prevent overheating
            time.sleep(0.1)
        
        return results
    
    def cleanup(self):
        """Clean up GPU memory"""
        if hasattr(self, 'model'):
            del self.model
        if hasattr(self, 'tokenizer'):
            del self.tokenizer
        torch.cuda.empty_cache()