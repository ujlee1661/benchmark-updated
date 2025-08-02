import requests
import json

class AzureLMAPIEvaluator:
    def __init__(self, api_key: str, azure_endpoint: str, deployment_name: str, api_version: str = "2024-02-15-preview"):
        """
        Initialize Azure OpenAI API evaluator
        
        Args:
            api_key: Azure OpenAI API key
            azure_endpoint: Azure OpenAI endpoint URL (e.g., https://your-resource.openai.azure.com/)
            deployment_name: Name of your deployed model
            api_version: API version to use
        """
        self.api_key = api_key
        self.azure_endpoint = azure_endpoint.rstrip('/')
        self.deployment_name = deployment_name
        self.api_version = api_version
        
        self.api_url = f"{self.azure_endpoint}/openai/deployments/{self.deployment_name}/chat/completions"
        
        self.headers = {
            "api-key": api_key,
            "Content-Type": "application/json; charset=utf-8"  # Explicitly set charset
        }
    
    def query_model(self, prompt: str, max_tokens: int = 50, temperature: float = 0.0) -> str:
        """Query the Azure OpenAI model via API with improved error handling"""
        
        # Add system message for number-only responses
        messages = [
            {
                "role": "system", 
                "content": "You are a helpful assistant that answers multiple choice questions. Always respond with just the answer number (1, 2, 3, or 4) and nothing else. Do not use letters (A, B, C, D). Only use numbers 1, 2, 3, or 4. Handle Korean text properly."
            },
            {
                "role": "user", 
                "content": prompt
            }
        ]
        
        payload = {
            "messages": messages,
            "max_tokens": max_tokens,  # Increased from 10 to allow for proper responses
            "temperature": temperature,
            "top_p": 1.0,
            "frequency_penalty": 0,
            "presence_penalty": 0,
            "stop": ["Question:", "\n\nQuestion:"]  # Modified stop tokens
        }
        
        params = {
            "api-version": self.api_version
        }
        
        try:
            # Ensure proper encoding
            response = requests.post(
                self.api_url, 
                headers=self.headers, 
                json=payload,
                params=params,
                timeout=30  # Add timeout
            )
            
            print(f"API Response Status: {response.status_code}")  # Debug info
            
            response.raise_for_status()
            result = response.json()
            
            # More detailed error checking
            if 'choices' not in result:
                print(f"No 'choices' in response: {result}")
                return ""
            
            if len(result['choices']) == 0:
                print("Empty choices array")
                return ""
            
            if 'message' not in result['choices'][0]:
                print(f"No 'message' in choice: {result['choices'][0]}")
                return ""
            
            content = result['choices'][0]['message']['content']
            if content is None:
                print("Content is None")
                return ""
            
            return content.strip()
            
        except requests.exceptions.RequestException as e:
            print(f"API request failed: {e}")
            if hasattr(e, 'response') and e.response is not None:
                print(f"Response content: {e.response.text}")
            return ""
        except KeyError as e:
            print(f"Unexpected API response format: {e}")
            print(f"Full response: {response.text if 'response' in locals() else 'No response object'}")
            return ""
        except json.JSONDecodeError as e:
            print(f"JSON decode error: {e}")
            print(f"Response text: {response.text if 'response' in locals() else 'No response object'}")
            return ""
    
    def extract_answer(self, response: str) -> str:
        """Extract answer choice from model response - numbers only (1, 2, 3, 4)"""
        if not response:
            return "INVALID"
        
        response = response.strip()
        print(f"Raw response for extraction: '{response}'")  # Debug info
        
        # Look for patterns that contain numbers 1-4
        import re
        
        # Primary patterns - look for numbers 1-4 in various formats
        number_patterns = [
            r'\b([1-4])\)',  # 1)
            r'\b([1-4])\.',  # 1.
            r'\b([1-4]):',   # 1:
            r'\(([1-4])\)',  # (1)
            r'\[([1-4])\]',  # [1]
            r'^([1-4])\b',   # 1 at start
            r'\b([1-4])\b'   # Any standalone 1, 2, 3, 4
        ]
        
        for pattern in number_patterns:
            match = re.search(pattern, response)
            if match:
                return match.group(1)
        
        # Look for Korean answer patterns and convert to numbers
        korean_to_number = {'가': '1', '나': '2', '다': '3', '라': '4'}
        for korean, number in korean_to_number.items():
            if korean in response:
                return number
        
        # Look for letter patterns and convert to numbers
        letter_to_number = {'A': '1', 'B': '2', 'C': '3', 'D': '4',
                           'a': '1', 'b': '2', 'c': '3', 'd': '4'}
        response_upper = response.upper()
        for letter, number in letter_to_number.items():
            if letter.upper() in response_upper:
                return number
        
        # Last resort: look for any digit 1-4 in the response
        for char in response:
            if char in '1234':
                return char
        
        return "INVALID"
    
    def test_connection(self) -> bool:
        """Test if the API connection is working"""
        test_prompt = "Hello, please respond with just the number 1."
        response = self.query_model(test_prompt, max_tokens=10)
        print(f"Test response: '{response}'")
        return len(response) > 0

# Debug helper function
def debug_korean_encoding(text: str):
    """Helper function to debug Korean text encoding"""
    print(f"Original text: {text}")
    print(f"UTF-8 encoded: {text.encode('utf-8')}")
    print(f"Unicode escape: {repr(text)}")

# Example usage with debugging:
"""
# Test the evaluator
evaluator = AzureLMAPIEvaluator(
    api_key="your-key",
    azure_endpoint="your-endpoint", 
    deployment_name="your-deployment"
)

# Test connection first
if evaluator.test_connection():
    print("Connection successful!")
else:
    print("Connection failed!")

# Debug Korean text
korean_question = "안전교육 방법 중 TWI의 교육과정이 아닌 것은?"
debug_korean_encoding(korean_question)

# Test answer extraction with numbers
test_responses = ["1", "답: 2", "정답은 3번입니다", "(4)", "The answer is 2."]
for test_resp in test_responses:
    extracted = evaluator.extract_answer(test_resp)
    print(f"Response: '{test_resp}' -> Extracted: '{extracted}'")
"""