# azure_evaluator.py
import requests
import json
import re

class AzureLMAPIEvaluator:
    def __init__(self, api_key: str, azure_endpoint: str, deployment_name: str, api_version: str = "2024-02-15-preview"):
        self.api_key = api_key
        self.azure_endpoint = azure_endpoint.rstrip('/')
        self.deployment_name = deployment_name
        self.api_version = api_version
        self.api_url = f"{self.azure_endpoint}/openai/deployments/{self.deployment_name}/chat/completions"
        self.headers = {
            "api-key": api_key,
            "Content-Type": "application/json; charset=utf-8"
        }

    def query_model(self, prompt: str, max_tokens: int = 50, temperature: float = 0.0) -> str:
        messages = [
            {"role": "system", "content": "You are a helpful assistant that answers multiple choice questions. Respond only with 1, 2, 3, or 4."},
            {"role": "user", "content": prompt}
        ]
        payload = {
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature
        }
        params = {"api-version": self.api_version}

        try:
            response = requests.post(self.api_url, headers=self.headers, json=payload, params=params, timeout=30)
            response.raise_for_status()
            result = response.json()
            return result["choices"][0]["message"]["content"].strip()
        except Exception as e:
            print(f"API request failed: {e}")
            return ""

    def extract_answer(self, response: str) -> str:
        match = re.search(r'\b([1-4])\b', response)
        return match.group(1) if match else "INVALID"
