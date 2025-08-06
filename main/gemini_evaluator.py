import requests
import re

class GeminiAPIEvaluator:
    def __init__(self, api_key: str, model: str = "gemini-1.5-flash"):
        self.api_key = api_key
        self.model = model
        self.api_url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent"
        self.headers = {"Content-Type": "application/json"}

    def query_model(self, prompt: str) -> str:
        instructions = (
            "당신은 산업안전 기사 시험 문제를 푸는 전문가입니다.\n"
            "문제와 선택지를 보고 정답 번호만 숫자로 출력하세요.\n"
            "출력 형식: 1, 2, 3, 또는 4 중 하나만.\n"
            "그 외의 설명, 단어, 문장은 절대 포함하지 마세요.\n\n"
        )

        payload = {
            "contents": [{
                "parts": [{
                    "text": instructions + prompt
                }]
            }]
        }

        try:
            response = requests.post(
                self.api_url,
                headers=self.headers,
                params={"key": self.api_key},
                json=payload,
                timeout=60
            )
            response.raise_for_status()
            data = response.json()

            raw_text = data["candidates"][0]["content"]["parts"][0]["text"].strip()
            return raw_text

        except Exception as e:
            print(f"Gemini API request failed: {e}")
            return ""

    def extract_answer(self, response: str) -> str:
        """답안을 숫자(1~4)로 변환"""
        if not response:
            return "INVALID"

        response = response.strip()

        # 1차: 단일 숫자만 있는 경우
        match = re.match(r'^[1-4]$', response)
        if match:
            return match.group(0)

        # 2차: 문자열 안에 숫자 (1~4) 포함
        match = re.search(r'\b([1-4])\b', response)
        if match:
            return match.group(1)

        # 3차: 한글 번호 기호 → 숫자 변환
        kor_map = {'①': '1', '②': '2', '③': '3', '④': '4'}
        for k, v in kor_map.items():
            if k in response:
                return v

        # 4차: "3번", "(2)", "4." 등
        match = re.search(r'\(?([1-4])\)?\.?', response)
        if match:
            return match.group(1)

        return "INVALID"
