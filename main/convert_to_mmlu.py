import pandas as pd
import json
from pathlib import Path

# 파일 읽기
df = pd.read_csv(r"C:\Users\82103\OneDrive - UNIST\바탕 화면\safeQnA\jedai\산업안전기사_층화추출_300.csv")

# 열 이름 변환 (eval_model.py 필드 + 메타데이터 필드)
df = df.rename(columns={
    "Question": "question",
    "Option 1": "choice_1",
    "Option 2": "choice_2",
    "Option 3": "choice_3",
    "Option 4": "choice_4",
    "Answer": "answer"
})

# raw_data 폴더 생성
Path("raw_data").mkdir(exist_ok=True)

# CSV 저장 (eval_model.py가 읽을 수 있도록)
df[["question", "choice_1", "choice_2", "choice_3", "choice_4", "answer"]].to_csv(
    "raw_data/industrial_safety.csv", index=False, encoding="utf-8"
)

# JSON 변환 (메타데이터 포함)
dataset_json = []
for _, row in df.iterrows():
    entry = {
        "question": row["question"],
        "choices": {
            "1": row["choice_1"],
            "2": row["choice_2"],
            "3": row["choice_3"],
            "4": row["choice_4"],
        },
        "answer": row["answer"],
        "metadata": {
            "Test Name": row["Test Name"],
            "Year": row["Year"],
            "Session": row["Session"],
            "Subject": row["Subject"],
            "Number": row["Number"],
            "Question_image": row["Question_image"],
            "법령": row["법령"]
        }
    }
    dataset_json.append(entry)

with open("raw_data/industrial_safety.json", "w", encoding="utf-8") as f:
    json.dump(dataset_json, f, indent=2, ensure_ascii=False)

# config.json 생성
config = {
    "data_loader": {
        "directory": "raw_data",
        "output_directory": "processed_data",
        "file_pattern": "industrial_safety.csv",
        "has_header": True,
        "column_config": {
            "with_header": {
                "question": "question",
                "choice_1": "choice_1",
                "choice_2": "choice_2",
                "choice_3": "choice_3",
                "choice_4": "choice_4",
                "answer": "answer"
            },
            "no_header": {
                "question": 0,
                "choice_1": 1,
                "choice_2": 2,
                "choice_3": 3,
                "choice_4": 4,
                "answer": 5
            }
        },
        "encoding": "utf-8"
    },
    "huggingface_models": {
        "my_model": {
            "base_model_id": "모델이름",
            "adapter_model_id": "어댑터이름"
        }
    }
}

with open("config.json", "w", encoding="utf-8") as f:
    json.dump(config, f, indent=2, ensure_ascii=False)

print("✅ CSV, JSON, config.json 생성 완료")
print("현재 작업 폴더:", Path.cwd())  # 현재 작업 경로 확인
print("CSV 파일 경로:", Path("raw_data/industrial_safety.csv").resolve())
print("JSON 파일 경로:", Path("raw_data/industrial_safety.json").resolve())
print("Config 경로:", Path("config.json").resolve())

