import os
import json
from pathlib import Path

script_dir = os.path.dirname(os.path.abspath(__file__))
root_data_dir = os.path.join(script_dir, "data")

print(f"script_dir: {script_dir}")
print(f"root_data_dir: {root_data_dir}")
print(f"exists: {os.path.exists(root_data_dir)}")

if os.path.exists(root_data_dir):
    files = os.listdir(root_data_dir)
    print(f"Files: {files}")

    for fname in files:
        if fname.endswith("_jobs.json"):
            full_path = os.path.join(root_data_dir, fname)
            print(f"\n파일: {fname}")
            print(f"경로: {full_path}")
            print(f"존재: {os.path.exists(full_path)}")

            try:
                with open(full_path, 'r', encoding='utf-8') as f:
                    jobs = json.load(f)
                    if isinstance(jobs, list):
                        print(f"✓ 로드 성공: {len(jobs)}개")
                    elif isinstance(jobs, dict):
                        print(f"✓ 로드 성공: dict 1개")
            except Exception as e:
                print(f"✗ 에러: {e}")
