# Skill Set 추출 시스템

OpenAI LLM과 LangChain을 활용하여 채용공고에서 기술 스택을 자동으로 추출합니다.

## 주요 기능

- `description.json`에 정의된 공통 스킬 및 직무별 스킬만 추출
- 유사 스킬명 자동 매핑 (예: NodeJS → Node.js, ReactJS → React)
- 소프트 스킬 자동 제외
- 채용공고별 기술 스택 리스트 생성

## 설치

필요한 라이브러리를 설치합니다:

```bash
pip install -r ../../../../requirements.txt
```

## 환경 설정

OpenAI API 키를 환경 변수로 설정해야 합니다:

### Windows (PowerShell)
```powershell
$env:OPENAI_API_KEY="your-api-key-here"
```

### Windows (CMD)
```cmd
set OPENAI_API_KEY=your-api-key-here
```

### Linux/Mac
```bash
export OPENAI_API_KEY=your-api-key-here
```

## 사용 방법

```bash
python extract_skillsets.py
```

## 입력 파일

### description.json
직무 기술서 리스트를 포함하는 JSON 파일입니다.

구조:
```json
[
  {
    "직무": "Backend 개발",
    "industry": "IT/개발",
    "공통_skill_set": ["Java", "Python", "Git", ...],
    "skill_set": "웹프레임워크(Spring, Django, Flask), 데이터베이스(MySQL, PostgreSQL)..."
  },
  ...
]
```

### data/*.json
`data` 폴더의 모든 `*_jobs.json` 파일을 처리합니다.

예: `hanwha_jobs.json`, `kakao_jobs.json`, `naver_jobs.json` 등

## 출력

각 채용공고 JSON 파일에 `skill_set_info` 필드가 추가됩니다:

```json
{
  "title": "LLM Engineer(AI 서비스 개발)",
  "company": "한화시스템/ICT",
  "description": "...",
  "skill_set_info": {
    "matched": true,
    "match_score": 15,
    "skill_set": [
      "AWS",
      "Docker",
      "Kubernetes",
      "LangChain",
      "PyTorch",
      "Python",
      ...
    ]
  }
}
```

## 결과 예시

```
============================================================
🚀 LLM 기반 Skill Set 추출 시작
============================================================
📋 스킬 목록 파일: description.json
📂 데이터 디렉토리: C:\workspace\Final_project\backend-model\data
✅ 총 450개의 스킬 로드 완료
   - 공통 스킬: 233개
   - 직무별 스킬: 217개
📁 처리할 파일 수: 4

============================================================
📁 처리 중: hanwha_jobs.json
============================================================

[1/20] SAP MM 운영/개발 경력사원 채용
  ✅ 8개 스킬 추출: ABAP, Java, MS-SQL, SAP, Spring...

[2/20] LLM Engineer(AI 서비스 개발) 경력사원 채용
  ✅ 15개 스킬 추출: AWS, Docker, Kubernetes, LangChain...

...

============================================================
🎉 전체 처리 완료
============================================================
  - 전체 매칭 성공: 78개
  - 전체 매칭 실패: 2개
  - 매칭 성공률: 97.50%
  - 결과 파일 위치: C:\workspace\Final_project\backend-model\data
============================================================
```

## 주의사항

1. OpenAI API 사용료가 발생합니다 (GPT-4o-mini 사용)
2. 처리 시간은 채용공고 수와 길이에 따라 다릅니다
3. API 호출 제한이 있을 수 있습니다

## 문제 해결

### "OPENAI_API_KEY 환경 변수가 설정되지 않았습니다"
→ 위의 환경 설정 방법을 따라 API 키를 설정하세요.

### "description.json 파일을 찾을 수 없습니다"
→ 파일이 `app/services/crawler/skillset_parser/` 폴더에 있는지 확인하세요.

### "데이터 디렉토리를 찾을 수 없습니다"
→ `data` 폴더 경로가 올바른지 확인하세요.

