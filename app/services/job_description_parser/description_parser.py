import fitz  # PyMuPDF
import os
import json
import csv
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Load environment variables from .env file
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

pdf_path = r"C:\workspace\Final_project\backend-model\app\services\job_description_parser\data\SKAX_Jobdescription.pdf"


# ChatOpenAI 사용 (OpenAI 대신)
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, openai_api_key=openai_api_key)

prompt_template = """
[원문]
{context}

[스키마]
직무: string
industry: string 
공통_skill_set: string[]
skill_set: string

[요구]
1) 원문에서 Software Development 직무에 대한 industry별 블록을 식별해 3개의 JSON 객체로 출력.
2) 공통_skill_set은 '공통' 섹션의 항목을 리스트로 수집 후 중복/변형어 정규화.
3) skill_set은 해당 industry 섹션의 기술을 요약 나열(문장 가능). 세부 목록은 콤마로 연결.
4) 한 줄에 JSON 하나(JSONL). 주석/설명 금지.

결과 예시
{{"직무":"Software Development","industry":"Front-end Development","공통_skill_set":["Java","node.js","Python","C / C++","Go","ASP.NET","Perl","Ruby","C#","PHP","Visual Basic","Git","Github","SVN","Bitbucket","Jira","Confluence","Slack","Teams","Notion","Google Docs","AI Literacy / Collaboration","1/2금융","대외제조","대내Hi-Tech","대내Process","통신","유통/물류/서비스","미디어/콘텐츠","공공","Global"],"skill_set":"사용자와 직접 상호작용하는 UI를 구현하고 UX를 개선. UI/UX 도구(Sketch, Adobe XD, Figma), 웹 프레임워크(React, Angular, Vue.js, Next.js, Nuxt.js, jQuery), 퍼블리싱(HTML, CSS, Bootstrap, Material UI, Sass/Scss/Less)를 다룸."}}
{{"직무":"Software Development","industry":"Back-end Development","공통_skill_set":["Java","node.js","Python","C / C++","Go","ASP.NET","Perl","Ruby","C#","PHP","Visual Basic","Git","Github","SVN","Bitbucket","Jira","Confluence","Slack","Teams","Notion","Google Docs","AI Literacy / Collaboration","1/2금융","대외제조","대내Hi-Tech","대내Process","통신","유통/물류/서비스","미디어/콘텐츠","공공","Global"],"skill_set":"웹프레임워크(Spring, Spring Boot, Spring Cloud, Django, Flask, Nexcore J2EE/.Net, ASP.NET), 데이터베이스(MySQL, PostgreSQL, Oracle, MS_SQL, MongoDB, MariaDB, Cassandra, Redis, Altibase), 접근기술(JPA, mybatis, ibatis, JDBC, Entity Framework), 연동(API: REST, SOAP, RPC, WebSocket; 메시징: Kafka, Rabbit, MSMQ)."}}
{{"직무":"Software Development","industry":"Mobile Development","공통_skill_set":["Java","node.js","Python","C / C++","Go","ASP.NET","Perl","Ruby","C#","PHP","Visual Basic","Git","Github","SVN","Bitbucket","Jira","Confluence","Slack","Teams","Notion","Google Docs","AI Literacy / Collaboration","1/2금융","대외제조","대내Hi-Tech","대내Process","통신","유통/물류/서비스","미디어/콘텐츠","공공","Global"],"skill_set":"iOS(휴먼 인터페이스 지침), Android(머티리얼 디자인), React Native, Xamarin, Flutter, Swift, Objective-C, Xcode, iOS SDK, TestFlight, Java, Kotlin, Android Studio, Android SDK, XML, SQLite, RESTful API, Firebase."}}
"""

# LCEL 방식: prompt | llm | output_parser
prompt = ChatPromptTemplate.from_template(prompt_template)
output_parser = StrOutputParser()
chain = prompt | llm | output_parser

output_dir = os.path.dirname(pdf_path)
csv_file_path = os.path.join(output_dir, "output.csv")
json_file_path = os.path.join(output_dir, "output.json")

all_results = []

print(f"[확인] 출력 파일 경로:")
print(f"  CSV: {csv_file_path}")
print(f"  JSON: {json_file_path}\n")

with fitz.open(pdf_path) as doc:
    for page_num in range(0, len(doc), 2):
        pages_text = []
        for offset in range(2):
            if page_num + offset < len(doc):
                page = doc[page_num + offset]
                pages_text.append(page.get_text())
        context = "\n\n".join(pages_text)
        if context.strip():
            response = chain.invoke({"context": context})
            
            # 마크다운 코드 블록 제거
            response_cleaned = response.strip()
            if response_cleaned.startswith("```json"):
                response_cleaned = response_cleaned[7:]  # ```json 제거
            if response_cleaned.startswith("```"):
                response_cleaned = response_cleaned[3:]  # ``` 제거
            if response_cleaned.endswith("```"):
                response_cleaned = response_cleaned[:-3]  # 끝의 ``` 제거
            response_cleaned = response_cleaned.strip()
            
            print(f"\n[페이지 {page_num+1}-{min(page_num+2, len(doc))}]")
            print(f"원본 응답:\n{response}\n")
            print(f"정제 후:\n{response_cleaned}\n")
            
            # response는 JSONL 형식(한 줄에 JSON)
            lines = [line.strip() for line in response_cleaned.splitlines() if line.strip()]
            print(f"파싱할 줄 수: {len(lines)}")
            
            for idx, line in enumerate(lines, 1):
                try:
                    item = json.loads(line)
                    all_results.append(item)
                    print(f"  ✅ 줄 {idx}: JSON 파싱 성공 - industry: {item.get('industry', 'N/A')}")
                except Exception as e:
                    print(f"  ❌ 줄 {idx}: JSON 파싱 실패")
                    print(f"     에러: {e}")
                    print(f"     내용: {line[:100]}...")
            
            print("=" * 80)

print(f"\n{'='*80}")
print(f"총 {len(all_results)}개 레코드 수집됨")
print(f"{'='*80}\n")

# 결과를 JSON 파일로 저장
if all_results:
    with open(json_file_path, "w", encoding="utf-8") as jf:
        json.dump(all_results, jf, ensure_ascii=False, indent=2)
    print(f"✅ JSON 저장 완료: {json_file_path}")
    print(f"   레코드 수: {len(all_results)}")
    
    # 결과를 CSV 파일로 저장
    # 모든 키 수집 (공통 열, skill_set은 string, 공통_skill_set은 리스트이므로 문자열 변환)
    keys = set()
    for item in all_results:
        keys.update(item.keys())
    keys = list(keys)
    
    with open(csv_file_path, "w", newline='', encoding="utf-8-sig") as cf:
        writer = csv.DictWriter(cf, fieldnames=keys)
        writer.writeheader()
        for item in all_results:
            # 리스트 필드는 문자열로 변환
            item_row = item.copy()
            for k, v in item_row.items():
                if isinstance(v, list):
                    item_row[k] = ", ".join(v)
            writer.writerow(item_row)
    
    print(f"✅ CSV 저장 완료: {csv_file_path}")
    print(f"   행 수: {len(all_results)}, 열 수: {len(keys)}")
    print(f"   열 이름: {keys}")
else:
    print("❌ 저장할 레코드가 없습니다!")
    print("   all_results 리스트가 비어있습니다.")
    print("   위의 JSON 파싱 에러 메시지를 확인하세요.")