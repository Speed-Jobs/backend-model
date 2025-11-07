import json
from pathlib import Path
from typing import List, Dict, Any
import os
import re
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

# .env 지원을 위해 dotenv import 및 적용
from dotenv import load_dotenv
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
class SkillSetOutput(BaseModel):
    
    """스킬셋 추출 결과 모델"""
    skill_set: List[str] = Field(description="추출된 기술 스택 리스트")

class SkillSetMatcher:
    def __init__(self, job_description_path: str):
        """직무 기술서 데이터 로드 및 LLM 초기화"""
        with open(job_description_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 데이터 형식에 따라 처리
        if isinstance(data, dict):
            self.common_skill_set = data.get('공통_skill_set', [])
            raw_skill_set = data.get('skill_set', [])
            if not self.common_skill_set and not raw_skill_set:
                raise ValueError("description.json에서 '공통_skill_set' 또는 'skill_set' 키를 찾을 수 없습니다.")
            self.skill_set = self._parse_skill_descriptions(raw_skill_set)
            
        elif isinstance(data, list):
            self.common_skill_set = []
            all_descriptions = []
            for job_desc in data:
                if isinstance(job_desc, dict):
                    common = job_desc.get('공통_skill_set', [])
                    if isinstance(common, list):
                        self.common_skill_set.extend(common)
                    skill = job_desc.get('skill_set', '')
                    if skill:
                        all_descriptions.append(skill)
            self.common_skill_set = list(set(self.common_skill_set))
            self.skill_set = self._parse_skill_descriptions(all_descriptions)
        else:
            raise ValueError(f"지원하지 않는 데이터 형식입니다. dict 또는 list여야 합니다. 현재 타입: {type(data).__name__}")
        self.all_skills = list(set(self.common_skill_set + self.skill_set))
        self._initialize_llm()
    
    def _parse_skill_descriptions(self, descriptions: List[str]) -> List[str]:
        """긴 설명문에서 개별 스킬 이름 추출"""
        skills = []
        for desc in descriptions:
            bracket_matches = re.findall(r'\(([^)]+)\)', desc)
            for match in bracket_matches:
                items = re.split(r'[,/]', match)
                for item in items:
                    item = item.strip()
                    if len(item) > 1 and not item.replace(' ', '').replace('-', '').replace('.', '').isalpha():
                        skills.append(item)
                    elif any(c.isalnum() for c in item) and len(item) > 1:
                        skills.append(item)
        return list(set(skills))
    
    def _initialize_llm(self):
        """LLM 및 프롬프트 초기화"""
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("OPENAI_API_KEY 환경 변수가 설정되지 않았습니다.\n.env 파일 또는 환경 변수에 키를 설정해주세요.")
        
        self.llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0,
            api_key=api_key
        )
        self.parser = PydanticOutputParser(pydantic_object=SkillSetOutput)
        self.prompt = PromptTemplate(
            template="""당신의 역할: 채용공고(description) 텍스트에서 기술 스택을 추출하는 엔진.
common_skill_set 과 skill_set 내에서만 선택하며, 그 외 새로운 스킬은 생성하지 않는다.

규칙:
1) common_skill_set ∪ skill_set 안에 있는 기술만 추출한다.
2) 스킬명이 description에 등장하면 유사/동의/철자 변형/대소문자 차이를 허용하되, 결과는 canonical 명칭으로 출력한다.
   예: Node, NodeJS → Node.js / ReactJS → React / PyTorch → PyTorch
3) 소프트 스킬, 성향, 업무 방식, 도메인 키워드는 제외한다.
   예: 소통능력, 문제 해결, 핀테크, 애자일 등 제외.
4) "우대", "선호", "경험 있으면 가산점"등의 문맥에서도 기술명만 등장하면 포함한다.
5) 최종 출력은 중복 제거, 알파벳 오름차순 정렬.

사용 가능한 스킬 목록:
{all_skills}

채용공고 내용:
{description}

{format_instructions}

출력 예시:
{{"skill_set": ["AWS", "Docker", "Java", "Kubernetes", "Python", "Spring Boot"]}}
""",
            input_variables=["all_skills", "description"],
            partial_variables={"format_instructions": self.parser.get_format_instructions()}
        )
    
    def match_job_to_skillset(self, job: Dict[str, Any]) -> Dict[str, Any]:
        """단일 job에 대해 LLM을 활용하여 skill_set 추출"""
        description = job.get('description', '')
        title = job.get('title', '')
        full_text = f"제목: {title}\n\n{description}"
        if len(full_text.strip()) < 50:
            print(f"  ⚠️  텍스트가 너무 짧아 스킵: {title}")
            return {
                'matched': False,
                'match_score': 0,
                'skill_set': []
            }
        try:
            chain = self.prompt | self.llm | self.parser
            result = chain.invoke({
                "all_skills": ", ".join(self.all_skills),
                "description": full_text[:4000]
            })
            extracted_skills = result.skill_set
            extracted_skills.sort()
            if extracted_skills:
                return {
                    'matched': True,
                    'match_score': len(extracted_skills),
                    'skill_set': extracted_skills
                }
            else:
                return {
                    'matched': False,
                    'match_score': 0,
                    'skill_set': []
                }
        except Exception as e:
            print(f"  ❌ LLM 호출 중 오류 발생: {str(e)}")
            return {
                'matched': False,
                'match_score': 0,
                'skill_set': [],
                'error': str(e)
            }
    
    def process_jobs_file(self, input_path: str, output_path: str):
        """jobs 파일 처리 및 skill_set 정보 추가"""
        print(f"\n{'='*60}")
        print(f"📁 처리 중: {input_path}")
        print(f"{'='*60}")
        with open(input_path, 'r', encoding='utf-8') as f:
            jobs = json.load(f)
        matched_count = 0
        unmatched_count = 0
        total_skills_extracted = 0
        for idx, job in enumerate(jobs, 1):
            print(f"\n[{idx}/{len(jobs)}] {job.get('title', 'Unknown')}")
            skill_info = self.match_job_to_skillset(job)
            if skill_info['matched']:
                matched_count += 1
                skill_count = len(skill_info['skill_set'])
                total_skills_extracted += skill_count
                print(f"  ✅ {skill_count}개 스킬 추출: {', '.join(skill_info['skill_set'][:5])}{'...' if skill_count > 5 else ''}")
            else:
                unmatched_count += 1
                print(f"  ⚠️  스킬 추출 실패")
            job['skill_set_info'] = skill_info
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(jobs, f, ensure_ascii=False, indent=2)
        print(f"\n{'='*60}")
        print(f"📊 처리 완료 요약")
        print(f"{'='*60}")
        print(f"  - 총 채용공고: {len(jobs)}개")
        print(f"  - 매칭 성공: {matched_count}개 ({matched_count/len(jobs)*100:.1f}%)")
        print(f"  - 매칭 실패: {unmatched_count}개 ({unmatched_count/len(jobs)*100:.1f}%)")
        if matched_count > 0:
            print(f"  - 평균 추출 스킬 수: {total_skills_extracted/matched_count:.1f}개")
        print(f"  - 저장 위치: {output_path}")
        print(f"{'='*60}\n")
        return matched_count, unmatched_count

def main():
    """메인 실행 함수"""
    description_path = Path(__file__).parent / 'description.json'
    data_dir = Path(r"C:\workspace\Final_project\backend-model\data")
    
    print("\n" + "="*60)
    print("🚀 LLM 기반 Skill Set 추출 시작")
    print("="*60)
    print(f"📋 스킬 목록 파일: {description_path}")
    print(f"📂 데이터 디렉토리: {data_dir}")
    if not description_path.exists():
        print(f"\n❌ 오류: description.json 파일을 찾을 수 없습니다.")
        print(f"   찾는 경로: {description_path}")
        print(f"   파일이 존재하는지 확인해주세요.")
        return
    if not data_dir.exists():
        print(f"\n❌ 오류: 데이터 디렉토리를 찾을 수 없습니다.")
        print(f"   찾는 경로: {data_dir}")
        print(f"   디렉토리가 존재하는지 확인해주세요.")
        return
    try:
        matcher = SkillSetMatcher(str(description_path))
        print(f"✅ 총 {len(matcher.all_skills)}개의 스킬 로드 완료")
        print(f"   - 공통 스킬: {len(matcher.common_skill_set)}개")
        print(f"   - 직무별 스킬: {len(matcher.skill_set)}개")
    except ValueError as e:
        print(f"❌ 초기화 실패: {e}")
        # .env도 지원하고 있으므로 안내 멘트도 보강
        if "OPENAI_API_KEY" in str(e):
            print("\n💡 해결 방법:")
            print("   .env 파일에 다음과 같이 추가하세요: OPENAI_API_KEY=your-api-key")
            print("   또는 환경 변수로 설정하세요 (Windows: set, Linux/Mac: export)")
        return
    except Exception as e:
        print(f"❌ 예상치 못한 오류: {e}")
        import traceback
        traceback.print_exc()
        return
    jobs_files = list(data_dir.glob('*_jobs.json'))
    if not jobs_files:
        print(f"\n⚠️  {data_dir}에서 *_jobs.json 파일을 찾을 수 없습니다.")
        return
    print(f"📁 처리할 파일 수: {len(jobs_files)}")
    total_matched = 0
    total_unmatched = 0
    for jobs_file in jobs_files:
        try:
            matched, unmatched = matcher.process_jobs_file(str(jobs_file), str(jobs_file))
            total_matched += matched
            total_unmatched += unmatched
        except Exception as e:
            print(f"\n❌ 파일 처리 중 오류: {jobs_file}")
            print(f"   오류 메시지: {str(e)}")
            continue
    print("\n" + "="*60)
    print("🎉 전체 처리 완료")
    print("="*60)
    print(f"  - 전체 매칭 성공: {total_matched}개")
    print(f"  - 전체 매칭 실패: {total_unmatched}개")
    total = total_matched + total_unmatched
    if total > 0:
        print(f"  - 매칭 성공률: {total_matched / total * 100:.2f}%")
    print(f"  - 결과 파일 위치: {data_dir}")
    print("="*60 + "\n")

if __name__ == "__main__":
    main()