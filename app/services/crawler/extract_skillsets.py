import os
import re
import json
from pathlib import Path
from typing import List, Dict, Any
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser

# .env 파일을 로드하여 환경 변수 설정
load_dotenv()


class SkillSetOutput(BaseModel):
    """
    LLM이 반환하는 스킬셋 추출 결과를 위한 모델
    """
    skill_set: List[str] = Field(description="추출된 기술 스택 리스트")


class SkillSetMatcher:
    """
    description.json을 읽고,
    LLM 기반으로 job description에서 skill set을 추출하는 클래스
    """

    def __init__(self, job_description_path: str):
        """
        직무 기술서 데이터와 SkillSet 추출 및 LLM 초기화
        """
        # 직무 기술서 파일 로드
        with open(job_description_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        self._extract_skillset_from_data(data)
        self._initialize_llm()

    def _extract_skillset_from_data(self, data: Any):
        """
        description.json 기본 구조에 따라
        공통 스킬셋과 개별 스킬셋, 전체 스킬셋을 계산
        """
        if isinstance(data, dict):
            # 딕셔너리 구조: 단일 설명서
            self.common_skill_set = data.get('공통_skill_set', [])
            raw_skill_set = data.get('skill_set', [])
            if not self.common_skill_set and not raw_skill_set:
                raise ValueError("description.json에서 '공통_skill_set' 또는 'skill_set' 키를 찾을 수 없습니다.")
            # skill_set이 배열인지 설명문인지 파악 후 처리
            self.skill_set = self._process_skill_set(raw_skill_set)
        elif isinstance(data, list):
            # 리스트 구조: 여러 직군 설명서
            self.common_skill_set = []
            all_descriptions = []
            for job_desc in data:
                if isinstance(job_desc, dict):
                    # 공통 스킬 합치기
                    common = job_desc.get('공통_skill_set', [])
                    if isinstance(common, list):
                        self.common_skill_set.extend(common)
                    # 직무별 스킬셋 또는 설명문 추가
                    skill = job_desc.get('skill_set', '')
                    if skill:
                        all_descriptions.append(skill)
            self.common_skill_set = list(set(self.common_skill_set))
            self.skill_set = self._process_skill_set(all_descriptions)
        else:
            raise ValueError(
                f"지원하지 않는 데이터 형식입니다. dict 또는 list여야 합니다. 현재 타입: {type(data).__name__}"
            )
        # 전체 스킬 리스트(중복제거, 정렬)
        self.all_skills = sorted(list(set(self.common_skill_set + self.skill_set)))

    def _process_skill_set(self, skill_set: Any) -> List[str]:
        """
        skill_set(배열/설명문)을 파싱하여 스킬 리스트 반환

        Args:
            skill_set: 배열([str,...]) 또는 str/설명문

        Returns:
            List[str]: 추출된 스킬 문자열 리스트
        """
        if not skill_set:
            return []

        # 배열이며 각 요소가 괄호 없는 문자열이면(이미 스킬명 배열)
        if isinstance(skill_set, list):
            if all(isinstance(item, str) and '(' not in item for item in skill_set):
                return [skill.strip() for skill in skill_set if skill.strip()]
            # 설명문 형식(문장 리스트)이면 파싱
            else:
                return self._parse_skill_descriptions(skill_set)
        # str(설명문) -> 파싱
        elif isinstance(skill_set, str):
            return self._parse_skill_descriptions([skill_set])
        return []

    def _parse_skill_descriptions(self, descriptions: List[str]) -> List[str]:
        """
        설명문이나 문장 리스트에서 괄호 등을 파싱해 개별 스킬 명 추출
        예: "웹프레임워크(Spring, Django, Flask)"

        Args:
            descriptions: 설명문 리스트

        Returns:
            List[str]: 추출된 스킬명 목록(중복제거)
        """
        skills = []
        for desc in descriptions:
            if not desc or not isinstance(desc, str):
                continue

            # 괄호로 묶인 스킬 추출
            matches = re.findall(r'\(([^)]+)\)', desc)
            if matches:
                for match in matches:
                    # 콤마(,)나 슬래시로 여러 스킬 구분
                    for item in re.split(r'[,/]', match):
                        item = item.strip()
                        # 조건1: 문자열 내 문자/숫자 포함, 길이 > 1
                        if len(item) > 1 and not item.replace(' ', '').replace('-', '').replace('.', '').isalpha():
                            skills.append(item)
                        elif any(c.isalnum() for c in item) and len(item) > 1:
                            skills.append(item)
            else:
                # 괄호 없으면 한 문장짜리 스킬로 간주
                desc = desc.strip()
                if desc and len(desc) > 1:
                    skills.append(desc)
        return list(set(skills))

    def _initialize_llm(self):
        """
        LLM 및 프롬프트, 파서 초기화
        """
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError(
                "OPENAI_API_KEY 환경 변수가 설정되지 않았습니다.\n.env 파일 또는 환경 변수에 키를 설정해주세요."
            )
        # LLM 인스턴스 준비
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=api_key)
        # 결과 파싱을 위한 PydanticOutputParser
        self.parser = PydanticOutputParser(pydantic_object=SkillSetOutput)
        # LLM 프롬프트 템플릿 작성
        self.prompt = PromptTemplate(
            template=(
                "당신의 역할: 채용공고(description) 텍스트에서 기술 스택을 추출하는 엔진.\n"
                "사용 가능한 스킬 목록 내에서만 선택하며, 그 외 새로운 스킬은 생성하지 않는다.\n\n"
                "규칙:\n"
                "1) 사용 가능한 스킬 목록 안에 있는 기술만 추출한다.\n"
                "2) 스킬명이 description에 등장하면 유사/동의/철자 변형/대소문자 차이를 허용하되, "
                "결과는 canonical 명칭으로 출력한다.\n"
                "   예: Node, NodeJS → Node.js / ReactJS → React / PyTorch → PyTorch\n"
                "3) 소프트 스킬, 성향, 업무 방식, 도메인 키워드는 제외한다.\n"
                "   예: 소통능력, 문제 해결, 핀테크, 애자일 등 제외.\n"
                '4) "우대", "선호", "경험 있으면 가산점"등의 문맥에서도 기술명만 등장하면 포함한다.\n'
                "5) 최종 출력은 중복 제거, 알파벳 오름차순 정렬.\n\n"
                "사용 가능한 스킬 목록:\n"
                "{all_skills}\n\n"
                "채용공고 내용:\n"
                "{description}\n\n"
                "{format_instructions}\n\n"
                "출력 예시:\n"
                "{{\"skill_set\": [\"AWS\", \"Docker\", \"Java\", \"Kubernetes\", \"Python\", \"Spring Boot\"]}}"
            ),
            input_variables=["all_skills", "description"],
            partial_variables={"format_instructions": self.parser.get_format_instructions()}
        )

    def match_job_to_skillset(self, job: Dict[str, Any]) -> Dict[str, Any]:
        """
        단일 채용공고(job)에서 LLM을 사용하여 skill_set 추출
        """
        description = job.get('description', '')
        title = job.get('title', '')
        full_text = f"제목: {title}\n\n{description}"
        # 채용공고 내용이 너무 짧으면 skip
        if len(full_text.strip()) < 50:
            print(f"  :경고:  텍스트가 너무 짧아 스킵: {title}")
            return {'matched': False, 'match_score': 0, 'skill_set': []}
        try:
            # 프롬프트 → LLM → 파서 연결
            chain = self.prompt | self.llm | self.parser
            result = chain.invoke({
                "all_skills": ", ".join(self.all_skills),
                "description": full_text[:4000],  # LLM context 제한 고려
            })
            extracted_skills = sorted(result.skill_set)
            if extracted_skills:
                return {
                    'matched': True,
                    'match_score': len(extracted_skills),
                    'skill_set': extracted_skills
                }
            return {'matched': False, 'match_score': 0, 'skill_set': []}
        except Exception as e:
            print(f"  :x: LLM 호출 중 오류 발생: {str(e)}")
            return {'matched': False, 'match_score': 0, 'skill_set': [], 'error': str(e)}

    def process_jobs_file(self, input_path: str, output_path: str):
        """
        jobs 파일 내 모든 채용공고에 대해 skill_set 추출 후 결과 추가 및 저장

        Args:
            input_path (str): 입력 파일 경로
            output_path (str): 결과 저장 경로
        """
        print(f"\n{'='*60}\n:파일_폴더: 처리 중: {input_path}\n{'='*60}")

        # 파일 로드
        with open(input_path, 'r', encoding='utf-8') as f:
            jobs = json.load(f)

        matched_count, unmatched_count, total_skills = 0, 0, 0

        # 각 채용공고마다 skill_set 추출
        for idx, job in enumerate(jobs, 1):
            print(f"\n[{idx}/{len(jobs)}] {job.get('title', 'Unknown')}")
            skill_info = self.match_job_to_skillset(job)
            if skill_info['matched']:
                matched_count += 1
                skill_count = len(skill_info['skill_set'])
                total_skills += skill_count
                skills_preview = ', '.join(skill_info['skill_set'][:5])
                print(f"  :흰색_확인_표시: {skill_count}개 스킬 추출: {skills_preview}{'...' if skill_count > 5 else ''}")
            else:
                unmatched_count += 1
                print("  :경고:  스킬 추출 실패")
            # 추출 결과를 job에 추가
            job['skill_set_info'] = skill_info

        # 결과 파일로 저장
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(jobs, f, ensure_ascii=False, indent=2)

        # 최종 요약 출력
        print(
            f"\n{'='*60}\n:막대_차트: 처리 완료 요약\n{'='*60}\n"
            f"  - 총 채용공고: {len(jobs)}개\n"
            f"  - 매칭 성공: {matched_count}개 ({matched_count/len(jobs)*100:.1f}%)\n"
            f"  - 매칭 실패: {unmatched_count}개 ({unmatched_count/len(jobs)*100:.1f}%)"
        )
        if matched_count > 0:
            print(f"  - 평균 추출 스킬 수: {total_skills/matched_count:.1f}개")
        print(f"  - 저장 위치: {output_path}\n{'='*60}\n")

        return matched_count, unmatched_count


def main():
    """
    스킬셋 추출 전체 파이프라인 메인 함수
    (경로 설정, matcher 생성/파일 순회, 전체 진행상황/요약 출력)
    """
    # 현재 파일 위치에서 프로젝트 루트 찾기
    # extract_skillsets.py 위치: backend-model/app/utils/skillset_parser/
    # 프로젝트 루트: backend-model/ (4단계 상위: skillset_parser -> utils -> app -> backend-model)
    current_file = Path(__file__).resolve()
    backend_model_dir = current_file.parents[3]  # 4단계 상위 (프로젝트 루트)

    # description.json 파일 찾기 (여러 위치 확인)
    # 1. 프로젝트 루트의 data 디렉토리 
    description_path = backend_model_dir / 'data' / 'description.json'
  
    
    # backend-model/data/SKAX_Jobdescription.pdf 경로 설정
    description_path = backend_model_dir / 'data' / 'description.json'
    
    # backend-model/data 디렉토리 경로 설정
    data_dir = backend_model_dir / 'data' / 'output'

    print("\n" + "=" * 60)
    print(":로켓: LLM 기반 Skill Set 추출 시작")
    print("=" * 60)
    print(f":클립보드: 스킬 목록 파일: {description_path}")
    print(f":열린_파일_폴더: 데이터 디렉토리: {data_dir}")

    # 사전 파일(경로) 체크
    if not description_path.exists():
        print(f"\n:x: 오류: description.json 파일을 찾을 수 없습니다.\n   찾는 경로: {description_path}\n   파일이 존재하는지 확인해주세요.")
        return
    if not data_dir.exists():
        print(f"\n:x: 오류: 데이터 디렉토리를 찾을 수 없습니다.\n   찾는 경로: {data_dir}\n   디렉토리가 존재하는지 확인해주세요.")
        return

    # SkillSetMatcher 생성 및 스킬셋 로드
    try:
        matcher = SkillSetMatcher(str(description_path))
        print(
            f":흰색_확인_표시: 총 {len(matcher.all_skills)}개의 스킬 로드 완료\n"
            f"   - 공통 스킬: {len(matcher.common_skill_set)}개\n"
            f"   - 직무별 스킬: {len(matcher.skill_set)}개"
        )
    except ValueError as e:
        print(f":x: 초기화 실패: {e}")
        if "OPENAI_API_KEY" in str(e):
            print("\n:전구: 해결 방법:\n   .env 파일에 다음과 같이 추가하세요: OPENAI_API_KEY=your-api-key\n   또는 환경 변수로 설정하세요 (Windows: set, Linux/Mac: export)")
        return
    except Exception as e:
        print(f":x: 예상치 못한 오류: {e}")
        import traceback
        traceback.print_exc()
        return

    # *_jobs.json 파일 리스트 찾기
    jobs_files = list(data_dir.glob('*_jobs.json'))
    if not jobs_files:
        print(f"\n:경고:  {data_dir}에서 *_jobs.json 파일을 찾을 수 없습니다.")
        return

    print(f":파일_폴더: 처리할 파일 수: {len(jobs_files)}")
    total_matched, total_unmatched = 0, 0

    # 각 파일을 순회하며 skill_set 추출 처리
    for jobs_file in jobs_files:
        try:
            matched, unmatched = matcher.process_jobs_file(str(jobs_file), str(jobs_file))
            total_matched += matched
            total_unmatched += unmatched
        except Exception as e:
            print(f"\n:x: 파일 처리 중 오류: {jobs_file}\n   오류 메시지: {str(e)}")
            continue

    # 전체 요약 출력
    print("\n" + "=" * 60)
    print(":짠: 전체 처리 완료")
    print("=" * 60)
    print(f"  - 전체 매칭 성공: {total_matched}개")
    print(f"  - 전체 매칭 실패: {total_unmatched}개")
    total = total_matched + total_unmatched
    if total > 0:
        print(f"  - 매칭 성공률: {total_matched / total * 100:.2f}%")
    print(f"  - 결과 파일 위치: {data_dir}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()