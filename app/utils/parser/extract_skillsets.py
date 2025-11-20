import os
import re
import json
import asyncio
import time
from pathlib import Path
from typing import List, Dict, Any, Optional
from collections import defaultdict
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser

# .env 파일을 로드하여 환경 변수 설정
load_dotenv()

'''
채용공고에서 원래 없던 Skill Set을 추출하는 .py 파일
'''


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
            self.common_skill_set = data.get('공통_skill_set', [])
            raw_skill_set = data.get('skill_set', [])
            if not self.common_skill_set and not raw_skill_set:
                raise ValueError("description.json에서 '공통_skill_set' 또는 'skill_set' 키를 찾을 수 없습니다.")
            self.skill_set = self._process_skill_set(raw_skill_set)
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
            self.skill_set = self._process_skill_set(all_descriptions)
        else:
            raise ValueError(
                f"지원하지 않는 데이터 형식입니다. dict 또는 list여야 합니다. 현재 타입: {type(data).__name__}"
            )
        self.all_skills = sorted(list(set(self.common_skill_set + self.skill_set)))

    def _process_skill_set(self, skill_set: Any) -> List[str]:
        if not skill_set:
            return []
        if isinstance(skill_set, list):
            if all(isinstance(item, str) and '(' not in item for item in skill_set):
                return [skill.strip() for skill in skill_set if skill.strip()]
            else:
                return self._parse_skill_descriptions(skill_set)
        elif isinstance(skill_set, str):
            return self._parse_skill_descriptions([skill_set])
        return []

    def _parse_skill_descriptions(self, descriptions: List[str]) -> List[str]:
        skills = []
        for desc in descriptions:
            if not desc or not isinstance(desc, str):
                continue
            matches = re.findall(r'\(([^)]+)\)', desc)
            if matches:
                for match in matches:
                    for item in re.split(r'[,/]', match):
                        item = item.strip()
                        if len(item) > 1 and not item.replace(' ', '').replace('-', '').replace('.', '').isalpha():
                            skills.append(item)
                        elif any(c.isalnum() for c in item) and len(item) > 1:
                            skills.append(item)
            else:
                desc = desc.strip()
                if desc and len(desc) > 1:
                    skills.append(desc)
        return list(set(skills))

    def _initialize_llm(self):
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError(
                "OPENAI_API_KEY 환경 변수가 설정되지 않았습니다.\n.env 파일 또는 환경 변수에 키를 설정해주세요."
            )
        # LLM 인스턴스 준비
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=api_key)
        self.parser = PydanticOutputParser(pydantic_object=SkillSetOutput)
        self.prompt = PromptTemplate(
            template=(
                "당신의 역할: 채용공고(description) 텍스트에서 기술 스택을 추출하는 엔진.\n"
                "사용 가능한 스킬 목록을 참고하여 생성한다.\n\n"
                "규칙:\n"
                "1) 스킬명이 description에 등장하면 유사/동의/철자 변형/대소문자 차이를 허용하되, "
                "결과는 canonical 명칭으로 출력한다.\n"
                "   예: Node, NodeJS → Node.js / ReactJS → React / PyTorch → PyTorch\n"
                "2) 소프트 스킬, 성향, 업무 방식, 도메인 키워드는 제외한다.\n"
                "   예: 소통능력, 문제 해결, 핀테크, 애자일 등 제외.\n"
                '3) "우대", "선호", "경험 있으면 가산점"등의 문맥에서도 기술명만 등장하면 포함한다.\n'
                "4) 최종 출력은 중복 제거, 알파벳 오름차순 정렬.\n\n"
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

    async def match_job_to_skillset_async(
        self, 
        job: Dict[str, Any], 
        max_retries: int = 3,
        timeout: int = 60
    ) -> Dict[str, Any]:
        """
        비동기 방식으로 단일 채용공고(job)에 대해 LLM을 사용하여 skill_set 추출
        재시도 로직과 타임아웃 포함
        """
        description = job.get('description', '')
        title = job.get('title', '')
        full_text = f"제목: {title}\n\n{description}"
        
        if len(full_text.strip()) < 50:
            return {'matched': False, 'match_score': 0, 'skill_set': [], 'reason': 'text_too_short'}
        
        chain = self.prompt | self.llm | self.parser
        
        for attempt in range(max_retries):
            try:
                # 타임아웃 설정
                result = await asyncio.wait_for(
                    chain.ainvoke({
                        "all_skills": ", ".join(self.all_skills),
                        "description": full_text[:4000],
                    }),
                    timeout=timeout
                )
                extracted_skills = sorted(result.skill_set)
                if extracted_skills:
                    return {
                        'matched': True,
                        'match_score': len(extracted_skills),
                        'skill_set': extracted_skills
                    }
                return {'matched': False, 'match_score': 0, 'skill_set': []}
                
            except asyncio.TimeoutError:
                if attempt < max_retries - 1:
                    wait_time = (attempt + 1) * 2  # 지수 백오프
                    await asyncio.sleep(wait_time)
                    continue
                return {'matched': False, 'match_score': 0, 'skill_set': [], 'error': 'timeout'}
                
            except Exception as e:
                if attempt < max_retries - 1:
                    wait_time = (attempt + 1) * 2
                    await asyncio.sleep(wait_time)
                    continue
                return {'matched': False, 'match_score': 0, 'skill_set': [], 'error': str(e)}
        
        return {'matched': False, 'match_score': 0, 'skill_set': [], 'error': 'max_retries_exceeded'}

    async def process_jobs_file_async(
        self, 
        input_path: str, 
        output_path: str, 
        batch_size: int = 30,
        show_progress: bool = True
    ):
        """
        jobs 파일 내 모든 채용공고에 대해 skill_set 추출 후 결과 추가 및 저장
        비동기 병렬 처리 (기본 50개 동시 처리)
        """
        print(f"\n{'='*60}\n:파일_폴더: 처리 중: {input_path}\n{'='*60}")
        start_time = time.time()
        
        with open(input_path, 'r', encoding='utf-8') as f:
            jobs = json.load(f)
        
        total_jobs = len(jobs)
        print(f"총 {total_jobs}개의 채용공고 처리 시작 (병렬 처리: {batch_size}개)")
        
        # Thread-safe 카운터를 위한 Lock
        stats_lock = asyncio.Lock()
        stats = {
            'matched': 0,
            'unmatched': 0,
            'total_skills': 0,
            'errors': defaultdict(int),
            'completed': 0
        }

        async def process_job(idx: int, job: Dict[str, Any]):
            """단일 작업 처리"""
            title = job.get('title', 'Unknown')
            try:
                skill_info = await self.match_job_to_skillset_async(job)
                job['skill_set_info'] = skill_info
                
                async with stats_lock:
                    stats['completed'] += 1
                    if skill_info['matched']:
                        stats['matched'] += 1
                        skill_count = len(skill_info['skill_set'])
                        stats['total_skills'] += skill_count
                    else:
                        stats['unmatched'] += 1
                        if 'error' in skill_info:
                            error_type = skill_info.get('error', 'unknown')
                            stats['errors'][error_type] += 1
                    
                    # 진행 상황 출력 (10개마다 또는 마지막)
                    if show_progress and (stats['completed'] % 10 == 0 or stats['completed'] == total_jobs):
                        progress_pct = (stats['completed'] / total_jobs) * 100
                        elapsed = time.time() - start_time
                        rate = stats['completed'] / elapsed if elapsed > 0 else 0
                        remaining = (total_jobs - stats['completed']) / rate if rate > 0 else 0
                        print(
                            f"[진행: {stats['completed']}/{total_jobs} ({progress_pct:.1f}%)] "
                            f"성공: {stats['matched']}, 실패: {stats['unmatched']} | "
                            f"속도: {rate:.1f}개/초 | 예상 남은 시간: {remaining:.0f}초"
                        )
                        
            except Exception as e:
                async with stats_lock:
                    stats['completed'] += 1
                    stats['unmatched'] += 1
                    stats['errors'][type(e).__name__] += 1
                    job['skill_set_info'] = {
                        'matched': False,
                        'match_score': 0,
                        'skill_set': [],
                        'error': str(e)
                    }
                if show_progress:
                    print(f"  [오류] [{idx}/{total_jobs}] {title}: {str(e)[:100]}")

        # 세마포어로 동시 실행 수 제한
        semaphore = asyncio.Semaphore(batch_size)

        async def sem_task(idx: int, job: Dict[str, Any]):
            """세마포어로 제어된 작업"""
            async with semaphore:
                await process_job(idx, job)

        # 모든 작업 생성 및 실행
        tasks = [
            asyncio.create_task(sem_task(idx, job))
            for idx, job in enumerate(jobs, 1)
        ]
        
        # 모든 작업 완료 대기 (에러가 있어도 계속 진행)
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # 예외 처리
        exception_count = sum(1 for r in results if isinstance(r, Exception))
        if exception_count > 0:
            print(f"\n⚠ {exception_count}개의 작업에서 예외 발생 (계속 진행)")

        # 결과 저장
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(jobs, f, ensure_ascii=False, indent=2)

        elapsed_time = time.time() - start_time
        
        # 최종 통계 출력
        print(
            f"\n{'='*60}\n:막대_차트: 처리 완료 요약\n{'='*60}\n"
            f"  - 총 채용공고: {total_jobs}개\n"
            f"  - 매칭 성공: {stats['matched']}개 ({stats['matched']/total_jobs*100:.1f}%)\n"
            f"  - 매칭 실패: {stats['unmatched']}개 ({stats['unmatched']/total_jobs*100:.1f}%)\n"
            f"  - 처리 시간: {elapsed_time:.1f}초 ({total_jobs/elapsed_time:.1f}개/초)\n"
        )
        
        if stats['matched'] > 0:
            print(f"  - 평균 추출 스킬 수: {stats['total_skills']/stats['matched']:.1f}개")
        
        if stats['errors']:
            print(f"  - 오류 유형:")
            for error_type, count in stats['errors'].items():
                print(f"    - {error_type}: {count}개")
        
        print(f"  - 저장 위치: {output_path}\n{'='*60}\n")

        return stats['matched'], stats['unmatched']


def main():
    """
    스킬셋 추출 전체 파이프라인 메인 함수
    (경로 설정, matcher 생성/파일 순회, 전체 진행상황/요약 출력)
    """
    current_file = Path(__file__).resolve()
    backend_model_dir = current_file.parents[3]

<<<<<<< HEAD
    # description.json 파일 찾기 (여러 위치 확인)
    # 1. 프로젝트 루트의 data 디렉토리
    # 우선순위: new_job_description.json -> description.json
    description_path = backend_model_dir / 'data' / 'new_job_description.json'
    if not description_path.exists():
        description_path = backend_model_dir / 'data' / 'description.json'
  
    
    # data/output 디렉토리 경로 (jobs 파일들이 있는 곳)
    data_dir = backend_model_dir / 'data' / 'output'
=======
    description_path = backend_model_dir / 'data' / 'description.json'
    data_dir = backend_model_dir / 'data'
>>>>>>> jaemin

    print("\n" + "=" * 60)
    print(":로켓: LLM 기반 Skill Set 추출 시작")
    print("=" * 60)
    print(f":클립보드: 스킬 목록 파일: {description_path}")
    print(f":열린_파일_폴더: 데이터 디렉토리: {data_dir}")

    if not description_path.exists():
        print(f"\n:x: 오류: description.json 파일을 찾을 수 없습니다.\n   찾는 경로: {description_path}\n   파일이 존재하는지 확인해주세요.")
        return
    if not data_dir.exists():
        print(f"\n:x: 오류: 데이터 디렉토리를 찾을 수 없습니다.\n   찾는 경로: {data_dir}\n   디렉토리가 존재하는지 확인해주세요.")
        return

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

    jobs_files = list(data_dir.glob('*_jobs.json'))
    if not jobs_files:
        print(f"\n:경고:  {data_dir}에서 *_jobs.json 파일을 찾을 수 없습니다.")
        return

    print(f":파일_폴더: 처리할 파일 수: {len(jobs_files)}")
    total_matched, total_unmatched = 0, 0

    async def process_all_files():
        nonlocal total_matched, total_unmatched
        for jobs_file in jobs_files:
            try:
                matched, unmatched = await matcher.process_jobs_file_async(str(jobs_file), str(jobs_file))
                total_matched += matched
                total_unmatched += unmatched
            except Exception as e:
                print(f"\n:x: 파일 처리 중 오류: {jobs_file}\n   오류 메시지: {str(e)}")
                continue

    asyncio.run(process_all_files())

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