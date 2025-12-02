"""
통합 데이터 파이프라인

전체 프로세스:
1. 크롤러 실행 (call_all_crawler.py) → data/output/*_jobs.json 생성
2. 스킬셋 추출 (extract_skillsets_async.py) → skill_set_info 추가 (비동기)
3. 직무 매칭 (job_matching_system.py) → 매칭 결과 생성

각 단계는 순차적으로 실행되며, 이전 단계의 출력이 다음 단계의 입력이 됩니다.
에러 발생 시 즉시 중단됩니다.
"""

import sys
import logging
import time
import subprocess
import os
import asyncio
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict, Any
import json

# 프로젝트 루트를 sys.path에 추가
backend_root = Path(__file__).resolve().parents[3]
if str(backend_root) not in sys.path:
    sys.path.insert(0, str(backend_root))

# 각 단계의 모듈 import
from app.services.crawler.call_all_crawler import run_all_crawlers_sequentially
from app.utils.parser.extract_skillsets import main as extract_skillsets_main_async
from app.core.job_matching.job_matching_system import JobMatchingSystem
from app.scripts.db.insert_post_to_db import main as insert_post_main


# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)


class DataPipeline:
    """통합 데이터 파이프라인 관리 클래스"""
    
    def __init__(
        self,
        skip_crawler: bool = False,
        skip_skillset: bool = False,
        skip_post_insert: bool = False
    ):
        """
        파이프라인 초기화
        
        Args:
            skip_crawler: True면 크롤링 단계 건너뛰기 (테스트용)
            skip_skillset: True면 스킬셋 추출 단계 건너뛰기 (테스트용)
            skip_post_insert: True면 채용공고 DB 삽입 단계 건너뛰기
        """
        self.skip_crawler = skip_crawler
        self.skip_skillset = skip_skillset
        self.skip_post_insert = skip_post_insert
        self.backend_root = Path(__file__).resolve().parents[3]
        self.data_output_dir = self.backend_root / 'data' / 'output'
        self.data_dir = self.backend_root / 'data'
        
        # 타임스탬프 (파이프라인 실행 시점 기록)
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 결과 저장용
        self.pipeline_results: Dict[str, Any] = {
            'timestamp': self.timestamp,
            'crawler': None,
            'skillset_extraction': None,
            'job_matching': None,
            'post_insert': None,
        }
    
    def _check_prerequisites(self) -> bool:
        """
        파이프라인 실행 전 사전 체크
        
        Returns:
            True: 모든 사전 조건 충족
            False: 사전 조건 미충족
        """
        logger.info("\n" + "="*80)
        logger.info("🔍 사전 조건 체크")
        logger.info("="*80)
        
        all_passed = True
        
        # 1. OPENAI_API_KEY 환경변수 체크 (스킬셋 추출에 필요)
        if not self.skip_skillset:
            if not os.getenv('OPENAI_API_KEY'):
                logger.error("❌ OPENAI_API_KEY 환경변수가 설정되지 않았습니다.")
                logger.error("   스킬셋 추출에 필요합니다. .env 파일을 확인하세요.")
                all_passed = False
            else:
                logger.info("✅ OPENAI_API_KEY 환경변수 확인")
        
        # 2. Playwright 브라우저 설치 여부 체크 (크롤링에 필요)
        if not self.skip_crawler:
            playwright_installed = self._check_playwright_browsers()
            if not playwright_installed:
                logger.error("❌ Playwright 브라우저가 설치되지 않았습니다.")
                logger.error("   다음 명령어를 실행하세요:")
                logger.error("   playwright install")
                logger.error("   또는: python -m playwright install")
                all_passed = False
            else:
                logger.info("✅ Playwright 브라우저 설치 확인")
        
        # 3. 필수 디렉토리 존재 여부 체크
        if not self.data_dir.exists():
            logger.error(f"❌ 데이터 디렉토리가 없습니다: {self.data_dir}")
            all_passed = False
        else:
            logger.info(f"✅ 데이터 디렉토리 확인: {self.data_dir}")
        
        # 4. 직무 정의 파일 존재 여부 체크
        job_desc_file = self.data_dir / 'new_job_description.json'
        job_desc_file_alt = self.data_dir / 'description.json'
        
        if not job_desc_file.exists() and not job_desc_file_alt.exists():
            logger.error(f"❌ 직무 정의 파일이 없습니다:")
            logger.error(f"   {job_desc_file} 또는")
            logger.error(f"   {job_desc_file_alt}")
            all_passed = False
        else:
            logger.info("✅ 직무 정의 파일 확인")
        
        # 5. output 디렉토리 생성
        self.data_output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"✅ 출력 디렉토리 확인: {self.data_output_dir}")
        
        logger.info("="*80)
        
        if not all_passed:
            logger.error("\n❌ 사전 조건을 충족하지 못했습니다. 위의 문제를 해결한 후 다시 실행하세요.")
        
        return all_passed
    
    def _check_playwright_browsers(self) -> bool:
        """
        Playwright 브라우저 설치 여부 확인
        
        Returns:
            True: 브라우저 설치됨
            False: 브라우저 미설치
        """
        try:
            # playwright 브라우저 목록 확인 시도
            result = subprocess.run(
                [sys.executable, '-m', 'playwright', 'install', '--dry-run'],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            # dry-run이 성공적으로 실행되고 "is already installed" 메시지가 있으면 설치됨
            # 또는 AppData 경로에 playwright 브라우저 폴더가 있는지 직접 확인
            if sys.platform == 'win32':
                playwright_dir = Path.home() / 'AppData' / 'Local' / 'ms-playwright'
            elif sys.platform == 'darwin':
                playwright_dir = Path.home() / 'Library' / 'Caches' / 'ms-playwright'
            else:
                playwright_dir = Path.home() / '.cache' / 'ms-playwright'
            
            if playwright_dir.exists():
                # chromium 폴더가 있는지 확인
                chromium_dirs = list(playwright_dir.glob('chromium*'))
                return len(chromium_dirs) > 0
            
            return False
            
        except Exception as e:
            logger.warning(f"⚠️ Playwright 설치 확인 중 오류: {e}")
            # 확인 실패 시 일단 진행 (실제 실행 시 에러 발생하면 그때 중단)
            return True
    
    def run(self) -> Dict[str, Any]:
        """
        전체 파이프라인 실행
        
        Returns:
            각 단계별 실행 결과를 담은 딕셔너리
        """
        logger.info("="*80)
        logger.info(f"🚀 데이터 파이프라인 시작 ({self.timestamp})")
        logger.info("="*80)
        
        # 사전 조건 체크
        if not self._check_prerequisites():
            self.pipeline_results['status'] = 'failed'
            self.pipeline_results['error'] = '사전 조건 미충족'
            return self.pipeline_results
        
        start_time = time.time()
        
        try:
            # Step 1: 크롤러 실행
            if not self.skip_crawler:
                success = self._run_crawler()
                if not success:
                    logger.error("\n❌ 크롤러 실행 실패. 파이프라인을 중단합니다.")
                    self.pipeline_results['status'] = 'failed'
                    return self.pipeline_results
            else:
                logger.info("\n[Step 1] 크롤러 실행 - SKIPPED")
                self.pipeline_results['crawler'] = {'status': 'skipped'}
            
            # Step 2: 스킬셋 추출 (비동기)
            if not self.skip_skillset:
                success = self._run_skillset_extraction()
                if not success:
                    logger.error("\n❌ 스킬셋 추출 실패. 파이프라인을 중단합니다.")
                    self.pipeline_results['status'] = 'failed'
                    return self.pipeline_results
            else:
                logger.info("\n[Step 2] 스킬셋 추출 - SKIPPED")
                self.pipeline_results['skillset_extraction'] = {'status': 'skipped'}
            
            # Step 3: 직무 매칭
            success = self._run_job_matching()
            if not success:
                logger.error("\n❌ 직무 매칭 실패. 파이프라인을 중단합니다.")
                self.pipeline_results['status'] = 'failed'
                return self.pipeline_results
            
            # Step 4: 채용공고 DB 삽입 (옵션)
            if not self.skip_post_insert:
                success = self._run_post_insert()
                if not success:
                    logger.error("\n❌ 채용공고 DB 삽입 실패. 파이프라인을 중단합니다.")
                    self.pipeline_results['status'] = 'failed'
                    return self.pipeline_results
            else:
                logger.info("\n[Step 4] 채용공고 DB 삽입 - SKIPPED")
                self.pipeline_results['post_insert'] = {'status': 'skipped'}
            
            # 총 실행 시간
            duration = time.time() - start_time
            self.pipeline_results['total_duration_minutes'] = round(duration / 60, 2)
            
            logger.info("\n" + "="*80)
            logger.info(f"✅ 파이프라인 완료 (총 {duration/60:.1f}분)")
            logger.info("="*80)
            
            # 결과 요약 출력
            self._print_summary()
            
            # 결과를 JSON 파일로 저장
            self._save_pipeline_results()
            
            return self.pipeline_results
            
        except Exception as e:
            logger.error(f"\n{'='*80}")
            logger.error(f"❌ 파이프라인 실행 중 예외 발생")
            logger.error(f"{'='*80}")
            logger.error(f"오류 메시지: {e}")
            logger.error(f"{'='*80}\n")
            
            import traceback
            traceback.print_exc()
            
            self.pipeline_results['error'] = str(e)
            self.pipeline_results['status'] = 'failed'
            
            return self.pipeline_results
    
    def _run_crawler(self) -> bool:
        """
        Step 1: 크롤러 실행
        
        Returns:
            True: 성공
            False: 실패
        """
        logger.info("\n" + "="*80)
        logger.info("[Step 1/4] 크롤러 실행")
        logger.info("="*80)
        
        step_start = time.time()
        
        try:
            # call_all_crawler.py의 run_all_crawlers_sequentially() 실행
            run_all_crawlers_sequentially()
            
            duration = time.time() - step_start
            
            # 생성된 파일 확인
            job_files = list(self.data_output_dir.glob('*_jobs.json'))
            
            if len(job_files) == 0:
                logger.error("❌ 크롤링은 완료되었으나 생성된 파일이 없습니다.")
                self.pipeline_results['crawler'] = {
                    'status': 'failed',
                    'error': '생성된 파일 없음',
                    'duration_minutes': round(duration / 60, 2),
                }
                return False
            
            self.pipeline_results['crawler'] = {
                'status': 'success',
                'duration_minutes': round(duration / 60, 2),
                'output_files': [f.name for f in job_files],
                'output_count': len(job_files),
            }
            
            logger.info(f"\n✅ 크롤링 완료 ({duration/60:.1f}분)")
            logger.info(f"   생성된 파일: {len(job_files)}개")
            for f in job_files:
                logger.info(f"   - {f.name}")
            
            return True
            
        except Exception as e:
            duration = time.time() - step_start
            logger.error(f"\n{'='*80}")
            logger.error(f"❌ 크롤러 실행 중 오류 발생")
            logger.error(f"{'='*80}")
            logger.error(f"오류 타입: {type(e).__name__}")
            logger.error(f"오류 메시지: {e}")
            logger.error(f"소요 시간: {duration/60:.1f}분")
            logger.error(f"{'='*80}\n")
            
            import traceback
            traceback.print_exc()
            
            self.pipeline_results['crawler'] = {
                'status': 'failed',
                'error': str(e),
                'error_type': type(e).__name__,
                'duration_minutes': round(duration / 60, 2),
            }
            
            return False
    
    def _run_skillset_extraction(self) -> bool:
        """
        Step 2: 스킬셋 추출 (비동기 버전)
        
        Returns:
            True: 성공
            False: 실패
        """
        logger.info("\n" + "="*80)
        logger.info("[Step 2/4] 스킬셋 추출 (LLM 기반 - 비동기)")
        logger.info("="*80)
        
        step_start = time.time()
        
        try:
            # extract_skillsets_async.py의 main() 실행 (비동기)
            # asyncio.run()으로 비동기 함수 실행
            extract_skillsets_main_async()
            
            duration = time.time() - step_start
            
            # 처리된 파일 확인
            job_files = list(self.data_output_dir.glob('*_jobs.json'))
            
            if len(job_files) == 0:
                logger.error("❌ 처리할 파일이 없습니다. 크롤링이 정상적으로 완료되었는지 확인하세요.")
                self.pipeline_results['skillset_extraction'] = {
                    'status': 'failed',
                    'error': '처리할 파일 없음',
                    'duration_minutes': round(duration / 60, 2),
                }
                return False
            
            self.pipeline_results['skillset_extraction'] = {
                'status': 'success',
                'duration_minutes': round(duration / 60, 2),
                'processed_files': [f.name for f in job_files],
                'processed_count': len(job_files),
            }
            
            logger.info(f"\n✅ 스킬셋 추출 완료 ({duration/60:.1f}분)")
            logger.info(f"   처리된 파일: {len(job_files)}개")
            logger.info(f"   ⚡ 비동기 처리로 최대 30개씩 동시 실행")
            
            return True
            
        except Exception as e:
            duration = time.time() - step_start
            logger.error(f"\n{'='*80}")
            logger.error(f"❌ 스킬셋 추출 중 오류 발생")
            logger.error(f"{'='*80}")
            logger.error(f"오류 타입: {type(e).__name__}")
            logger.error(f"오류 메시지: {e}")
            logger.error(f"소요 시간: {duration/60:.1f}분")
            logger.error(f"{'='*80}\n")
            
            import traceback
            traceback.print_exc()
            
            self.pipeline_results['skillset_extraction'] = {
                'status': 'failed',
                'error': str(e),
                'error_type': type(e).__name__,
                'duration_minutes': round(duration / 60, 2),
            }
            
            return False
    
    def _run_job_matching(self) -> bool:
        """
        Step 3: 직무 매칭
        
        Returns:
            True: 성공
            False: 실패
        """
        logger.info("\n" + "="*80)
        logger.info("[Step 3/4] 직무 매칭 시스템")
        logger.info("="*80)
        
        step_start = time.time()
        
        try:
            # 로그 파일 경로 설정
            log_file = self.data_output_dir / f"job_matching_log_{self.timestamp}.txt"
            
            # JobMatchingSystem 초기화
            system = JobMatchingSystem(log_file=str(log_file))
            
            # 직무 정의 로드
            logger.info("\n[3-1] 직무 정의 로드")
            system.load_job_descriptions()
            
            # 학습 데이터 로드
            logger.info("\n[3-2] 학습 데이터 로드")
            job_files = list(self.data_output_dir.glob('*_jobs.json'))
            training_files = [str(f) for f in job_files]
            system.load_training_data(training_files)
            
            # 그래프 구축
            logger.info("\n[3-3] 그래프 구축")
            system.build_graph()
            
            # Matchers 초기화 (Louvain + SBERT)
            logger.info("\n[3-4] Matchers 초기화 (Louvain + SBERT)")
            system.build_matchers()
            
            logger.info("\n[3-5] 직무 매칭 실행")
            logger.info("="*80)
            
            # 각 파일에 대해 매칭 실행
            all_matching_results = []
            
            for job_file in job_files:
                logger.info(f"\n처리 중: {job_file.name}")
                
                try:
                    results = system.match_company_jobs(
                        str(job_file),
                        ppr_top_n=20,
                        final_top_k=2,
                    )
                    
                    all_matching_results.append({
                        'file': job_file.name,
                        'results': results,
                    })
                    
                    # 매칭 결과를 원본 파일에 병합
                    self._merge_matching_results_to_original(job_file, results)
                    
                    logger.info(f"   ✅ 매칭 결과가 원본 파일에 병합되었습니다: {job_file.name}")
                    
                except Exception as e:
                    logger.error(f"   ❌ {job_file.name} 처리 실패: {e}")
                    import traceback
                    traceback.print_exc()
                    continue
            
            duration = time.time() - step_start
            
            self.pipeline_results['job_matching'] = {
                'status': 'success',
                'duration_minutes': round(duration / 60, 2),
                'processed_files': len(all_matching_results),
                'log_file': log_file.name,
            }
            
            logger.info(f"\n✅ 직무 매칭 완료 ({duration/60:.1f}분)")
            logger.info(f"   로그 파일: {log_file.name}")
            
            return True
            
        except Exception as e:
            duration = time.time() - step_start
            logger.error(f"\n{'='*80}")
            logger.error(f"❌ 직무 매칭 중 오류 발생")
            logger.error(f"{'='*80}")
            logger.error(f"오류 타입: {type(e).__name__}")
            logger.error(f"오류 메시지: {e}")
            logger.error(f"소요 시간: {duration/60:.1f}분")
            logger.error(f"{'='*80}\n")
            
            import traceback
            traceback.print_exc()
            
            self.pipeline_results['job_matching'] = {
                'status': 'failed',
                'error': str(e),
                'error_type': type(e).__name__,
                'duration_minutes': round(duration / 60, 2),
            }
            
            return False
    
    def _run_post_insert(self) -> bool:
        """
        Step 4: 채용공고 DB 삽입
        
        Returns:
            True: 성공
            False: 실패
        """
        logger.info("\n" + "="*80)
        logger.info("[Step 4/4] 채용공고 DB 삽입")
        logger.info("="*80)
        
        step_start = time.time()
        
        try:
            # data/output/*_jobs.json 파일 확인
            job_files = list(self.data_output_dir.glob('*_jobs.json'))
            
            if len(job_files) == 0:
                logger.error("❌ 삽입할 파일이 없습니다. 이전 단계가 정상적으로 완료되었는지 확인하세요.")
                self.pipeline_results['post_insert'] = {
                    'status': 'failed',
                    'error': '삽입할 파일 없음'
                }
                return False
            
            logger.info(f"삽입할 파일: {len(job_files)}개")
            for f in job_files:
                logger.info(f"  - {f.name}")
            
            # insert_post_to_db의 main() 실행
            insert_post_main()
            
            duration = time.time() - step_start
            
            self.pipeline_results['post_insert'] = {
                'status': 'success',
                'duration_minutes': round(duration / 60, 2),
                'inserted_files': [f.name for f in job_files],
                'file_count': len(job_files),
            }
            
            logger.info(f"\n✅ 채용공고 DB 삽입 완료 ({duration/60:.1f}분)")
            
            return True
            
        except Exception as e:
            duration = time.time() - step_start
            logger.error(f"\n{'='*80}")
            logger.error(f"❌ 채용공고 DB 삽입 중 오류 발생")
            logger.error(f"{'='*80}")
            logger.error(f"오류 타입: {type(e).__name__}")
            logger.error(f"오류 메시지: {e}")
            logger.error(f"소요 시간: {duration/60:.1f}분")
            logger.error(f"{'='*80}\n")
            
            import traceback
            traceback.print_exc()
            
            self.pipeline_results['post_insert'] = {
                'status': 'failed',
                'error': str(e),
                'error_type': type(e).__name__,
                'duration_minutes': round(duration / 60, 2),
            }
            
            return False
    
    def _merge_matching_results_to_original(self, job_file: Path, results: List[Dict]) -> None:
        """
        매칭 결과를 원본 파일에 병합
        
        Args:
            job_file: 원본 크롤링 데이터 파일 경로
            results: 매칭 결과 리스트
        """
        # 원본 파일 로드
        with open(job_file, 'r', encoding='utf-8') as f:
            original_jobs = json.load(f)
        
        # 매칭 결과를 딕셔너리로 변환 (company, title, url을 키로 사용)
        matching_dict = {}
        for result_item in results:
            if result_item.get('db_result'):
                posting = result_item['posting']
                # 키 생성: company + title + url의 조합
                key = self._create_job_key(posting.company, posting.title, posting.url)
                matching_dict[key] = result_item['db_result']
        
        # 원본 데이터에 매칭 결과 병합
        merged_count = 0
        for job in original_jobs:
            key = self._create_job_key(
                job.get('company', ''),
                job.get('title', ''),
                job.get('url', '')
            )
            
            if key in matching_dict:
                match_result = matching_dict[key]
                # sim_position, sim_industry, sim_score, sim_skill_matching 추가
                job['sim_position'] = match_result.get('position')
                job['sim_industry'] = match_result.get('industry')
                job['sim_score'] = match_result.get('sim_score')
                job['sim_skill_matching'] = match_result.get('sim_skill_matching', [])
                merged_count += 1
        
        # 병합된 데이터를 원본 파일에 저장 (덮어쓰기)
        with open(job_file, 'w', encoding='utf-8') as f:
            json.dump(original_jobs, f, ensure_ascii=False, indent=2)
        
        logger.info(f"   📊 {merged_count}/{len(original_jobs)}개 채용공고에 매칭 결과 병합 완료")
        
        # 별도로 매칭 결과만 저장 (백업/참고용)
        backup_file = self.data_output_dir / f"matched_{job_file.stem}_{self.timestamp}.json"
        db_results = []
        for result_item in results:
            if result_item.get('db_result'):
                db_entry = {
                    'company': result_item['posting'].company,
                    'title': result_item['posting'].title,
                    'url': result_item['posting'].url,
                    'skills': result_item['posting'].skills,
                    **result_item['db_result']
                }
                db_results.append(db_entry)
        
        with open(backup_file, 'w', encoding='utf-8') as f:
            json.dump(db_results, f, ensure_ascii=False, indent=2)
        
        logger.info(f"   💾 매칭 결과 백업: {backup_file.name}")
    
    def _create_job_key(self, company: str, title: str, url: str) -> str:
        """
        채용공고를 고유하게 식별하는 키 생성
        
        Args:
            company: 회사명
            title: 채용공고 제목
            url: URL
            
        Returns:
            고유 키 문자열
        """
        # 공백 제거 및 소문자 변환으로 정규화
        normalized_company = company.strip().lower()
        normalized_title = title.strip().lower()
        normalized_url = url.strip().lower()
        
        return f"{normalized_company}||{normalized_title}||{normalized_url}"
    
    def _print_summary(self):
        """파이프라인 실행 결과 요약 출력"""
        logger.info("\n" + "="*80)
        logger.info("📊 파이프라인 실행 요약")
        logger.info("="*80)
        
        for step_name, result in self.pipeline_results.items():
            if step_name in ['timestamp', 'total_duration_minutes']:
                continue
            
            if result and isinstance(result, dict):
                status = result.get('status', 'unknown')
                status_emoji = "✅" if status == 'success' else "⏭️" if status == 'skipped' else "❌"
                
                logger.info(f"\n{step_name.upper()}:")
                logger.info(f"  상태: {status_emoji} {status}")
                
                if 'duration_minutes' in result:
                    logger.info(f"  소요 시간: {result['duration_minutes']:.1f}분")
                
                if 'output_count' in result:
                    logger.info(f"  생성 파일: {result['output_count']}개")
                
                if 'processed_count' in result:
                    logger.info(f"  처리 파일: {result['processed_count']}개")
                
                if 'error' in result:
                    logger.info(f"  오류: {result['error']}")
        
        if 'total_duration_minutes' in self.pipeline_results:
            logger.info(f"\n총 실행 시간: {self.pipeline_results['total_duration_minutes']:.1f}분")
        
        logger.info("="*80)
    
    def _save_pipeline_results(self):
        """파이프라인 실행 결과를 JSON 파일로 저장"""
        result_file = self.data_output_dir / f"pipeline_results_{self.timestamp}.json"
        
        # NewJobPosting 등의 객체를 JSON 직렬화 가능하도록 변환
        serializable_results = self._make_serializable(self.pipeline_results)
        
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(serializable_results, f, ensure_ascii=False, indent=2)
        
        logger.info(f"\n💾 파이프라인 결과 저장: {result_file.name}")
    
    def _make_serializable(self, obj):
        """객체를 JSON 직렬화 가능한 형태로 변환"""
        if isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(item) for item in obj]
        elif hasattr(obj, '__dict__'):
            return self._make_serializable(obj.__dict__)
        else:
            return obj


def main():
    """
    메인 실행 함수
    
    명령줄 인자:
        --skip-crawler: 크롤링 단계 건너뛰기
        --skip-skillset: 스킬셋 추출 단계 건너뛰기
        --skip-post-insert: 채용공고 DB 삽입 단계 건너뛰기
    
    Note:
        직무 정의 DB 삽입(insert_job_description_to_db)은 별도로 실행하세요:
        python -m app.scripts.db.insert_job_description_to_db
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='통합 데이터 파이프라인 실행')
    parser.add_argument('--skip-crawler', action='store_true', help='크롤링 단계 건너뛰기')
    parser.add_argument('--skip-skillset', action='store_true', help='스킬셋 추출 단계 건너뛰기')
    parser.add_argument('--skip-post-insert', action='store_true', help='채용공고 DB 삽입 단계 건너뛰기')
    
    args = parser.parse_args()
    
    # 파이프라인 실행
    pipeline = DataPipeline(
        skip_crawler=args.skip_crawler,
        skip_skillset=args.skip_skillset,
        skip_post_insert=args.skip_post_insert,
    )
    
    results = pipeline.run()
    
    # 성공 여부에 따라 종료 코드 반환
    if results.get('status') == 'failed':
        sys.exit(1)
    else:
        sys.exit(0)


if __name__ == '__main__':
    main()