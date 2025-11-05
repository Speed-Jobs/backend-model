"""
LG 채용공고 크롤러 (실제 API 사용)

API: https://api.careers.lg.com/rmk/job/retrieveJobNoticesList
각 공고의 jobNoticeId를 추출하여 상세 페이지 URL 생성
"""

import requests
import json
import time
from typing import List, Dict, Optional
from datetime import datetime


class LGCareerCrawler:
    def __init__(self):
        self.base_url = "https://api.careers.lg.com"
        self.list_endpoint = "/rmk/job/retrieveJobNoticesList"
        self.detail_url_template = "https://careers.lg.com/apply/detail?id={}"
        
        self.headers = {
            'authority': 'api.careers.lg.com',
            'accept': 'application/json, text/plain, */*',
            'accept-encoding': 'gzip, deflate, br, zstd',
            'accept-language': 'ko,en-US;q=0.9,en;q=0.8,ko-KR;q=0.7',
            'content-type': 'application/json',
            'cookie': 'SCOUTER=x5uv8k0el4hj7o; rmkonba=YjQ3ODMwZWUtY2I0Ny00NmNiLWE0OTMtYTI1N2I0OWIxNTg4',
            'origin': 'https://careers.lg.com',
            'referer': 'https://careers.lg.com/',
            'sec-ch-ua': '"Google Chrome";v="141", "Not?A_Brand";v="8", "Chromium";v="141"',
            'sec-ch-ua-mobile': '?0',
            'sec-ch-ua-platform': '"Windows"',
            'sec-fetch-dest': 'empty',
            'sec-fetch-mode': 'cors',
            'sec-fetch-site': 'same-site',
            'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/141.0.0.0 Safari/537.36'
        }
    
    def get_job_list(self, page: int = 1, page_size: int = 20) -> Dict:
        """
        채용공고 리스트 가져오기
        
        Args:
            page: 페이지 번호 (1부터 시작)
            page_size: 페이지당 공고 수
            
        Returns:
            API 응답 데이터
        """
        url = f"{self.base_url}{self.list_endpoint}"
        
        # 실제 API 요청 바디 (개발자 도구에서 확인한 실제 payload)
        payload = {
            "lnbSearch": "",
            "hashTagText": "",
            "recDate": "CREATION_DATE",
            "order": "DESC",
            "careerList": [],
            "companyCodeList": [],
            "desireLocList": [],
            "jobGroupList": []
        }
        
        try:
            print(f"📡 API 호출 중... (페이지: {page})")
            response = requests.post(
                url,
                headers=self.headers,
                json=payload,
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                return data
            else:
                print(f"❌ API 호출 실패: {response.status_code}")
                print(f"응답: {response.text}")
                return {}
                
        except requests.exceptions.RequestException as e:
            print(f"❌ 요청 오류: {e}")
            return {}
    
    def parse_job_list(self, api_response: Dict) -> List[Dict]:
        """
        API 응답에서 채용공고 정보 파싱
        
        Args:
            api_response: API 응답 데이터
            
        Returns:
            파싱된 채용공고 리스트
        """
        jobs = []
        
        try:
            # data.jobNoticeList에서 공고 목록 추출
            job_notice_list = api_response.get('data', {}).get('jobNoticeList', [])
            
            for job in job_notice_list:
                job_info = {
                    'id': job.get('jobNoticeId'),
                    'title': job.get('jobNoticeName'),
                    'company': job.get('companyName'),
                    'career_type': job.get('careerTypeName'),
                    'job_group': job.get('jobGroupName'),
                    'status': job.get('noticeStatusName'),
                    'deadline': job.get('recEndDateTime'),
                    'url': self.detail_url_template.format(job.get('jobNoticeId'))
                }
                
                jobs.append(job_info)
            
            return jobs
            
        except Exception as e:
            print(f"❌ 파싱 오류: {e}")
            return []
    
    def crawl_page(self, page: int = 1, page_size: int = 20) -> List[Dict]:
        """
        특정 페이지의 채용공고 크롤링
        
        Args:
            page: 페이지 번호
            page_size: 페이지당 공고 수
            
        Returns:
            채용공고 리스트
        """
        # API 호출
        api_response = self.get_job_list(page, page_size)
        
        if not api_response:
            return []
        
        # 데이터 파싱
        jobs = self.parse_job_list(api_response)
        
        return jobs
    
    def crawl_all_pages(self, max_pages: int = None) -> List[Dict]:
        """
        모든 페이지의 채용공고 크롤링
        
        Args:
            max_pages: 최대 페이지 수 (None이면 전체)
            
        Returns:
            전체 채용공고 리스트
        """
        all_jobs = []
        page = 1
        
        print("🚀 LG 채용공고 크롤링 시작...\n")
        
        while True:
            # 페이지 크롤링
            jobs = self.crawl_page(page)
            
            if not jobs:
                print(f"✅ 더 이상 데이터가 없습니다. (총 {page-1}페이지)")
                break
            
            all_jobs.extend(jobs)
            print(f"📄 {page}페이지: {len(jobs)}개 수집 (누적: {len(all_jobs)}개)")
            
            # 최대 페이지 수 체크
            if max_pages and page >= max_pages:
                print(f"✅ 설정한 최대 페이지({max_pages})에 도달했습니다.")
                break
            
            page += 1
            
            # 서버 부담 줄이기 위한 딜레이
            time.sleep(1)
        
        print(f"\n✅ 크롤링 완료! 총 {len(all_jobs)}개 공고 수집")
        return all_jobs
    
    def print_job_info(self, jobs: List[Dict]):
        """채용공고 정보를 예쁘게 출력"""
        print("\n" + "="*80)
        print("📋 LG 채용공고 목록")
        print("="*80 + "\n")
        
        for idx, job in enumerate(jobs, 1):
            print(f"{idx}. {job['title']}")
            print(f"   회사: {job['company']}")
            print(f"   구분: {job['career_type']}")
            print(f"   직군: {job['job_group']}")
            print(f"   상태: {job['status']}")
            print(f"   마감: {job['deadline']}")
            print(f"   🔗 {job['url']}")
            print()
    
    def save_to_json(self, jobs: List[Dict], filename: str = "lg_jobs.json"):
        """JSON 파일로 저장"""
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(jobs, f, ensure_ascii=False, indent=2)
            print(f"저장 완료: {filename}")
        except Exception as e:
            print(f"저장 실패: {e}")
    
    def save_to_csv(self, jobs: List[Dict], filename: str = "lg_jobs.csv"):
        """CSV 파일로 저장"""
        try:
            import csv
            
            if not jobs:
                print("저장할 데이터가 없습니다.")
                return
            
            with open(filename, 'w', encoding='utf-8-sig', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=jobs[0].keys())
                writer.writeheader()
                writer.writerows(jobs)
            
            print(f"저장 완료: {filename}")
        except Exception as e:
            print(f"저장 실패: {e}")
    
    def get_job_detail(self, job_id: str) -> Optional[str]:
        """
        개별 채용공고 상세 페이지 URL 반환
        
        Args:
            job_id: 채용공고 ID (jobNoticeId)
            
        Returns:
            상세 페이지 URL
        """
        return self.detail_url_template.format(job_id)
    
    def filter_jobs(self, jobs: List[Dict], **filters) -> List[Dict]:
        """
        채용공고 필터링
        
        Args:
            jobs: 채용공고 리스트
            **filters: 필터 조건 (company, career_type, status 등)
            
        Returns:
            필터링된 채용공고 리스트
        """
        filtered = jobs
        
        for key, value in filters.items():
            if value:
                filtered = [job for job in filtered if value.lower() in str(job.get(key, '')).lower()]
        
        return filtered


def main():
    """실행 예제"""
    
    print("="*80)
    print("LG 채용공고 크롤러")
    print("="*80)
    print()
    
    # 크롤러 초기화
    crawler = LGCareerCrawler()
    
    # 방법 1: 첫 페이지만 가져오기
    print("📋 첫 페이지 크롤링 중...\n")
    jobs = crawler.crawl_page(page=1, page_size=20)
    
    if jobs:
        crawler.print_job_info(jobs)
        
        # 특정 회사만 필터링
        # lg_electronics = crawler.filter_jobs(jobs, company="LG전자")
        # crawler.print_job_info(lg_electronics)
    
    # 방법 2: 모든 페이지 가져오기 (최대 3페이지)
    # all_jobs = crawler.crawl_all_pages(max_pages=3)
    # crawler.print_job_info(all_jobs)
    
    # 방법 3: 저장하기
    if jobs:
        crawler.save_to_json(jobs, "lg_jobs.json")
        crawler.save_to_csv(jobs, "lg_jobs.csv")
    
    # 방법 4: 개별 공고 URL 생성
    if jobs:
        print("\n" + "="*80)
        print("🔗 상세 페이지 URL 예시")
        print("="*80 + "\n")
        
        for i, job in enumerate(jobs[:5], 1):  # 처음 5개만
            detail_url = crawler.get_job_detail(job['id'])
            print(f"{i}. {job['title']}")
            print(f"   {detail_url}\n")


if __name__ == "__main__":
    main()