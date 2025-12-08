"""
JSON 파일에서 채용 공고 데이터를 읽어 데이터베이스에 삽입하는 스크립트 (개선 버전)
각 공고마다 개별적으로 처리하여 일부 에러가 전체 데이터에 영향을 주지 않도록 함
"""
import json
import os
import glob
from datetime import datetime
from typing import List, Dict, Any
from sqlalchemy.orm import Session
from sqlalchemy.exc import IntegrityError

from app.db.config.base import SessionLocal, engine
from app.models.company import Company
from app.models.position import Position
from app.models.industry import Industry
from app.models.post import Post
from app.models.skill import Skill
from app.models.post_skill import PostSkill
from app.models.industry_skill import IndustrySkill
from app.models.position_skill import PositionSkill


def parse_date(date_str: str) -> datetime:
    """날짜 문자열을 datetime 객체로 변환"""
    if not date_str:
        return None
    try:
        return datetime.strptime(date_str, "%Y-%m-%d")
    except ValueError:
        return None


def get_or_create_company(db: Session, company_name: str, company_location: str) -> Company:
    """회사를 조회하거나 새로 생성"""
    company = db.query(Company).filter(Company.name == company_name).first()
    if not company:
        company = Company(name=company_name, location=company_location)
        db.add(company)
        db.flush()
        print(f"새 회사 생성: {company_name} (ID: {company.id})")
    return company


def get_or_create_skill(db: Session, skill_name: str) -> Skill:
    """스킬을 조회하거나 새로 생성"""
    skill = db.query(Skill).filter(Skill.name == skill_name).first()
    if not skill:
        skill = Skill(name=skill_name)
        db.add(skill)
        db.flush()
        print(f"새 스킬 생성: {skill_name} (ID: {skill.id})")
    return skill


def create_post(db: Session, job_data: Dict[str, Any], company: Company) -> tuple[Post, bool]:
    """
    채용 공고 생성
    Returns: (Post 객체, 새로 생성되었는지 여부)
    """
    # 이미 존재하는 공고인지 확인 (source_url과 title로 중복 체크)
    existing_post = db.query(Post).filter(
        Post.source_url == job_data.get("url"),
        Post.title == job_data.get("title", "")
    ).first()
    if existing_post:
        print(f"이미 존재하는 공고: {job_data.get('title')} (URL: {job_data.get('url')})")
        return existing_post, False  # 기존 공고, 스킬은 여전히 추가 가능

    # meta_data 구성
    meta_data = {}
    if "meta_data" in job_data:
        meta_data = job_data["meta_data"]

    # tech_stack이 별도로 있으면 meta_data에 추가
    if "tech_stack" in job_data:
        meta_data["tech_stack"] = job_data["tech_stack"]

    # Industry ID 조회 (매칭 결과에서)
    industry_id = None
    sim_position = job_data.get("sim_position")
    sim_industry = job_data.get("sim_industry")
    
    if sim_position and sim_industry:
        # Position 조회
        position = db.query(Position).filter(Position.name == sim_position).first()
        if position:
            # Industry 조회 (해당 Position에 속한 Industry)
            industry = db.query(Industry).filter(
                Industry.name == sim_industry,
                Industry.position_id == position.id
            ).first()
            if industry:
                industry_id = industry.id
                print(f"  - 매칭된 산업: {sim_industry} (Industry ID: {industry_id})")
            else:
                print(f"  ⚠️ Industry '{sim_industry}'를 Position '{sim_position}' 하에서 찾을 수 없습니다.")
        else:
            print(f"  ⚠️ Position '{sim_position}'를 찾을 수 없습니다.")

    # Post 객체 생성
    post = Post(
        title=job_data.get("title", ""),
        employment_type=job_data.get("employment_type"),
        experience=job_data.get("experience"),
        work_type=job_data.get("work_type"),
        description=job_data.get("description"),
        meta_data=meta_data if meta_data else None,
        posted_at=parse_date(job_data.get("posted_date")),
        close_at=parse_date(job_data.get("expired_date")),
        crawled_at=parse_date(job_data.get("crawl_date")) or datetime.now(),
        source_url=job_data.get("url", ""),
        url_hash=None,  # 일단 null로 설정
        screenshot_url=job_data.get("screenshots", {}).get("combined"),
        company_id=company.id,
        industry_id=industry_id  # 매칭된 Industry ID 할당
    )

    db.add(post)
    db.flush()
    print(f"새 공고 생성: {post.title} (ID: {post.id})")
    return post, True


def create_post_skills(db: Session, post: Post, skill_names: List[str]) -> int:
    """
    채용 공고와 스킬의 관계 생성
    Returns: 새로 추가된 스킬 연결 개수
    """
    added_count = 0
    for skill_name in skill_names:
        if not skill_name or skill_name.strip() == "":
            continue

        skill = get_or_create_skill(db, skill_name.strip())

        # 이미 관계가 존재하는지 확인
        existing_relation = db.query(PostSkill).filter(
            PostSkill.post_id == post.id,
            PostSkill.skill_id == skill.id
        ).first()

        if not existing_relation:
            post_skill = PostSkill(
                post_id=post.id,
                skill_id=skill.id
            )
            db.add(post_skill)
            print(f"  - 스킬 연결: {skill_name} -> Post #{post.id}")
            added_count += 1

    return added_count


def create_industry_skills(db: Session, industry_id: int, skill_names: List[str]) -> int:
    """
    산업과 스킬의 관계 생성
    Returns: 새로 추가된 스킬 연결 개수
    """
    if not industry_id:
        return 0

    added_count = 0
    for skill_name in skill_names:
        if not skill_name or skill_name.strip() == "":
            continue

        skill = get_or_create_skill(db, skill_name.strip())

        # 이미 관계가 존재하는지 확인
        existing_relation = db.query(IndustrySkill).filter(
            IndustrySkill.industry_id == industry_id,
            IndustrySkill.skill_id == skill.id
        ).first()

        if not existing_relation:
            industry_skill = IndustrySkill(
                industry_id=industry_id,
                skill_id=skill.id
            )
            db.add(industry_skill)
            print(f"  - 스킬 연결: {skill_name} -> Industry #{industry_id}")
            added_count += 1

    return added_count


def create_position_skills(db: Session, position_id: int, skill_names: List[str]) -> int:
    """
    직무와 스킬의 관계 생성
    Returns: 새로 추가된 스킬 연결 개수
    """
    if not position_id:
        return 0

    added_count = 0
    for skill_name in skill_names:
        if not skill_name or skill_name.strip() == "":
            continue

        skill = get_or_create_skill(db, skill_name.strip())

        # 이미 관계가 존재하는지 확인
        existing_relation = db.query(PositionSkill).filter(
            PositionSkill.position_id == position_id,
            PositionSkill.skill_id == skill.id
        ).first()

        if not existing_relation:
            position_skill = PositionSkill(
                position_id=position_id,
                skill_id=skill.id
            )
            db.add(position_skill)
            print(f"  - 스킬 연결: {skill_name} -> Position #{position_id}")
            added_count += 1

    return added_count


def process_single_job(db: Session, job_data: Dict[str, Any], idx: int) -> bool:
    """
    단일 job을 처리 (개별 트랜잭션)
    Returns: 성공 여부
    """
    try:
        # Savepoint 생성 (중첩 트랜잭션)
        savepoint = db.begin_nested()
        
        try:
            # 회사 조회/생성
            company_name = job_data.get("company", "Unknown Company")
            company_location = job_data.get("location", "Unknown Location")
            
            company = get_or_create_company(db, company_name, company_location=company_location)

            # 공고 생성 (또는 기존 공고 조회)
            post, is_new = create_post(db, job_data, company)

            # 스킬 정보 추출
            skill_names = []

            # skill_set_info에서 스킬 추출
            if "skill_set_info" in job_data and "skill_set" in job_data["skill_set_info"]:
                skill_names.extend(job_data["skill_set_info"]["skill_set"])

            # meta_data.tech_stack에서 스킬 추출
            if "meta_data" in job_data and "tech_stack" in job_data["meta_data"]:
                skill_names.extend(job_data["meta_data"]["tech_stack"])

            # 중복 제거
            skill_names = list(set(skill_names))

            # 스킬 연결 (기존 공고여도 새로운 스킬은 추가)
            if skill_names:
                # 1. PostSkill 관계 생성
                added = create_post_skills(db, post, skill_names)
                if not is_new and added > 0:
                    print(f"  기존 공고에 {added}개 스킬 추가됨")

                # 2. IndustrySkill 관계 생성
                if post.industry_id:
                    industry_added = create_industry_skills(db, post.industry_id, skill_names)
                    if industry_added > 0:
                        print(f"  Industry #{post.industry_id}에 {industry_added}개 스킬 추가됨")

                    # 3. PositionSkill 관계 생성
                    industry = db.query(Industry).filter(Industry.id == post.industry_id).first()
                    if industry and industry.position_id:
                        position_added = create_position_skills(db, industry.position_id, skill_names)
                        if position_added > 0:
                            print(f"  Position #{industry.position_id}에 {position_added}개 스킬 추가됨")

            # Savepoint 커밋
            savepoint.commit()
            return True
            
        except Exception as e:
            # 이 job만 롤백
            savepoint.rollback()
            print(f"공고 #{idx} 처리 실패: {str(e)}")
            return False
            
    except Exception as e:
        print(f"공고 #{idx} 처리 중 예상치 못한 오류: {str(e)}")
        return False


def process_json_file(db: Session, file_path: str) -> tuple[int, int]:
    """
    JSON 파일 하나를 처리
    Returns: (성공 개수, 실패 개수)
    """
    print(f"\n{'='*80}")
    print(f"파일 처리 중: {file_path}")
    print(f"{'='*80}\n")

    with open(file_path, 'r', encoding='utf-8') as f:
        jobs_data = json.load(f)

    if not isinstance(jobs_data, list):
        print(f"경고: {file_path}는 리스트 형식이 아닙니다.")
        return 0, 0

    success_count = 0
    error_count = 0

    for idx, job_data in enumerate(jobs_data, 1):
        if process_single_job(db, job_data, idx):
            success_count += 1
        else:
            error_count += 1

    # 파일 단위로 커밋
    try:
        db.commit()
        print(f"\n파일 처리 완료: 성공 {success_count}개, 실패 {error_count}개")
        print(f"데이터베이스에 커밋됨\n")
    except Exception as e:
        db.rollback()
        print(f"\n파일 커밋 실패: {str(e)}")
        return 0, success_count + error_count  # 모두 실패로 처리
    
    return success_count, error_count


def main():
    """메인 함수"""
    # data/output 디렉토리의 모든 *_jobs.json 파일 처리
    output_dir = os.path.join(os.path.dirname(__file__), "../../../data/output")
    json_files = glob.glob(os.path.join(output_dir, "*_jobs.json"))

    if not json_files:
        print(f"경고: {output_dir}에 *_jobs.json 파일이 없습니다.")
        return

    print(f"\n발견된 파일: {len(json_files)}개")
    for file in json_files:
        print(f"  - {os.path.basename(file)}")

    # 데이터베이스 세션 생성
    db = SessionLocal()

    total_success = 0
    total_error = 0

    try:
        # 각 파일 처리
        for json_file in json_files:
            success, error = process_json_file(db, json_file)
            total_success += success
            total_error += error

        print(f"\n{'='*80}")
        print("모든 파일 처리 완료!")
        print(f"전체 성공: {total_success}개")
        print(f"전체 실패: {total_error}개")
        print(f"{'='*80}\n")

    except Exception as e:
        print(f"전체 프로세스 오류: {str(e)}")
        db.rollback()
    finally:
        db.close()


if __name__ == "__main__":
    main()