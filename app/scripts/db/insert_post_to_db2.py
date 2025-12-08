"""
JSON 파일에서 채용 공고 데이터를 읽어 데이터베이스에 삽입하는 스크립트
models 정의에 정확히 맞춰 작성됨
"""
import json
import os
import glob
from datetime import datetime
from typing import List, Dict, Any
from sqlalchemy.orm import Session
from sqlalchemy.exc import IntegrityError

from app.db.config.base import SessionLocal
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


def get_or_create_company(db: Session, company_name: str, company_location: str = None) -> Company:
    """회사를 조회하거나 새로 생성"""
    company = db.query(Company).filter(Company.name == company_name).first()
    if not company:
        company = Company(
            name=company_name,
            location=company_location
        )
        db.add(company)
        db.flush()
        print(f"✓ 새 회사 생성: {company_name} (ID: {company.id})")
    return company


def get_or_create_skill(db: Session, skill_name: str) -> Skill:
    """스킬을 조회하거나 새로 생성"""
    skill = db.query(Skill).filter(Skill.name == skill_name).first()
    if not skill:
        skill = Skill(name=skill_name)
        db.add(skill)
        db.flush()
        print(f"  ✓ 새 스킬: {skill_name}")
    return skill


def create_post(db: Session, job_data: Dict[str, Any], company: Company) -> tuple[Post, bool]:
    """
    채용 공고 생성
    Returns: (Post 객체, 새로 생성되었는지 여부)
    """
    source_url = job_data.get("url", "")

    # source_url로 중복 체크
    existing_post = db.query(Post).filter(Post.source_url == source_url).first()
    if existing_post:
        print(f"  → 기존 공고: {job_data.get('title')[:50]}")
        return existing_post, False

    # meta_data 구성
    meta_data = {}
    if "meta_data" in job_data:
        meta_data = job_data["meta_data"]
    if "tech_stack" in job_data:
        meta_data["tech_stack"] = job_data["tech_stack"]

    # Industry ID 조회
    industry_id = None
    sim_position = job_data.get("sim_position")
    sim_industry = job_data.get("sim_industry")

    if sim_position and sim_industry:
        position = db.query(Position).filter(Position.name == sim_position).first()
        if position:
            industry = db.query(Industry).filter(
                Industry.name == sim_industry,
                Industry.position_id == position.id
            ).first()
            if industry:
                industry_id = industry.id

    # Post 객체 생성 (models 정의에 맞춤)
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
        source_url=source_url,
        url_hash=None,  # 일단 null
        screenshot_url=job_data.get("screenshots", {}).get("combined"),
        company_id=company.id,
        industry_id=industry_id
    )

    db.add(post)
    db.flush()
    print(f"  ✓ 새 공고: {post.title[:50]} (ID: {post.id})")
    return post, True


def create_post_skills(db: Session, post: Post, skill_names: List[str]) -> int:
    """PostSkill 관계 생성"""
    added = 0
    for skill_name in skill_names:
        if not skill_name or not skill_name.strip():
            continue

        skill = get_or_create_skill(db, skill_name.strip())

        existing = db.query(PostSkill).filter(
            PostSkill.post_id == post.id,
            PostSkill.skill_id == skill.id
        ).first()

        if not existing:
            db.add(PostSkill(post_id=post.id, skill_id=skill.id))
            added += 1

    return added


def create_industry_skills(db: Session, industry_id: int, skill_names: List[str]) -> int:
    """IndustrySkill 관계 생성"""
    if not industry_id:
        return 0

    added = 0
    for skill_name in skill_names:
        if not skill_name or not skill_name.strip():
            continue

        skill = get_or_create_skill(db, skill_name.strip())

        existing = db.query(IndustrySkill).filter(
            IndustrySkill.industry_id == industry_id,
            IndustrySkill.skill_id == skill.id
        ).first()

        if not existing:
            db.add(IndustrySkill(industry_id=industry_id, skill_id=skill.id))
            added += 1

    return added


def create_position_skills(db: Session, position_id: int, skill_names: List[str]) -> int:
    """PositionSkill 관계 생성"""
    if not position_id:
        return 0

    added = 0
    for skill_name in skill_names:
        if not skill_name or not skill_name.strip():
            continue

        skill = get_or_create_skill(db, skill_name.strip())

        existing = db.query(PositionSkill).filter(
            PositionSkill.position_id == position_id,
            PositionSkill.skill_id == skill.id
        ).first()

        if not existing:
            db.add(PositionSkill(position_id=position_id, skill_id=skill.id))
            added += 1

    return added


def process_single_job(db: Session, job_data: Dict[str, Any], idx: int) -> bool:
    """단일 job 처리"""
    try:
        savepoint = db.begin_nested()

        try:
            # 1. 회사 조회/생성
            company_name = job_data.get("company", "Unknown Company")
            company_location = job_data.get("location")
            company = get_or_create_company(db, company_name, company_location)

            # 2. 공고 생성
            post, is_new = create_post(db, job_data, company)

            # 3. 스킬 추출
            skill_names = []
            if "skill_set_info" in job_data and "skill_set" in job_data["skill_set_info"]:
                skill_names.extend(job_data["skill_set_info"]["skill_set"])
            if "meta_data" in job_data and "tech_stack" in job_data["meta_data"]:
                skill_names.extend(job_data["meta_data"]["tech_stack"])
            skill_names = list(set(skill_names))  # 중복 제거

            # 4. 스킬 관계 생성
            if skill_names:
                # PostSkill
                post_skill_added = create_post_skills(db, post, skill_names)
                if post_skill_added > 0:
                    print(f"    → PostSkill: {post_skill_added}개 연결")

                # IndustrySkill
                if post.industry_id:
                    industry_skill_added = create_industry_skills(db, post.industry_id, skill_names)
                    if industry_skill_added > 0:
                        print(f"    → IndustrySkill: {industry_skill_added}개 연결")

                    # PositionSkill
                    industry = db.query(Industry).filter(Industry.id == post.industry_id).first()
                    if industry and industry.position_id:
                        position_skill_added = create_position_skills(db, industry.position_id, skill_names)
                        if position_skill_added > 0:
                            print(f"    → PositionSkill: {position_skill_added}개 연결")

            savepoint.commit()
            return True

        except Exception as e:
            savepoint.rollback()
            print(f"  ✗ 공고 #{idx} 실패: {str(e)}")
            return False

    except Exception as e:
        print(f"  ✗ 공고 #{idx} 오류: {str(e)}")
        return False


def process_json_file(db: Session, file_path: str) -> tuple[int, int]:
    """JSON 파일 처리"""
    print(f"\n{'='*80}")
    print(f"파일: {os.path.basename(file_path)}")
    print(f"{'='*80}")

    with open(file_path, 'r', encoding='utf-8') as f:
        jobs_data = json.load(f)

    if not isinstance(jobs_data, list):
        print(f"⚠ 경고: 리스트 형식이 아닙니다")
        return 0, 0

    success = 0
    error = 0

    for idx, job_data in enumerate(jobs_data, 1):
        print(f"\n[{idx}/{len(jobs_data)}] {job_data.get('company', 'Unknown')}")
        if process_single_job(db, job_data, idx):
            success += 1
        else:
            error += 1

    # 파일 단위 커밋
    try:
        db.commit()
        print(f"\n✓ 커밋 완료: 성공 {success}개, 실패 {error}개")
    except Exception as e:
        db.rollback()
        print(f"\n✗ 커밋 실패: {str(e)}")
        return 0, success + error

    return success, error


def main():
    """메인 함수"""
    output_dir = os.path.join(os.path.dirname(__file__), "../../../data/output")
    json_files = glob.glob(os.path.join(output_dir, "*_jobs.json"))

    if not json_files:
        print(f"⚠ {output_dir}에 *_jobs.json 파일이 없습니다")
        return

    print(f"\n발견된 파일: {len(json_files)}개")
    for f in json_files:
        print(f"  - {os.path.basename(f)}")

    db = SessionLocal()
    total_success = 0
    total_error = 0

    try:
        for json_file in json_files:
            success, error = process_json_file(db, json_file)
            total_success += success
            total_error += error

        print(f"\n{'='*80}")
        print(f"전체 완료!")
        print(f"성공: {total_success}개 | 실패: {total_error}개")
        print(f"{'='*80}\n")

    except Exception as e:
        print(f"\n✗ 전체 오류: {str(e)}")
        db.rollback()
    finally:
        db.close()


if __name__ == "__main__":
    main()
