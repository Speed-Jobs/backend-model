"""
Script to insert job description data from JSON files into the database
- description.json: Complete skill list
- job_description.json: Position, industry, and skill relationship data
"""
import json
import os
from typing import List, Dict, Any, Optional
from sqlalchemy.orm import Session
from sqlalchemy.exc import IntegrityError

from app.db.config.base import SessionLocal, engine

from app.models.company import Company
from app.models.position import Position
from app.models.industry import Industry
from app.models.post import Post
from app.models.skill import Skill
from app.models.post_skill import PostSkill
from app.models.position_skill import PositionSkill
from app.models.industry_skill import IndustrySkill  


def get_or_create_skill(db: Session, skill_name: str) -> Skill:
    """Get or create a skill"""
    skill = db.query(Skill).filter(Skill.name == skill_name).first()
    if not skill:
        skill = Skill(name=skill_name)
        db.add(skill)
        db.flush()
        print(f"  Created new skill: {skill_name} (ID: {skill.id})")
    return skill


def get_or_create_position(
    db: Session,
    name: str,
    description: Optional[str] = None,
    skillset: Optional[str] = None
) -> Position:
    """Get or create a position"""
    position = db.query(Position).filter(Position.name == name).first()
    if not position:
        position = Position(
            name=name,
            description=description,
            skillset=skillset
        )
        db.add(position)
        db.flush()
        print(f"  Created new position: {name} (ID: {position.id})")
    else:
        # Update description and skillset if they don't exist (first occurrence takes priority)
        if description and not position.description:
            position.description = description
        if skillset and not position.skillset:
            position.skillset = skillset
        db.flush()
    return position


def get_or_create_industry(
    db: Session,
    name: str,
    description: Optional[str] = None,
    position_id: Optional[int] = None
) -> Industry:
    """Get or create an industry"""
    industry = db.query(Industry).filter(
        Industry.name == name,
        Industry.position_id == position_id
    ).first()
    if not industry:
        industry = Industry(
            name=name,
            description=description,
            position_id=position_id
        )
        db.add(industry)
        db.flush()
        print(f"  Created new industry: {name} (ID: {industry.id})")
    return industry


def create_position_skills(db: Session, position: Position, skill_names: List[str]) -> int:
    """Create relationships between position and skills"""
    added_count = 0
    for skill_name in skill_names:
        if not skill_name or skill_name.strip() == "":
            continue

        skill = get_or_create_skill(db, skill_name.strip())

        # Check if relationship already exists
        existing_relation = db.query(PositionSkill).filter(
            PositionSkill.position_id == position.id,
            PositionSkill.skill_id == skill.id
        ).first()

        if not existing_relation:
            position_skill = PositionSkill(
                position_id=position.id,
                skill_id=skill.id
            )
            db.add(position_skill)
            print(f"    - Linked skill: {skill_name} -> Position #{position.id}")
            added_count += 1

    return added_count


def create_industry_skills(db: Session, industry: Industry, skill_names: List[str]) -> int:
    """Create relationships between industry and skills"""
    added_count = 0
    for skill_name in skill_names:
        if not skill_name or skill_name.strip() == "":
            continue

        skill = get_or_create_skill(db, skill_name.strip())

        # Check if relationship already exists
        existing_relation = db.query(IndustrySkill).filter(
            IndustrySkill.industry_id == industry.id,
            IndustrySkill.skill_id == skill.id
        ).first()

        if not existing_relation:
            industry_skill = IndustrySkill(
                industry_id=industry.id,
                skill_id=skill.id
            )
            db.add(industry_skill)
            print(f"    - Linked skill: {skill_name} -> Industry #{industry.id}")
            added_count += 1

    return added_count


def process_description_json(db: Session, file_path: str) -> tuple[int, int]:
    """
    Process description.json (insert all skills)
    Returns: (success_count, error_count)
    """
    print(f"\n{'='*80}")
    print(f"Processing description.json: {file_path}")
    print(f"{'='*80}\n")

    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    success_count = 0
    error_count = 0

    try:
        # Process common_skill_set
        if "공통_skill_set" in data:
            print(f"Processing common_skill_set... (total {len(data['공통_skill_set'])} skills)")
            for skill_name in data["공통_skill_set"]:
                try:
                    get_or_create_skill(db, skill_name.strip())
                    success_count += 1
                except Exception as e:
                    print(f"  Failed to insert skill: {skill_name} - {str(e)}")
                    error_count += 1

        # Process skill_set
        if "skill_set" in data:
            print(f"\nProcessing skill_set... (total {len(data['skill_set'])} skills)")
            for skill_name in data["skill_set"]:
                try:
                    get_or_create_skill(db, skill_name.strip())
                    success_count += 1
                except Exception as e:
                    print(f"  Failed to insert skill: {skill_name} - {str(e)}")
                    error_count += 1

        db.commit()
        print(f"\ndescription.json completed: {success_count} succeeded, {error_count} failed")

    except Exception as e:
        db.rollback()
        print(f"\nFailed to process description.json: {str(e)}")
        return 0, success_count + error_count

    return success_count, error_count


def process_single_job_description(db: Session, job_data: Dict[str, Any], idx: int) -> bool:
    """
    Process a single job description
    Returns: success status
    """
    try:
        # Create savepoint (nested transaction)
        savepoint = db.begin_nested()

        try:
            position_name = job_data.get("직무", "")
            position_description = job_data.get("직무 정의", "")
            position_skillset = job_data.get("공통_skill_set_description", "")

            industry_name = job_data.get("industry", "")
            industry_description = job_data.get("skill_set_description", "")

            print(f"\n[{idx}] Processing: position={position_name}, industry={industry_name}")

            # 1. Create/get Position
            position = get_or_create_position(
                db,
                position_name,
                description=position_description,
                skillset=position_skillset
            )

            # 2. Create Industry
            industry = get_or_create_industry(
                db,
                industry_name,
                description=industry_description,
                position_id=position.id
            )

            # 3. Link Position-Skill (common_skill_set)
            common_skills = job_data.get("공통_skill_set", [])
            if common_skills:
                added = create_position_skills(db, position, common_skills)
                print(f"  Position-Skill links: {added} added")

            # 4. Link Industry-Skill (skill_set)
            industry_skills = job_data.get("skill_set", [])
            if industry_skills:
                added = create_industry_skills(db, industry, industry_skills)
                print(f"  Industry-Skill links: {added} added")

            # Commit savepoint
            savepoint.commit()
            print(f"  ✓ Completed")
            return True

        except Exception as e:
            # Rollback this job only
            savepoint.rollback()
            print(f"  ✗ Failed: {str(e)}")
            return False

    except Exception as e:
        print(f"  ✗ Unexpected error: {str(e)}")
        return False


def process_job_description_json(db: Session, file_path: str) -> tuple[int, int]:
    """
    Process job_description.json
    Returns: (success_count, error_count)
    """
    print(f"\n{'='*80}")
    print(f"Processing job_description.json: {file_path}")
    print(f"{'='*80}\n")

    with open(file_path, 'r', encoding='utf-8') as f:
        jobs_data = json.load(f)

    if not isinstance(jobs_data, list):
        print(f"Warning: {file_path} is not a list format.")
        return 0, 0

    success_count = 0
    error_count = 0

    for idx, job_data in enumerate(jobs_data, 1):
        if process_single_job_description(db, job_data, idx):
            success_count += 1
        else:
            error_count += 1

    # Commit all
    try:
        db.commit()
        print(f"\njob_description.json completed: {success_count} succeeded, {error_count} failed")
        print(f"Committed to database\n")
    except Exception as e:
        db.rollback()
        print(f"\nFailed to commit: {str(e)}")
        return 0, success_count + error_count

    return success_count, error_count


def main():
    """Main function"""
    # File paths
    base_dir = os.path.dirname(__file__)
    description_file = os.path.join(base_dir, "../../../data/description.json")
    job_description_file = os.path.join(base_dir, "../../../data/job_description.json")

    # Check file existence
    if not os.path.exists(description_file):
        print(f"✗ File not found: {description_file}")
        return

    if not os.path.exists(job_description_file):
        print(f"✗ File not found: {job_description_file}")
        return

    print(f"\n{'='*80}")
    print("Job Description Data Insertion Started")
    print(f"{'='*80}\n")

    # Create database session
    db = SessionLocal()

    try:
        # Step 1: Process description.json (insert all skills)
        desc_success, desc_error = process_description_json(db, description_file)

        # Step 2: Process job_description.json (Position, Industry, relationships)
        job_success, job_error = process_job_description_json(db, job_description_file)

        # Final results
        print(f"\n{'='*80}")
        print("All files processed!")
        print(f"{'='*80}")
        print(f"\n[description.json]")
        print(f"  - Success: {desc_success}")
        print(f"  - Failed: {desc_error}")
        print(f"\n[job_description.json]")
        print(f"  - Success: {job_success}")
        print(f"  - Failed: {job_error}")
        print(f"\n{'='*80}\n")

    except Exception as e:
        print(f"\nOverall process error: {str(e)}")
        db.rollback()
    finally:
        db.close()


if __name__ == "__main__":
    main()
