"""
Test script to verify that database data matches the source JSON files
Tests with JOIN queries to validate relationships
"""
import json
import os
from typing import Dict, List, Any
from sqlalchemy.orm import Session

from app.db.config.base import SessionLocal
from app.models.company import Company
from app.models.position import Position
from app.models.industry import Industry
from app.models.post import Post
from app.models.skill import Skill
from app.models.post_skill import PostSkill
from app.models.position_skill import PositionSkill
from app.models.industry_skill import IndustrySkill


class DataIntegrityTester:
    """Test data integrity between JSON sources and database"""

    def __init__(self):
        self.db = SessionLocal()
        self.json_data = self.load_json_files()

    def load_json_files(self) -> Dict[str, Any]:
        """Load description.json and job_description.json"""
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        description_file = os.path.join(base_dir, "data/description.json")
        job_description_file = os.path.join(base_dir, "data/job_description.json")

        with open(description_file, 'r', encoding='utf-8') as f:
            description_data = json.load(f)

        with open(job_description_file, 'r', encoding='utf-8') as f:
            job_description_data = json.load(f)

        return {
            "description": description_data,
            "job_description": job_description_data
        }

    def test_skills_from_description_json(self):
        """Test that all skills from description.json are in the database"""
        print("\n" + "="*80)
        print("TEST 1: Skills from description.json")
        print("="*80)

        # Get skills from JSON
        json_common_skills = set(self.json_data["description"].get("공통_skill_set", []))
        json_skills = set(self.json_data["description"].get("skill_set", []))
        all_json_skills = json_common_skills | json_skills

        print(f"\nTotal skills in JSON: {len(all_json_skills)}")

        # Get skills from DB (create lowercase mapping for case-insensitive comparison)
        db_skills_list = self.db.query(Skill).all()
        db_skills_lower_map = {skill.name.lower(): skill.name for skill in db_skills_list}
        db_skills = set([skill.name for skill in db_skills_list])
        print(f"Total skills in DB: {len(db_skills)}")

        # Find missing skills (case-insensitive)
        json_skills_lower_map = {skill.lower(): skill for skill in all_json_skills}
        missing_skills_lower = set(json_skills_lower_map.keys()) - set(db_skills_lower_map.keys())
        missing_skills = [json_skills_lower_map[s] for s in missing_skills_lower]

        # Find extra skills (case-insensitive)
        extra_skills_lower = set(db_skills_lower_map.keys()) - set(json_skills_lower_map.keys())
        extra_skills = [db_skills_lower_map[s] for s in extra_skills_lower]

        if missing_skills:
            print(f"\n❌ Missing {len(missing_skills)} skills in DB:")
            for skill in list(missing_skills)[:10]:
                print(f"  - {skill}")
            if len(missing_skills) > 10:
                print(f"  ... and {len(missing_skills) - 10} more")
        else:
            print("\n✓ All skills from JSON are in DB")

        if extra_skills:
            print(f"\n⚠ Found {len(extra_skills)} extra skills in DB (not in JSON):")
            for skill in list(extra_skills)[:10]:
                print(f"  - {skill}")
            if len(extra_skills) > 10:
                print(f"  ... and {len(extra_skills) - 10} more")

        # Check for case mismatches (skills exist but with different casing)
        case_mismatches = []
        for json_skill in all_json_skills:
            json_lower = json_skill.lower()
            if json_lower in db_skills_lower_map:
                db_skill = db_skills_lower_map[json_lower]
                if json_skill != db_skill:
                    case_mismatches.append((json_skill, db_skill))

        if case_mismatches:
            print(f"\n⚠ Found {len(case_mismatches)} case mismatches (skill exists but with different casing):")
            for json_skill, db_skill in case_mismatches[:10]:
                print(f"  JSON: '{json_skill}' -> DB: '{db_skill}'")
            if len(case_mismatches) > 10:
                print(f"  ... and {len(case_mismatches) - 10} more")

        # Sample 5 skills to verify
        print("\n📋 Sample verification (first 5 skills):")
        for skill_name in list(all_json_skills)[:5]:
            db_skill = self.db.query(Skill).filter(Skill.name == skill_name).first()
            status = "✓" if db_skill else "✗"
            print(f"  {status} {skill_name}")

    def test_positions_from_job_description_json(self):
        """Test that all positions from job_description.json are in the database"""
        print("\n" + "="*80)
        print("TEST 2: Positions from job_description.json")
        print("="*80)

        # Get unique positions from JSON
        json_positions = {}
        for entry in self.json_data["job_description"]:
            pos_name = entry.get("직무")
            if pos_name and pos_name not in json_positions:
                json_positions[pos_name] = {
                    "description": entry.get("직무 정의"),
                    "skillset": entry.get("공통_skill_set_description")
                }

        print(f"\nTotal unique positions in JSON: {len(json_positions)}")

        # Get positions from DB
        db_positions = {pos.name: pos for pos in self.db.query(Position).all()}
        print(f"Total positions in DB: {len(db_positions)}")

        # Verify each position
        print("\n📋 Position verification:")
        for pos_name, pos_data in json_positions.items():
            db_pos = db_positions.get(pos_name)
            if db_pos:
                print(f"\n  ✓ {pos_name}")
                print(f"    - Has description: {'✓' if db_pos.description else '✗'}")
                print(f"    - Has skillset: {'✓' if db_pos.skillset else '✗'}")

                # Show first 100 chars of description
                if db_pos.description:
                    print(f"    - Description preview: {db_pos.description[:100]}...")
            else:
                print(f"\n  ✗ {pos_name} - NOT FOUND IN DB")

    def test_industries_with_position_relationship(self):
        """Test industries and their relationships with positions (JOIN)"""
        print("\n" + "="*80)
        print("TEST 3: Industries with Position relationships (JOIN)")
        print("="*80)

        # Get industries with position JOIN
        industries_with_positions = self.db.query(
            Industry.name,
            Industry.description,
            Position.name.label("position_name")
        ).join(
            Position, Industry.position_id == Position.id
        ).all()

        print(f"\nTotal industries with position links: {len(industries_with_positions)}")

        # Group by position
        position_industries = {}
        for ind_name, ind_desc, pos_name in industries_with_positions:
            if pos_name not in position_industries:
                position_industries[pos_name] = []
            position_industries[pos_name].append(ind_name)

        # Compare with JSON
        json_position_industries = {}
        for entry in self.json_data["job_description"]:
            pos_name = entry.get("직무")
            ind_name = entry.get("industry")
            if pos_name and ind_name:
                if pos_name not in json_position_industries:
                    json_position_industries[pos_name] = []
                json_position_industries[pos_name].append(ind_name)

        print(f"\n📋 Position -> Industries mapping:")
        for pos_name in list(json_position_industries.keys())[:5]:
            json_industries = set(json_position_industries.get(pos_name, []))
            db_industries = set(position_industries.get(pos_name, []))

            print(f"\n  Position: {pos_name}")
            print(f"    JSON industries: {len(json_industries)}")
            print(f"    DB industries: {len(db_industries)}")

            missing = json_industries - db_industries
            if missing:
                print(f"    ❌ Missing in DB: {missing}")
            else:
                print(f"    ✓ All industries present")

            # Show some industries
            for ind in list(db_industries)[:3]:
                print(f"      - {ind}")

    def test_position_skills_with_join(self):
        """Test position-skill relationships with JOIN"""
        print("\n" + "="*80)
        print("TEST 4: Position-Skill relationships (JOIN)")
        print("="*80)

        # Get position-skills with JOIN
        position_skills = self.db.query(
            Position.name.label("position_name"),
            Skill.name.label("skill_name")
        ).join(
            PositionSkill, Position.id == PositionSkill.position_id
        ).join(
            Skill, PositionSkill.skill_id == Skill.id
        ).all()

        print(f"\nTotal position-skill relationships: {len(position_skills)}")

        # Group by position
        position_skill_map = {}
        for pos_name, skill_name in position_skills:
            if pos_name not in position_skill_map:
                position_skill_map[pos_name] = []
            position_skill_map[pos_name].append(skill_name)

        # Compare with JSON
        json_position_skills = {}
        for entry in self.json_data["job_description"]:
            pos_name = entry.get("직무")
            common_skills = entry.get("공통_skill_set", [])
            if pos_name and common_skills:
                if pos_name not in json_position_skills:
                    json_position_skills[pos_name] = set()
                json_position_skills[pos_name].update(common_skills)

        print(f"\n📋 Position -> Skills mapping:")
        for pos_name in list(json_position_skills.keys())[:3]:
            json_skills = json_position_skills.get(pos_name, set())
            db_skills = set(position_skill_map.get(pos_name, []))

            print(f"\n  Position: {pos_name}")
            print(f"    Expected skills from JSON: {len(json_skills)}")
            print(f"    Actual skills in DB: {len(db_skills)}")

            # Case-insensitive comparison
            json_skills_lower = {s.lower() for s in json_skills}
            db_skills_lower = {s.lower() for s in db_skills}
            missing_lower = json_skills_lower - db_skills_lower

            if missing_lower:
                missing = [s for s in json_skills if s.lower() in missing_lower]
                print(f"    ❌ Missing {len(missing)} skills:")
                for skill in list(missing)[:5]:
                    print(f"      - {skill}")
            else:
                print(f"    ✓ All skills present (case-insensitive)")

            # Show sample skills
            print(f"    Sample skills in DB:")
            for skill in list(db_skills)[:5]:
                print(f"      - {skill}")

    def test_industry_skills_with_join(self):
        """Test industry-skill relationships with JOIN"""
        print("\n" + "="*80)
        print("TEST 5: Industry-Skill relationships (JOIN)")
        print("="*80)

        # Get industry-skills with JOIN
        industry_skills = self.db.query(
            Industry.name.label("industry_name"),
            Skill.name.label("skill_name")
        ).join(
            IndustrySkill, Industry.id == IndustrySkill.industry_id
        ).join(
            Skill, IndustrySkill.skill_id == Skill.id
        ).all()

        print(f"\nTotal industry-skill relationships: {len(industry_skills)}")

        # Group by industry
        industry_skill_map = {}
        for ind_name, skill_name in industry_skills:
            if ind_name not in industry_skill_map:
                industry_skill_map[ind_name] = []
            industry_skill_map[ind_name].append(skill_name)

        # Compare with JSON
        json_industry_skills = {}
        for entry in self.json_data["job_description"]:
            ind_name = entry.get("industry")
            skills = entry.get("skill_set", [])
            if ind_name and skills:
                if ind_name not in json_industry_skills:
                    json_industry_skills[ind_name] = set()
                json_industry_skills[ind_name].update(skills)

        print(f"\n📋 Industry -> Skills mapping (sample):")
        for ind_name in list(json_industry_skills.keys())[:3]:
            json_skills = json_industry_skills.get(ind_name, set())
            db_skills = set(industry_skill_map.get(ind_name, []))

            print(f"\n  Industry: {ind_name}")
            print(f"    Expected skills from JSON: {len(json_skills)}")
            print(f"    Actual skills in DB: {len(db_skills)}")

            # Case-insensitive comparison
            json_skills_lower = {s.lower() for s in json_skills}
            db_skills_lower = {s.lower() for s in db_skills}
            missing_lower = json_skills_lower - db_skills_lower

            if missing_lower:
                missing = [s for s in json_skills if s.lower() in missing_lower]
                print(f"    ❌ Missing {len(missing)} skills:")
                for skill in list(missing)[:5]:
                    print(f"      - {skill}")
            else:
                print(f"    ✓ All skills present (case-insensitive)")

            # Show sample skills
            print(f"    Sample skills in DB:")
            for skill in list(db_skills)[:5]:
                print(f"      - {skill}")

    def test_complex_join_query(self):
        """Test complex JOIN query: Position -> Industry -> IndustrySkill -> Skill"""
        print("\n" + "="*80)
        print("TEST 6: Complex JOIN - Position -> Industry -> Skills")
        print("="*80)

        # Complex JOIN query
        results = self.db.query(
            Position.name.label("position_name"),
            Industry.name.label("industry_name"),
            Skill.name.label("skill_name")
        ).join(
            Industry, Position.id == Industry.position_id
        ).join(
            IndustrySkill, Industry.id == IndustrySkill.industry_id
        ).join(
            Skill, IndustrySkill.skill_id == Skill.id
        ).limit(20).all()

        print(f"\nShowing first 20 relationships:")
        print("-" * 80)

        for pos_name, ind_name, skill_name in results:
            print(f"{pos_name:35s} | {ind_name:35s} | {skill_name}")

    def run_all_tests(self):
        """Run all tests"""
        try:
            self.test_skills_from_description_json()
            self.test_positions_from_job_description_json()
            self.test_industries_with_position_relationship()
            self.test_position_skills_with_join()
            self.test_industry_skills_with_join()
            self.test_complex_join_query()

            print("\n" + "="*80)
            print("✓ ALL TESTS COMPLETED")
            print("="*80)
        finally:
            self.db.close()


if __name__ == "__main__":
    print("\n" + "="*80)
    print("DATA INTEGRITY TEST - JSON vs DATABASE")
    print("Testing with JOIN queries to validate relationships")
    print("="*80)

    tester = DataIntegrityTester()
    tester.run_all_tests()
