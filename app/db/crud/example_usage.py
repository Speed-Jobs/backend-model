"""
Example usage of Post CRUD operations
"""
import sys
from pathlib import Path

# Ensure project root is on PYTHONPATH when executed as a script
PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from app.db.config.base import SessionLocal
from app.db.crud.post import (
    get_posts,
    get_post_by_id,
    get_posts_by_company_id,
    get_posts_with_skills,
    count_posts,
    get_posts_by_skill_name
)


def main():
    """Demonstrate Post CRUD operations"""
    db = SessionLocal() # SQLAlchemy의 SessionLocal 객체를 사용하여 데이터베이스 세션을 생성.

    try:
        print("="*80)
        print("POST CRUD OPERATIONS - EXAMPLE USAGE")
        print("="*80)

        # 1. Get total count
        total = count_posts(db)
        print(f"\n1. Total posts in database: {total}")

        # 2. Get first 30 posts
        print("\n2. Getting first 30 posts...")
        posts = get_posts(db, skip=0, limit=30)
        print(f"   Retrieved {len(posts)} posts")

        if posts:
            print("\n   Sample post:")
            sample = posts[0]
            print(f"   - ID: {sample.id}")
            print(f"   - Title: {sample.title}")
            print(f"   - Company: {sample.company.name if sample.company else 'N/A'}")
            print(f"   - Source URL: {sample.source_url}")

        # 3. Get post by ID
        if posts:
            post_id = posts[0].id
            print(f"\n3. Getting post by ID ({post_id})...")
            post = get_post_by_id(db, post_id)
            if post:
                print(f"   ✓ Found: {post.title}")

        # 4. Get posts with skills
        print("\n4. Getting posts with their skills (first 5)...")
        posts_with_skills = get_posts_with_skills(db, skip=0, limit=5)
        for i, post in enumerate(posts_with_skills, 1):
            skills = getattr(post, 'skills', [])
            print(f"   Post {i}: {post.title[:50]}...")
            print(f"   Skills: {len(skills)} skills")
            if skills:
                print(f"   Sample skills: {', '.join([s.name for s in skills[:3]])}")

        # 5. Get posts by company
        if posts and posts[0].company_id:
            company_id = posts[0].company_id
            print(f"\n5. Getting posts by company ID ({company_id})...")
            company_posts = get_posts_by_company_id(db, company_id, limit=10)
            print(f"   Found {len(company_posts)} posts for this company")

        # 6. Get posts by skill
        print("\n6. Getting posts that require 'Python' skill...")
        python_posts = get_posts_by_skill_name(db, "Python", limit=5)
        print(f"   Found {len(python_posts)} posts requiring Python")
        for i, post in enumerate(python_posts[:3], 1):
            print(f"   - {post.title[:60]}...")

        print("\n" + "="*80)
        print("✓ CRUD operations completed successfully")
        print("="*80)

    except Exception as e:
        print(f"\n✗ Error: {str(e)}")
        import traceback
        traceback.print_exc()
    finally:
        db.close()


if __name__ == "__main__":
    main()
