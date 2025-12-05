"""
CSV 데이터 Import 테스트
"""
import pandas as pd
import numpy as np
from sqlalchemy.orm import Session
from typing import Dict, Any
from datetime import datetime
import os

from app.db.config.base import SessionLocal 

from app.models.company import Company
from app.models.position import Position
from app.models.industry import Industry
from app.models.post import Post
from app.models.skill import Skill
from app.models.post_skill import PostSkill
from app.models.position_skill import PositionSkill
from app.models.industry_skill import IndustrySkill

def import_posts_from_csv(
    db: Session,
    csv_path: str
) -> Dict[str, Any]:
    """
    CSV 파일에서 Post 데이터를 읽어 DB에 추가
    """
    try:
        # 파일 존재 확인
        if not os.path.exists(csv_path):
            return {
                'success': False,
                'error': f'파일을 찾을 수 없습니다: {csv_path}'
            }
        
        # CSV 읽기
        df = pd.read_csv(csv_path)
        
        print(f"📊 CSV 파일 로드 완료: {len(df)} rows")
        print(f"📋 컬럼: {list(df.columns)}")
        
        # 날짜 컬럼 변환
        date_columns = ['posted_at', 'close_at', 'crawled_at', 'created_at', 'modified_at']
        for col in date_columns:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors='coerce')
        
        # updated_at이 없으면 현재 시간으로
        if 'updated_at' not in df.columns:
            df['updated_at'] = datetime.now()
        else:
            df['updated_at'] = pd.to_datetime(df['updated_at'], errors='coerce')
            df['updated_at'] = df['updated_at'].fillna(datetime.now())
        
        # NaN, NaT, 빈 문자열을 None으로 변환 (중요!)
        df = df.replace({np.nan: None, pd.NaT: None, '': None})
        
        # dict 리스트로 변환
        records = df.to_dict('records')
        
        print(f"🔄 데이터 변환 완료: {len(records)} records")
        
        # 첫 번째 레코드 샘플 출력 (디버깅용)
        if records:
            print(f"\n📝 첫 번째 레코드 샘플:")
            for key, value in list(records[0].items())[:8]:
                print(f"  {key}: {value} (type: {type(value).__name__})")
        
        # bulk insert
        db.bulk_insert_mappings(Post, records)
        db.commit()
        
        print(f"\n✅ DB 삽입 완료!")
        
        return {
            'success': True,
            'added': len(records),
            'message': f'{len(records)}개의 레코드가 추가되었습니다.'
        }
        
    except Exception as e:
        db.rollback()
        print(f"\n❌ 에러 발생: {str(e)}")
        import traceback
        traceback.print_exc()
        return {
            'success': False,
            'error': str(e)
        }


if __name__ == "__main__":
    # DB 세션 생성 (generator가 아닌 실제 세션)
    db = SessionLocal()
    
    try:
        # CSV 파일 경로 (실제 파일 경로로 수정)
        csv_path = './posts_2024_2025.csv'  # 또는 절대 경로 사용
        
        # 파일이 현재 디렉토리에 있는지 확인
        print(f"📂 현재 디렉토리: {os.getcwd()}")
        print(f"📄 CSV 파일 경로: {csv_path}\n")
        
        result = import_posts_from_csv(db=db, csv_path=csv_path)
        
        if result['success']:
            print(f"✅ 성공: {result['message']}")
        else:
            print(f"❌ 실패: {result['error']}")
            
    finally:
        db.close()