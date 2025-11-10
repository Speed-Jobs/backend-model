# test_db_connection.py
from app.db.db_connection import DatabaseManager

def test_connection():
    """기본 연결 테스트"""
    db = DatabaseManager()
    
    try:
        # 연결 테스트
        conn = db.get_connection()
        print("✓ 데이터베이스 연결 성공!")
        
        # 간단한 쿼리 테스트
        with db.get_cursor() as cursor:
            cursor.execute("SELECT VERSION()")
            version = cursor.fetchone()
            print(f"✓ MySQL 버전: {version['VERSION()']}")
            
            # 현재 데이터베이스 확인
            cursor.execute("SELECT DATABASE()")
            current_db = cursor.fetchone()
            print(f"✓ 현재 데이터베이스: {current_db['DATABASE()']}")
            
            # 테이블 목록 확인
            cursor.execute("SHOW TABLES")
            tables = cursor.fetchall()
            print(f"✓ 테이블 개수: {len(tables)}")
            if tables:
                print("  테이블 목록:")
                for table in tables:
                    print(f"    - {list(table.values())[0]}")
        
        print("\n모든 테스트 통과!")
        
    except Exception as e:
        print(f"✗ 테스트 실패: {e}")
    finally:
        db.close()

if __name__ == "__main__":
    test_connection()