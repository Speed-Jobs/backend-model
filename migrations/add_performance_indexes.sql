-- 성능 최적화를 위한 인덱스 추가
-- 실행일: 2025-12-16

-- 1. position.category에 인덱스 추가 (직군 필터링 최적화)
CREATE INDEX IF NOT EXISTS idx_position_category ON position(category);

-- 2. post 테이블 복합 인덱스 추가 (통계 조회 쿼리 최적화)
-- 날짜, 회사, 산업을 함께 필터링하는 쿼리 최적화
CREATE INDEX IF NOT EXISTS idx_post_dates_company_industry 
ON post(posted_at, crawled_at, company_id, industry_id);

-- 산업별 날짜 필터링 최적화
CREATE INDEX IF NOT EXISTS idx_post_industry_dates 
ON post(industry_id, posted_at, crawled_at);

-- 인덱스 생성 확인
SELECT 
    tablename,
    indexname,
    indexdef
FROM pg_indexes
WHERE tablename IN ('post', 'position')
ORDER BY tablename, indexname;
