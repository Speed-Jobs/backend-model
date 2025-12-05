-- 경쟁 강도 분석 - 간단 버전
-- 2025-11-01 ~ 2025-11-30, 전체 (신입+경력)
-- 9개 경쟁사 그룹: 네이버, 토스, 라인, 우아한형제들, LG CNS, 현대오토에버, 한화시스템, 카카오, 쿠팡

SELECT
    dr.date,
    COUNT(DISTINCT rs.company_id) AS overlap_count
FROM (
    -- 날짜 시퀀스 생성 (2025-11-01 ~ 2025-11-30)
    SELECT DATE_ADD('2025-11-01', INTERVAL n DAY) AS date
    FROM (
        SELECT a.N + b.N * 10 AS n
        FROM
            (SELECT 0 AS N UNION SELECT 1 UNION SELECT 2 UNION SELECT 3 UNION SELECT 4
             UNION SELECT 5 UNION SELECT 6 UNION SELECT 7 UNION SELECT 8 UNION SELECT 9) a,
            (SELECT 0 AS N UNION SELECT 1 UNION SELECT 2 UNION SELECT 3 UNION SELECT 4
             UNION SELECT 5 UNION SELECT 6 UNION SELECT 7 UNION SELECT 8 UNION SELECT 9) b
    ) numbers
    WHERE DATE_ADD('2025-11-01', INTERVAL n DAY) <= '2025-11-30'
) dr
LEFT JOIN (
    -- 경쟁사의 recruit_schedule (각 schedule별 날짜 범위 계산)
    SELECT
        rs.schedule_id,
        rs.company_id,
        c.name AS company_name,
        -- 각 schedule의 5개 단계 중 최소값 = 시작일
        LEAST(
            COALESCE(JSON_UNQUOTE(JSON_EXTRACT(rs.application_date, '$[0][0]')), '9999-12-31'),
            COALESCE(JSON_UNQUOTE(JSON_EXTRACT(rs.document_screening_date, '$[0][0]')), '9999-12-31'),
            COALESCE(JSON_UNQUOTE(JSON_EXTRACT(rs.first_interview, '$[0][0]')), '9999-12-31'),
            COALESCE(JSON_UNQUOTE(JSON_EXTRACT(rs.second_interview, '$[0][0]')), '9999-12-31'),
            COALESCE(JSON_UNQUOTE(JSON_EXTRACT(rs.join_date, '$[0][0]')), '9999-12-31')
        ) AS start_date,
        -- 각 schedule의 5개 단계 중 최대값 = 종료일
        GREATEST(
            COALESCE(JSON_UNQUOTE(JSON_EXTRACT(rs.application_date, '$[0][1]')), '0000-01-01'),
            COALESCE(JSON_UNQUOTE(JSON_EXTRACT(rs.document_screening_date, '$[0][1]')), '0000-01-01'),
            COALESCE(JSON_UNQUOTE(JSON_EXTRACT(rs.first_interview, '$[0][1]')), '0000-01-01'),
            COALESCE(JSON_UNQUOTE(JSON_EXTRACT(rs.second_interview, '$[0][1]')), '0000-01-01'),
            COALESCE(JSON_UNQUOTE(JSON_EXTRACT(rs.join_date, '$[0][1]')), '0000-01-01')
        ) AS end_date
    FROM recruit_schedule rs
    INNER JOIN company c ON rs.company_id = c.id
    INNER JOIN post p ON rs.post_id = p.id
    WHERE
        -- 9개 경쟁사 그룹 (COMPETITOR_GROUPS와 완전히 동일한 패턴)
        (c.name LIKE '토스%' OR c.name LIKE '토스뱅크%' OR c.name LIKE '토스증권%' OR c.name LIKE '비바리퍼블리카%' OR c.name LIKE 'AICC%'
         OR c.name LIKE '카카오%'
         OR c.name LIKE '한화시스템%' OR c.name LIKE '한화시스템템%' OR c.name LIKE '한화시스템/ICT%' OR c.name LIKE '한화시스템·ICT%'
         OR c.name LIKE '현대오토에버%'
         OR c.name LIKE '우아한%' OR c.name LIKE '%배달의민족%' OR c.name LIKE '%배민%'
         OR c.name LIKE '쿠팡%' OR c.name LIKE 'Coupang%'
         OR c.name LIKE 'LINE%' OR c.name LIKE '라인%'
         OR c.name LIKE 'NAVER%' OR c.name LIKE '네이버%'
         OR c.name LIKE 'LG_CNS%' OR c.name LIKE 'LG CNS%')
        -- experience가 있는 것만
        AND p.experience IS NOT NULL
) rs
    ON dr.date BETWEEN rs.start_date AND rs.end_date
    AND rs.start_date != '9999-12-31'
    AND rs.end_date != '0000-01-01'
    AND rs.end_date >= '2025-11-01'
    AND rs.start_date <= '2025-11-30'
GROUP BY dr.date
HAVING overlap_count > 0
ORDER BY dr.date;
