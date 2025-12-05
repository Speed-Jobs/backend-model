-- Schedule 수 확인 (2025-11-01 ~ 2025-11-30 기간 내 겹치는 것들)
SELECT
    COUNT(DISTINCT rs.schedule_id) AS schedule_count,
    COUNT(DISTINCT rs.company_id) AS company_count
FROM recruit_schedule rs
INNER JOIN company c ON rs.company_id = c.id
INNER JOIN post p ON rs.post_id = p.id
WHERE
    -- 9개 경쟁사 그룹
    (c.name LIKE '토스%' OR c.name LIKE '토스뱅크%' OR c.name LIKE '토스증권%' OR c.name LIKE '비바리퍼블리카%' OR c.name LIKE 'AICC%'
     OR c.name LIKE '카카오%'
     OR c.name LIKE '한화시스템%' OR c.name LIKE '한화시스템템%' OR c.name LIKE '한화시스템/ICT%' OR c.name LIKE '한화시스템·ICT%'
     OR c.name LIKE '현대오토에버%'
     OR c.name LIKE '우아한%' OR c.name LIKE '%배달의민족%' OR c.name LIKE '%배민%'
     OR c.name LIKE '쿠팡%' OR c.name LIKE 'Coupang%'
     OR c.name LIKE 'LINE%' OR c.name LIKE '라인%'
     OR c.name LIKE 'NAVER%' OR c.name LIKE '네이버%'
     OR c.name LIKE 'LG_CNS%' OR c.name LIKE 'LG CNS%')
    AND p.experience IS NOT NULL
    -- 기간 필터 (최소한 하나의 날짜라도 있어야 함)
    AND (
        LEAST(
            COALESCE(JSON_UNQUOTE(JSON_EXTRACT(rs.application_date, '$[0][0]')), '9999-12-31'),
            COALESCE(JSON_UNQUOTE(JSON_EXTRACT(rs.document_screening_date, '$[0][0]')), '9999-12-31'),
            COALESCE(JSON_UNQUOTE(JSON_EXTRACT(rs.first_interview, '$[0][0]')), '9999-12-31'),
            COALESCE(JSON_UNQUOTE(JSON_EXTRACT(rs.second_interview, '$[0][0]')), '9999-12-31'),
            COALESCE(JSON_UNQUOTE(JSON_EXTRACT(rs.join_date, '$[0][0]')), '9999-12-31')
        ) != '9999-12-31'
        AND
        GREATEST(
            COALESCE(JSON_UNQUOTE(JSON_EXTRACT(rs.application_date, '$[0][1]')), '0000-01-01'),
            COALESCE(JSON_UNQUOTE(JSON_EXTRACT(rs.document_screening_date, '$[0][1]')), '0000-01-01'),
            COALESCE(JSON_UNQUOTE(JSON_EXTRACT(rs.first_interview, '$[0][1]')), '0000-01-01'),
            COALESCE(JSON_UNQUOTE(JSON_EXTRACT(rs.second_interview, '$[0][1]')), '0000-01-01'),
            COALESCE(JSON_UNQUOTE(JSON_EXTRACT(rs.join_date, '$[0][1]')), '0000-01-01')
        ) != '0000-01-01'
        AND
        GREATEST(
            COALESCE(JSON_UNQUOTE(JSON_EXTRACT(rs.application_date, '$[0][1]')), '0000-01-01'),
            COALESCE(JSON_UNQUOTE(JSON_EXTRACT(rs.document_screening_date, '$[0][1]')), '0000-01-01'),
            COALESCE(JSON_UNQUOTE(JSON_EXTRACT(rs.first_interview, '$[0][1]')), '0000-01-01'),
            COALESCE(JSON_UNQUOTE(JSON_EXTRACT(rs.second_interview, '$[0][1]')), '0000-01-01'),
            COALESCE(JSON_UNQUOTE(JSON_EXTRACT(rs.join_date, '$[0][1]')), '0000-01-01')
        ) >= '2025-11-01'
        AND
        LEAST(
            COALESCE(JSON_UNQUOTE(JSON_EXTRACT(rs.application_date, '$[0][0]')), '9999-12-31'),
            COALESCE(JSON_UNQUOTE(JSON_EXTRACT(rs.document_screening_date, '$[0][0]')), '9999-12-31'),
            COALESCE(JSON_UNQUOTE(JSON_EXTRACT(rs.first_interview, '$[0][0]')), '9999-12-31'),
            COALESCE(JSON_UNQUOTE(JSON_EXTRACT(rs.second_interview, '$[0][0]')), '9999-12-31'),
            COALESCE(JSON_UNQUOTE(JSON_EXTRACT(rs.join_date, '$[0][0]')), '9999-12-31')
        ) <= '2025-11-30'
    );
