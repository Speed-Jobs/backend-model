from typing import List, Optional, Union

from pydantic import BaseModel, Field


class RecruitScheduleData(BaseModel):
    """채용 일정 정보 (LLM 추출 결과를 구조화한 데이터)

    - 모든 날짜 필드: [[start_date, end_date], ...] 형태의 1차원 배열 리스트
    - 단일 날짜인 경우: [date, date] 형태로 동일한 날짜를 두 번 반복
    - application_date의 마감일은 없을 수 있음 (None 허용)
    """

    semester: Optional[str] = Field(
        None,
        description="상반기 또는 하반기"
    )
    application_date: List[List[Optional[str]]] = Field(
        default_factory=list,
        description="지원서 접수 기간 리스트 ([[시작일, 마감일], ...], 시작일은 posted_at, 마감일은 description에서 추출, 없으면 None)"
    )
    document_screening_date: List[List[str]] = Field(
        default_factory=list,
        description="서류전형 기간 리스트 ([[시작일, 마감일], ...], YYYY-MM-DD 형식)"
    )
    first_interview: List[List[str]] = Field(
        default_factory=list,
        description="1차 면접 날짜 리스트 ([[날짜, 날짜], ...], 단일 날짜는 [date, date] 형태)"
    )
    second_interview: List[List[str]] = Field(
        default_factory=list,
        description="2차 면접 날짜 리스트 ([[날짜, 날짜], ...], 단일 날짜는 [date, date] 형태)"
    )
    join_date: List[List[str]] = Field(
        default_factory=list,
        description="입사일 기간 리스트 ([[시작일, 마감일], ...], YYYY-MM-DD 형식)"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "semester": "상반기",
                "application_date": [["2025-03-01", "2025-03-10"]],
                "document_screening_date": [["2025-03-15", "2025-03-15"]],
                "first_interview": [["2025-03-20", "2025-03-20"]],
                "second_interview": [["2025-03-27", "2025-03-27"]],
                "join_date": [["2025-07-01", "2025-07-01"]],
            }
        }


