"""Query Logger Utility

LLM이 생성한 SQL 쿼리를 로깅하고 모니터링하기 위한 유틸리티
"""

import os
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional


class QueryLogger:
    """SQL 쿼리 생성 및 실행 로깅"""
    
    def __init__(self, log_dir: str = "logs/sql_queries"):
        """
        Args:
            log_dir: 로그 파일을 저장할 디렉토리 경로
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
    def _get_log_filename(self) -> Path:
        """날짜별 로그 파일명 생성"""
        today = datetime.now().strftime("%Y-%m-%d")
        return self.log_dir / f"sql_queries_{today}.jsonl"
    
    def log_query_generation(
        self,
        question: str,
        route_decision: str,
        extracted_entities: Dict[str, Any],
        generated_sql: str,
        query_type: str,
        llm_response: Optional[str] = None,
        execution_time_ms: Optional[float] = None,
        success: bool = True,
        error: Optional[str] = None,
        result_count: Optional[int] = None
    ):
        """
        SQL 쿼리 생성 및 실행 정보를 로깅
        
        Args:
            question: 사용자 질문
            route_decision: 라우팅 결정 (statistics_with_stats 등)
            extracted_entities: 추출된 엔티티
            generated_sql: LLM이 생성한 SQL 쿼리
            query_type: 쿼리 유형
            llm_response: LLM의 원본 응답 (선택)
            execution_time_ms: 쿼리 실행 시간 (밀리초)
            success: 쿼리 실행 성공 여부
            error: 에러 메시지 (실패 시)
            result_count: 결과 행 수
        """
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "question": question,
            "route_decision": route_decision,
            "extracted_entities": extracted_entities,
            "query_info": {
                "query_type": query_type,
                "generated_sql": generated_sql,
                "llm_response": llm_response
            },
            "execution": {
                "success": success,
                "execution_time_ms": execution_time_ms,
                "result_count": result_count,
                "error": error
            }
        }
        
        # JSONL 형식으로 추가 (한 줄에 하나의 JSON)
        log_file = self._get_log_filename()
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(log_entry, ensure_ascii=False) + '\n')
        
        # 콘솔에도 출력
        print(f"\n{'='*80}")
        print(f"[QueryLogger] SQL Query Generated at {log_entry['timestamp']}")
        print(f"{'='*80}")
        print(f"📝 Question: {question}")
        print(f"🎯 Route: {route_decision}")
        print(f"🏷️  Entities: {json.dumps(extracted_entities, ensure_ascii=False)}")
        print(f"📊 Query Type: {query_type}")
        print(f"\n💾 Generated SQL:")
        print("-" * 80)
        print(generated_sql)
        print("-" * 80)
        
        if execution_time_ms:
            print(f"⏱️  Execution Time: {execution_time_ms:.2f}ms")
        if result_count is not None:
            print(f"📈 Result Count: {result_count} rows")
        if error:
            print(f"❌ Error: {error}")
        else:
            print(f"✅ Status: Success")
        print(f"{'='*80}\n")
    
    def log_routing_decision(
        self,
        question: str,
        route_decision: str,
        extracted_entities: Dict[str, Any],
        needs_stats: bool,
        top_k: int,
        reason: str,
        llm_response: Optional[str] = None
    ):
        """
        라우팅 결정 정보를 로깅
        
        Args:
            question: 사용자 질문
            route_decision: 라우팅 결정
            extracted_entities: 추출된 엔티티
            needs_stats: 통계 분석 필요 여부
            top_k: 검색 결과 수
            reason: 라우팅 결정 이유
            llm_response: LLM의 원본 응답 (선택)
        """
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "type": "routing_decision",
            "question": question,
            "route_decision": route_decision,
            "extracted_entities": extracted_entities,
            "params": {
                "needs_stats": needs_stats,
                "top_k": top_k,
                "reason": reason
            },
            "llm_response": llm_response
        }
        
        # 날짜별 라우팅 로그 파일
        today = datetime.now().strftime("%Y-%m-%d")
        log_file = self.log_dir / f"routing_decisions_{today}.jsonl"
        
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(log_entry, ensure_ascii=False) + '\n')
        
        # 콘솔 출력
        print(f"\n{'='*80}")
        print(f"[QueryLogger] Routing Decision at {log_entry['timestamp']}")
        print(f"{'='*80}")
        print(f"📝 Question: {question}")
        print(f"🎯 Route: {route_decision}")
        print(f"🏷️  Entities: {json.dumps(extracted_entities, ensure_ascii=False)}")
        print(f"📊 Needs Stats: {needs_stats}")
        print(f"🔢 Top K: {top_k}")
        print(f"💭 Reason: {reason}")
        print(f"{'='*80}\n")
    
    def read_logs(self, date: Optional[str] = None, log_type: str = "sql_queries") -> list:
        """
        로그 파일 읽기
        
        Args:
            date: 날짜 (YYYY-MM-DD 형식, None이면 오늘)
            log_type: 로그 타입 ('sql_queries' 또는 'routing_decisions')
            
        Returns:
            로그 엔트리 리스트
        """
        if date is None:
            date = datetime.now().strftime("%Y-%m-%d")
        
        log_file = self.log_dir / f"{log_type}_{date}.jsonl"
        
        if not log_file.exists():
            return []
        
        logs = []
        with open(log_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    logs.append(json.loads(line))
        
        return logs
    
    def get_statistics(self, date: Optional[str] = None) -> Dict[str, Any]:
        """
        특정 날짜의 쿼리 통계
        
        Args:
            date: 날짜 (YYYY-MM-DD 형식, None이면 오늘)
            
        Returns:
            통계 정보
        """
        logs = self.read_logs(date, "sql_queries")
        
        if not logs:
            return {
                "total_queries": 0,
                "successful_queries": 0,
                "failed_queries": 0,
                "avg_execution_time_ms": 0,
                "query_types": {},
                "route_decisions": {}
            }
        
        total = len(logs)
        successful = sum(1 for log in logs if log["execution"]["success"])
        failed = total - successful
        
        # 평균 실행 시간
        exec_times = [
            log["execution"]["execution_time_ms"] 
            for log in logs 
            if log["execution"]["execution_time_ms"] is not None
        ]
        avg_exec_time = sum(exec_times) / len(exec_times) if exec_times else 0
        
        # 쿼리 타입별 분포
        query_types = {}
        for log in logs:
            qtype = log["query_info"]["query_type"]
            query_types[qtype] = query_types.get(qtype, 0) + 1
        
        # 라우팅 결정별 분포
        route_decisions = {}
        for log in logs:
            route = log["route_decision"]
            route_decisions[route] = route_decisions.get(route, 0) + 1
        
        return {
            "total_queries": total,
            "successful_queries": successful,
            "failed_queries": failed,
            "success_rate": f"{(successful/total*100):.1f}%",
            "avg_execution_time_ms": f"{avg_exec_time:.2f}",
            "query_types": query_types,
            "route_decisions": route_decisions
        }


# 전역 인스턴스
_query_logger = None

def get_query_logger() -> QueryLogger:
    """전역 QueryLogger 인스턴스 반환"""
    global _query_logger
    if _query_logger is None:
        _query_logger = QueryLogger()
    return _query_logger

