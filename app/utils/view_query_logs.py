"""Query Log Viewer

SQL 쿼리 로그를 조회하고 분석하는 유틸리티
"""

import argparse
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional
from app.utils.query_logger import QueryLogger


def view_logs(date: Optional[str] = None, log_type: str = "sql_queries", limit: Optional[int] = None):
    """로그 조회"""
    logger = QueryLogger()
    logs = logger.read_logs(date, log_type)
    
    if not logs:
        print(f"\n⚠️  {date or '오늘'} 날짜의 {log_type} 로그가 없습니다.\n")
        return
    
    print(f"\n{'='*100}")
    print(f"📊 {log_type.upper()} - {date or datetime.now().strftime('%Y-%m-%d')}")
    print(f"{'='*100}\n")
    
    for i, log in enumerate(logs[:limit] if limit else logs, 1):
        print(f"\n[{i}] {log['timestamp']}")
        print("-" * 100)
        
        if log_type == "sql_queries":
            print(f"📝 Question: {log['question']}")
            print(f"🎯 Route: {log['route_decision']}")
            print(f"🏷️  Entities: {json.dumps(log['extracted_entities'], ensure_ascii=False)}")
            print(f"📊 Query Type: {log['query_info']['query_type']}")
            print(f"\n💾 SQL:")
            print(log['query_info']['generated_sql'])
            
            exec_info = log['execution']
            status = "✅ Success" if exec_info['success'] else "❌ Failed"
            print(f"\n{status}")
            
            if exec_info.get('execution_time_ms'):
                print(f"⏱️  Execution Time: {exec_info['execution_time_ms']:.2f}ms")
            if exec_info.get('result_count') is not None:
                print(f"📈 Result Count: {exec_info['result_count']} rows")
            if exec_info.get('error'):
                print(f"❌ Error: {exec_info['error']}")
                
        elif log_type == "routing_decisions":
            print(f"📝 Question: {log['question']}")
            print(f"🎯 Route: {log['route_decision']}")
            print(f"🏷️  Entities: {json.dumps(log['extracted_entities'], ensure_ascii=False)}")
            params = log['params']
            print(f"📊 Needs Stats: {params['needs_stats']}")
            print(f"🔢 Top K: {params['top_k']}")
            print(f"💭 Reason: {params['reason']}")
    
    print(f"\n{'='*100}")
    print(f"Total: {len(logs)} entries")
    if limit and len(logs) > limit:
        print(f"(Showing first {limit} of {len(logs)})")
    print(f"{'='*100}\n")


def show_statistics(date: Optional[str] = None):
    """통계 정보 표시"""
    logger = QueryLogger()
    stats = logger.get_statistics(date)
    
    print(f"\n{'='*100}")
    print(f"📊 SQL QUERY STATISTICS - {date or datetime.now().strftime('%Y-%m-%d')}")
    print(f"{'='*100}\n")
    
    if stats['total_queries'] == 0:
        print("⚠️  No queries found for this date.\n")
        return
    
    print(f"📈 Overall Statistics:")
    print(f"  • Total Queries: {stats['total_queries']}")
    print(f"  • Successful: {stats['successful_queries']}")
    print(f"  • Failed: {stats['failed_queries']}")
    print(f"  • Success Rate: {stats['success_rate']}")
    print(f"  • Avg Execution Time: {stats['avg_execution_time_ms']}ms")
    
    print(f"\n📊 Query Types Distribution:")
    for qtype, count in stats['query_types'].items():
        percentage = (count / stats['total_queries']) * 100
        print(f"  • {qtype}: {count} ({percentage:.1f}%)")
    
    print(f"\n🎯 Route Decisions Distribution:")
    for route, count in stats['route_decisions'].items():
        percentage = (count / stats['total_queries']) * 100
        print(f"  • {route}: {count} ({percentage:.1f}%)")
    
    print(f"\n{'='*100}\n")


def search_logs(keyword: str, date: Optional[str] = None):
    """키워드로 로그 검색"""
    logger = QueryLogger()
    logs = logger.read_logs(date, "sql_queries")
    
    matching_logs = [
        log for log in logs
        if keyword.lower() in log['question'].lower() or
           keyword.lower() in log['query_info']['generated_sql'].lower()
    ]
    
    if not matching_logs:
        print(f"\n⚠️  '{keyword}' 키워드를 포함하는 로그가 없습니다.\n")
        return
    
    print(f"\n{'='*100}")
    print(f"🔍 Search Results for: '{keyword}' ({len(matching_logs)} found)")
    print(f"{'='*100}\n")
    
    for i, log in enumerate(matching_logs, 1):
        print(f"\n[{i}] {log['timestamp']}")
        print(f"📝 Question: {log['question']}")
        print(f"💾 SQL: {log['query_info']['generated_sql'][:200]}...")
        print(f"Status: {'✅' if log['execution']['success'] else '❌'}")
        print("-" * 100)
    
    print()


def main():
    parser = argparse.ArgumentParser(description="Query Log Viewer")
    parser.add_argument(
        "action",
        choices=["view", "stats", "search"],
        help="Action to perform"
    )
    parser.add_argument(
        "--date",
        "-d",
        help="Date in YYYY-MM-DD format (default: today)"
    )
    parser.add_argument(
        "--type",
        "-t",
        choices=["sql_queries", "routing_decisions"],
        default="sql_queries",
        help="Log type to view"
    )
    parser.add_argument(
        "--limit",
        "-l",
        type=int,
        help="Limit number of entries to display"
    )
    parser.add_argument(
        "--keyword",
        "-k",
        help="Keyword to search for"
    )
    
    args = parser.parse_args()
    
    if args.action == "view":
        view_logs(args.date, args.type, args.limit)
    elif args.action == "stats":
        show_statistics(args.date)
    elif args.action == "search":
        if not args.keyword:
            print("❌ Error: --keyword is required for search action")
            return
        search_logs(args.keyword, args.date)


if __name__ == "__main__":
    main()

