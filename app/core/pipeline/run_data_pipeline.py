"""
Data pipeline: Step 1 - Crawl job postings from 9 companies
"""
import time
from datetime import datetime
import traceback
from app.services.crawler.call_all_crawler import run_all_crawlers_sequentially

def main():
    pipeline_start = time.time()

    # Import crawler function
    print("\n📦 Loading crawler module...")

    # Execute crawling
    print("\n" + "="*80)
    print("STARTING CRAWLERS")
    print("="*80)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80 + "\n")

    start_time = time.time()

    try:
        # Execute the crawler function
        run_all_crawlers_sequentially()

        duration = time.time() - start_time
        print("\n" + "="*80)
        print("✅ CRAWLING COMPLETED")
        print(f"Duration: {duration/60:.2f} minutes")
        print("="*80 + "\n")

    except Exception as e:
        duration = time.time() - start_time
        print("\n" + "="*80)
        print("❌ CRAWLING FAILED")
        print(f"Error: {e}")
        print(f"Duration: {duration/60:.2f} minutes")
        print("="*80 + "\n")
        traceback.print_exc()
        return

    # Final summary
    pipeline_duration = time.time() - pipeline_start

    print("\n" + "="*80)
    print("PIPELINE SUMMARY")
    print("="*80)
    print(f"Total Duration: {pipeline_duration/60:.2f} minutes")
    print("Status: ✅ Success")
    print("="*80 + "\n")

    print("🎉 Crawling completed successfully!")
    print("📁 Check data/output/ for the crawled JSON files")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Pipeline interrupted by user (Ctrl+C)")
    except Exception as e:
        print(f"\n\n❌ Unexpected error: {e}")
        traceback.print_exc()
