# Data Pipeline

Complete data processing pipeline from crawling to database insertion.

## Pipeline Structure

```
┌─────────────────────────────────────────────────────────────────┐
│                      DATA PIPELINE                              │
└─────────────────────────────────────────────────────────────────┘

Step 1: Crawl Job Postings
├── Script: app/services/crawler/call_all_crawler.py
├── Input: None (fetches from 9 company websites)
└── Output: data/output/*_jobs.json (9 files)

Step 2: Extract Skillsets
├── Script: app/utils/parser/extract_skillsets.py
├── Input: data/output/*_jobs.json
└── Output: Processed skill data (uses LLM)

Step 3: Parse Job Descriptions
├── Script: app/utils/parser/description_parser.py
├── Input: data/SKAX_Jobdescription.pdf
└── Output: data/description.json, data/job_description.json

Step 4: Insert Posts to Database
├── Script: app/scripts/db/insert_post_to_db.py
├── Input: data/output/*_jobs.json
└── Output: Database tables (post, company, skill, post_skill)

Step 5: Insert Job Descriptions to Database
├── Script: app/scripts/db/insert_job_description_to_db.py
├── Input: data/description.json, data/job_description.json
└── Output: Database tables (position, industry, position_skill, industry_skill)
```

## Usage

### From Project Root
```bash
python run_pipeline.py
```

### Direct Execution
```bash
python app/core/pipeline/run_data_pipeline.py
```

## Components

### 1. Crawlers (Step 1)
- **Hyundai Autoever** (async)
- **LG CNS** (async)
- **Hanwha Systems** (sync)
- **Kakao** (sync)
- **Coupang** (sync)
- **Line** (sync)
- **Naver** (sync)
- **Toss** (sync)
- **Woowahan** (sync)

All crawlers save JSON output to `data/output/`

### 2. Parsers (Steps 2-3)
- **extract_skillsets.py**: Uses LLM (GPT-4o) to extract skills from job postings
- **description_parser.py**: Uses LLM to parse PDF and generate structured job descriptions

### 3. Database Scripts (Steps 4-5)
- **insert_post_to_db.py**: Inserts job posts with company and skill relationships
- **insert_job_description_to_db.py**: Inserts positions, industries, and skill mappings

## Requirements

- Python 3.11+
- OpenAI API key (for LLM steps)
- MySQL database configured
- All dependencies installed (see requirements.txt)

## Pipeline Features

- ✅ Sequential execution with error handling
- ✅ Duration tracking for each step
- ✅ Detailed logging and progress reporting
- ✅ User confirmation before starting
- ✅ Summary report with success/failure rates
- ✅ Graceful interruption (Ctrl+C)

## Error Handling

- **Step 1 fails**: Pipeline stops (no data to process)
- **Step 2-3 fail**: Pipeline continues (warnings shown)
- **Step 4 fails**: Pipeline stops (database insertion critical)
- **Step 5 fails**: Pipeline completes (job descriptions optional)

## Notes

- Step 2 and 3 use LLM and may take significant time
- Step 1 (crawling) can take 30-60 minutes for all 9 companies
- Ensure `.env` file has valid `OPENAI_API_KEY`
- Database must be initialized before running Step 4-5
