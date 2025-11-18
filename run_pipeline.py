"""
Wrapper script to run the data pipeline

This script imports and executes the main pipeline from app/core/pipeline
"""
from app.core.pipeline.run_data_pipeline import main

if __name__ == "__main__":
    main()
