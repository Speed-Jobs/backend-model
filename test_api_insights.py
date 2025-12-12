"""
Test script for competition intensity API with insights
"""
import json
from app.db.config.base import get_db
from app.services.dashboard.recruitment_schedule import get_competition_intensity

def test_api_with_insights():
    """Test the API with include_insights=True"""
    db = next(get_db())

    try:
        # Test parameters
        start_date = "2025-11-01"
        end_date = "2025-11-30"
        type_filter = "신입"

        print("="*80)
        print("Testing competition-intensity API with insights")
        print("="*80)
        print(f"Parameters:")
        print(f"  - start_date: {start_date}")
        print(f"  - end_date: {end_date}")
        print(f"  - type_filter: {type_filter}")
        print(f"  - include_insights: True")
        print()

        # Call service with include_insights=True
        result = get_competition_intensity(
            db=db,
            start_date=start_date,
            end_date=end_date,
            type_filter=type_filter,
            include_insights=True
        )

        # Print result
        print("Result:")
        print(json.dumps(result, ensure_ascii=False, indent=2))

        # Verify insights are included
        if "data" in result and "insights" in result["data"]:
            print("\n" + "="*80)
            print("✓ SUCCESS: Insights are included in the response!")
            print("="*80)

            insights = result["data"]["insights"]
            print(f"\nInsights sections:")
            print(f"  - statistics: {bool(insights.get('statistics'))}")
            print(f"  - recommended_dates: {bool(insights.get('recommended_dates'))}")
            print(f"  - company_patterns: {bool(insights.get('company_patterns'))}")
            print(f"  - optimal_period: {bool(insights.get('optimal_period'))}")
        else:
            print("\n" + "="*80)
            print("✗ FAILED: Insights are NOT included in the response")
            print("="*80)

    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
    finally:
        db.close()


def test_api_without_insights():
    """Test the API with include_insights=False (default)"""
    db = next(get_db())

    try:
        # Test parameters
        start_date = "2025-11-01"
        end_date = "2025-11-30"
        type_filter = "신입"

        print("\n" + "="*80)
        print("Testing competition-intensity API without insights (default)")
        print("="*80)
        print(f"Parameters:")
        print(f"  - start_date: {start_date}")
        print(f"  - end_date: {end_date}")
        print(f"  - type_filter: {type_filter}")
        print(f"  - include_insights: False (default)")
        print()

        # Call service with include_insights=False (default)
        result = get_competition_intensity(
            db=db,
            start_date=start_date,
            end_date=end_date,
            type_filter=type_filter,
            include_insights=False
        )

        # Verify insights are NOT included
        if "data" in result and "insights" not in result["data"]:
            print("✓ SUCCESS: Insights are NOT included (as expected for default behavior)")
            print(f"Response keys: {list(result['data'].keys())}")
        else:
            print("✗ FAILED: Insights should NOT be included when include_insights=False")

    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
    finally:
        db.close()


if __name__ == "__main__":
    # Test with insights
    test_api_with_insights()

    # Test without insights
    test_api_without_insights()
