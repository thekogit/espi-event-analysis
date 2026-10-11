"""Main entry point for espi-event-analysis."""
import argparse
from espi.pipeline import run_pipeline


def main():
    parser = argparse.ArgumentParser(description="espi-event-analysis: Warsaw Stock Exchange Event Study")
    parser.add_argument(
        "--no-overlap",
        action="store_true",
        help="Filter out clustered announcements within 5 trading days"
    )
    args = parser.parse_args()
    summary = run_pipeline(no_overlap=args.no_overlap)
    print("\nEvent Study Summary:")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
