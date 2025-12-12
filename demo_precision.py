"""
Demo script for the precision disparity method.

Run this in CML Workbench as the main script.
It will:
  - generate a synthetic linkage dataset
  - compute precision disparity by subgroup
  - print a summary and first rows of the results table
  - save results to precision_disparity_results.csv
"""

from bias_work.create_syn_dataset import generate_synthetic_linkage_data
from bias_work.precision_disparity import PrecisionDisparityAnalyser


def main() -> None:
    # --------------------------------------------------------
    # 1. Generate synthetic linkage data
    # --------------------------------------------------------
    df = generate_synthetic_linkage_data(
        N=5000,
        seed=123,
        scenario="ethnicity_bias",  # change this to try other scenarios
    )

    print(f"Generated synthetic dataset with {len(df)} rows")
    print("\nSample of input data (first 5 rows):")
    print(df.head())

    # --------------------------------------------------------
    # 2. Run precision disparity analysis
    # --------------------------------------------------------
    analyser = PrecisionDisparityAnalyser(
        df=df,
        truth_col="link_truth",
        group_vars=["sex", "age_band", "ethnicity", "id_quality"],
        min_linked=50,        # flag groups with < 50 linked records
        dropna_groups=True,   # drop rows with missing group values
    )

    results = analyser.run()

    # --------------------------------------------------------
    # 3. Print summary + first rows of results
    # --------------------------------------------------------
    print("\n=== Precision disparity summary ===")
    print(analyser.summary())

    print("\n=== Head of results table (first 20 rows) ===")
    print(results.head(20))

    # --------------------------------------------------------
    # 4. Save full results to CSV
    # --------------------------------------------------------
    out_path = "precision_disparity_results.csv"
    results.to_csv(out_path, index=False)
    print(f"\nFull results written to: {out_path}")


if __name__ == "__main__":
    main()
