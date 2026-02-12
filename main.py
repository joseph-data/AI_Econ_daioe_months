from pathlib import Path

import polars as pl


DAIOE_SOURCE = (
    "https://raw.githubusercontent.com/joseph-data/AI_Econ_daioe_years/development/"
    "data/daioe_scb_years_all_levels.parquet"
)

SCB_SOURCE = (
    "https://raw.githubusercontent.com/joseph-data/AI_Econ_daioe_months/daioe_pull/"
    "data/scb_months.parquet"
)


def inspect_lazy(name: str, lf: pl.LazyFrame) -> None:
    """Print a small shape summary for a LazyFrame."""
    n_rows = lf.select(pl.len()).collect().item()
    n_cols = len(lf.collect_schema())
    print(f"{name}: {n_rows:,} rows x {n_cols} columns")


def setup_paths() -> tuple[Path, Path]:
    """Create data directory and return paths used by the script."""
    root = Path(__file__).resolve().parent
    data_dir = root / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    output_path = data_dir / "scb_months_lvl1.parquet"
    return data_dir, output_path


def load_sources() -> tuple[pl.LazyFrame, pl.LazyFrame]:
    """Load DAIOE and SCB data as lazy frames."""
    daioe_lf = pl.scan_parquet(DAIOE_SOURCE)
    scb_lf = pl.scan_parquet(SCB_SOURCE)
    return daioe_lf, scb_lf


def clean_scb_months(scb_lf: pl.LazyFrame) -> pl.LazyFrame:
    """Remove military records and create a numeric year column."""
    return scb_lf.filter(~pl.col("code_1").str.starts_with("0")).with_columns(
        pl.col("month").str.extract(r"^(\d{4})", 1).cast(pl.Int64).alias("year")
    )


def prepare_daioe_level1(daioe_lf: pl.LazyFrame) -> pl.LazyFrame:
    """Keep SSYK1 DAIOE rows and compute yearly weighted means."""
    return (
        daioe_lf.filter(pl.col("level") == "SSYK1")
        .select(
            pl.col(["level", "ssyk_code", "year", "weight_sum"]),
            pl.col("^daioe_.*$"),
            pl.col("^pctl_daioe_.*$"),
        )
        .group_by(["level", "ssyk_code", "year"])
        .agg(
            [
                pl.col("weight_sum").mean().cast(pl.Int64),
                pl.col("^daioe_.*$").mean(),
                pl.col("^pctl_daioe_.*$").mean(),
            ]
        )
    )


def merge_months_with_daioe(
    scb_clean_lf: pl.LazyFrame, daioe_lvl1_lf: pl.LazyFrame
) -> pl.LazyFrame:
    """Join cleaned SCB months with DAIOE level-1 yearly metrics."""
    return scb_clean_lf.join(
        daioe_lvl1_lf,
        left_on=["code_1", "year"],
        right_on=["ssyk_code", "year"],
        how="left",
    ).drop("level")


def main() -> None:
    print("Step 1/7: Setup paths")
    _, output_path = setup_paths()

    print("Step 2/7: Load DAIOE and SCB sources")
    daioe_lf, scb_lf = load_sources()

    print("Step 3/7: Quick raw-data preview")
    print("DAIOE preview:")
    print(daioe_lf.head(5).collect())
    print("SCB preview:")
    print(scb_lf.head(5).collect())

    print("Step 4/7: Clean SCB monthly data")
    scb_clean_lf = clean_scb_months(scb_lf)
    inspect_lazy("Clean SCB", scb_clean_lf)

    print("Step 5/7: Prepare DAIOE level-1 aggregates")
    daioe_lvl1_lf = prepare_daioe_level1(daioe_lf)
    inspect_lazy("DAIOE level-1", daioe_lvl1_lf)

    print("Step 6/7: Merge cleaned SCB with DAIOE")
    scb_months_lf = merge_months_with_daioe(scb_clean_lf, daioe_lvl1_lf)
    inspect_lazy("Merged output", scb_months_lf)

    print("Step 7/7: Export parquet")
    scb_months_lf.sink_parquet(output_path)
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
