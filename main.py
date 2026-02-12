from pathlib import Path

import polars as pl
from pyscbwrapper import SCB

TABLES = {
    "month_tab": ("en", "AM", "AM0401", "AM0401I", "NAKUSysselYrke2012M"),
}
DEFAULT_TABLE_ID = "month_tab"
EXCLUDED_OCCUPATION_CODES = {"0000", "0002"}


def find_key(variables: dict, needle: str) -> str:
    for key in variables:
        if needle in key.lower():
            return key
    raise KeyError(f"Could not find key containing '{needle}' in SCB variables.")


def build_dataframe(tab_id: str = DEFAULT_TABLE_ID) -> pl.DataFrame:
    if tab_id not in TABLES:
        raise ValueError(f"Unknown table id '{tab_id}'. Valid ids: {list(TABLES)}")

    scb = SCB(*TABLES[tab_id])
    variables = scb.get_variables()

    occupations_key = find_key(variables, "occupation")
    observations_key = find_key(variables, "observations")
    months_key = find_key(variables, "month")
    sex_key = find_key(variables, "sex")

    occupations = variables[occupations_key]
    observations = variables[observations_key][0]
    months = variables[months_key]
    sex = variables[sex_key][:2]

    scb.set_query(
        **{
            occupations_key: occupations,
            months_key: months,
            observations_key: observations,
            sex_key: sex,
        }
    )

    scb_fetch = scb.get_data()["data"]

    query = scb.get_query()["query"]
    occupation_codes = query[0]["selection"]["values"]
    sex_codes = query[3]["selection"]["values"]
    occ_dict = dict(zip(occupation_codes, occupations))
    sex_dict = dict(zip(sex_codes, sex))

    df = (
        pl.DataFrame(scb_fetch)
        .with_columns(
            [
                pl.col("key").list.get(0).alias("code_1"),
                pl.col("key").list.get(1).alias("sex"),
                pl.col("key").list.get(2).alias("month"),
                pl.col("values").list.get(0).alias("value"),
            ]
        )
        .drop(["key", "values"])
        .with_columns(
            [
                pl.col("code_1").replace(occ_dict).alias("occupation"),
                pl.col("sex").replace(sex_dict).alias("sex"),
            ]
        )
        .filter(~pl.col("code_1").is_in(EXCLUDED_OCCUPATION_CODES))
        .with_columns(
            [
                pl.col("code_1").cast(pl.Utf8),
                pl.col("occupation").cast(pl.Utf8),
                pl.col("sex").cast(pl.Utf8),
                pl.col("month").cast(pl.Utf8),
                pl.col("value").cast(pl.Float64, strict=False),
            ]
        )
        .with_columns(
            pl.col("month")
            .str.replace("M", "-")
            .str.strptime(pl.Date, "%Y-%m")
            .dt.strftime("%Y-%b")
            .alias("month")
        )
    )

    return df


def main() -> None:
    root = Path.cwd().resolve()
    data_dir = root / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    df = build_dataframe()
    output_path = data_dir / "scb_months.parquet"
    df.write_parquet(output_path)
    print(f"Wrote {df.height} rows to {output_path}")


if __name__ == "__main__":
    main()
