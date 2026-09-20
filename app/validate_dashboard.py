import os


REQUIRED_FILES = [

    "data/processed/solar_uncertainty.csv",

    "data/processed/wind_uncertainty.csv",

    "data/processed/storage_backup_simulation.csv",

    "data/processed/cost_co2_impact.csv",

    "reports/uncertainty_summary.csv",

    "reports/storage_backup_summary.csv",

    "reports/cost_co2_summary.csv",

    "app/dashboard.py"

]


def main():

    print(
        "Starting Phase 14 dashboard validation..."
    )

    missing_files = [

        file_path
        for file_path in REQUIRED_FILES
        if not os.path.exists(file_path)

    ]

    if missing_files:

        raise FileNotFoundError(
            f"Missing required files: {missing_files}"
        )

    if os.path.getsize(
        "app/dashboard.py"
    ) == 0:

        raise ValueError(
            "Dashboard file is empty."
        )

    print(
        "\nRequired dashboard files verified:"
    )

    for file_path in REQUIRED_FILES:

        print(
            f"[OK] {file_path}"
        )

    print(
        "\nFINAL PHASE 14 DASHBOARD VALIDATION: PASSED"
    )


if __name__ == "__main__":
    main()