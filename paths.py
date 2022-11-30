from pathlib import Path


greenhouse = Path("../data/GreenhouseClimate.parquet").resolve()
production = Path("../data/Production.parquet").resolve()
resources = Path("../data/Resources.parquet").resolve()
weather = Path("../data/Weather.parquet").resolve()