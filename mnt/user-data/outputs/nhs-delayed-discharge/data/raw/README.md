# Data Download Instructions

The raw datasets used in this project are publicly available from Scottish government sources. They are **not included** in this repository due to file size and licensing.

## Dataset 1 — NHS Delayed Discharge by Health Board

- **Provider:** Public Health Scotland (PHS) Open Data
- **URL:** https://www.opendata.nhs.scot/dataset/delayed-discharges-in-nhsscotland
- **File to download:** `Delayed Discharge by NHS Health Board.csv`
- **Save to:** `data/raw/Delayed_Discharge_NHS.csv`

## Dataset 2 — Care Inspectorate Care Home Data

- **Provider:** Care Inspectorate Scotland
- **URL:** https://www.careinspectorate.com/index.php/statistics-and-analysis/data-and-statistics
- **File to download:** `DatastoreExternal.xls` (Care Services file)
- **Save to:** `data/raw/CareInspectorate_CareHomes.xls`

## Dataset 3 — Mid-Year Population Estimates by NHS Health Board

- **Provider:** National Records of Scotland (NRS)
- **URL:** https://www.nrscotland.gov.uk/statistics-and-data/statistics/statistics-by-theme/population/population-estimates/mid-year-population-estimates/mid-year-population-estimates-time-series-data
- **File to download:** Mid-year population estimates by NHS health board, sex and single year of age
- **Save to:** `data/raw/MidYear_PopulationEstimates.xlsx`

## Once Downloaded

Run the notebook from the beginning — all preprocessing, merging, and feature engineering steps will regenerate `data/processed/NHS_Discharge_Rate.csv` automatically.
