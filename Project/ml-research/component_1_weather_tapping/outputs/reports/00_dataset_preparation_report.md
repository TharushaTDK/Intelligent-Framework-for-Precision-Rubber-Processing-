# Stage 00 - dataset preparation

Source: `data\raw\Combined_Rubber_Latex_Dataset_Condition_Matched.xlsx` (sheet `Combined_Dataset`, unchanged).
Prepared copy: `data\validated\component1_realistic_prepared.csv`.

## Rows
- Total rows: 1210
- Farmer records (Open-Meteo weather joined): 610
- Model-development records (earlier synthetic set, no weather): 600
- Usable for weather-based models: 610

## Latex price
Price per kg of field latex = farm-gate latex price (Rs per kg dry rubber) x DRC / 100.
Field latex is paid on its dry rubber content, so watery (low-DRC) latex earns less per kg.

Farm-gate level used, by source status:

```
                           rows
price_basis_status             
ESTIMATED_CARRIED_FORWARD   305
ESTIMATED_FROM_CBSL_FOB     305
RDD_2025_ANNUAL_AVERAGE     600
```

Usable rows by visible latex condition:

```
                         rows  mean_drc_pct  mean_price_lkr_per_kg
visible_latex_condition                                           
Early coagulation          82         38.16                 292.39
Normal                    387         40.24                 307.19
Watery                    141         36.01                 268.90
```

## Rain forecast for the next 6 hours
The workbook's `rainfall_next_6h_observed_mm` is rain that fell afterwards, so it is kept only
as the outcome the tapping label is judged on. The model input `forecast_rain_next_6h_mm` is
the Open-Meteo forecast for the same grid point issued about a day earlier:

    https://previous-runs-api.open-meteo.com/v1/forecast?latitude=6.7838&longitude=80.1515&start_date=2026-05-29&end_date=2026-10-02&timezone=Asia%2FColombo&hourly=precipitation,precipitation_previous_day1

Correlation between forecast and observed next-6 h rain on usable rows: 0.47.
A forecast issued a day ahead is a little less accurate than the one a farmer sees at dawn,
so model scores here are, if anything, slightly pessimistic.

## Caveats carried over from the workbook
- DRC and Metrolac readings were normalized to valid Kudaligama chart entries (near 41%) for
  model development; they are not original field measurements.
- Visible latex condition was aligned with DRC by the dataset author (Watery lower, Normal 38-42%).
- Soil moisture used as a model input is the farmer-recorded value; the Open-Meteo value is
  modelled, not a sensor reading.
- June-September 2026 farm-gate levels are estimates (see the price reference file).
