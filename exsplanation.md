# Electricity dataset explanations

These explanations describe the CSV files in `Data/Raw/ENSTOE/`. All examples below are copied from your saved data. The main files currently span **1 January 2021 through 31 December 2024**, based on their first and last timestamps inspected on **28 October 2026**. This confirms their outer date range, but does not prove every interval inside it is present.

In these files, **`datetime` refers to the start of the delivery or operating interval**, not when the CSV was downloaded. For forecasts, it identifies the period being predicted, not when the forecast was issued. `+01:00` means one hour ahead of UTC, corresponding to French winter time; French summer time uses `+02:00`. Files with Great Britain as the source can use UK time, so `2020-12-31 23:00:00+00:00` represents the same instant as `2021-01-01 00:00:00+01:00`. Convert to a common timezone before joining datasets. Interval lengths can change across years and borders, and a gap can also represent missing data.

**MW** measures power, **MWh** measures energy, and **€/MWh** measures price. To convert average power to energy, multiply MW by the interval duration in hours: **100 MW for 15 minutes = 25 MWh**. A blank CSV value means no value was supplied for that field; it should not automatically be treated as zero.

**actual_load_fr.csv**

```csv
datetime,Actual Load
2021-01-01 00:00:00+01:00,65761.0
```

The `actual_load_fr.csv` dataset records France's actual total electricity demand during each operating interval, measured in **megawatts (MW)**. `datetime` marks the interval's start, and `Actual Load` gives the reported average demand during it. This example means that during **00:00–01:00 on 1 January 2021**, demand averaged **65,761 MW**, equivalent to **65.761 GW**. The beginning of your file is hourly; later records use 15-minute intervals, so this duration must not be assumed for every row. Demand of 65,761 MW sustained for one hour corresponds to **65,761 MWh** of energy. This is actual demand, determined from observations for the operating period and published afterwards, rather than a prediction made before midnight. The CSV does not include the exact publication time or revision history.

**balancing_energy_prices_fr.csv**

```csv
datetime,Direction,Price,ReserveType
2021-01-01 00:00:00+01:00,Up,50.87,FCR
2021-01-01 00:00:00+01:00,Down,0.0,RR
```

The `balancing_energy_prices_fr.csv` dataset contains reported balancing energy prices for France, measured in **euros per megawatt-hour (€/MWh)**. `Direction` describes upward or downward balancing, `Price` is the reported rate, and `ReserveType` identifies the reserve service. The first row reports an **upward FCR price of €50.87/MWh** for the operating interval beginning at midnight on **1 January 2021**. **FCR** means **Frequency Containment Reserve**; other labels include **aFRR** (automatic frequency restoration reserve), **mFRR** (manual frequency restoration reserve), and **RR** (replacement reserve). Upward action increases generation or reduces demand; downward action reduces generation or increases demand. **Up does not mean the electricity price increased, and €50.87 is not an amount of energy added.** The second row reports **€0/MWh for downward RR**; a zero price does not prove zero activation. These rows do not give activated MWh or total payments. Multiple services and directions share timestamps. FCR records at the beginning are spaced 30 minutes apart, while later data includes 15-minute timestamps; use each underlying series' resolution rather than adjacent rows across the whole table. The timestamp concerns operation, not publication, and the export does not retain publication times or all product details. The library's [balancing parser](https://github.com/EnergieID/entsoe-py/blob/master/entsoe/parsers.py#L571) supplies these direction and reserve labels.

**day_ahead_prices_fr.csv**

```csv
datetime,0
2021-01-01 00:00:00+01:00,50.87
```

The `day_ahead_prices_fr.csv` dataset contains France's wholesale **day-ahead market electricity price**, measured in **€/MWh**. The column named `0` is the price column; its name is an automatic label, not a unit. This row means the price for electricity delivered during **00:00–01:00 on 1 January 2021** was **€50.87 per MWh**. It does not mean the price rose by €50.87 at midnight. A hypothetical purchase of 10 MWh at this rate would have an energy cost of **€508.70**, before other charges. These prices were determined in the day-ahead market before delivery; the timestamp marks delivery, not the auction or publication time. Your 2021–2024 file contains hourly delivery timestamps. This is a wholesale price, not a household tariff; negative wholesale prices are also possible. [ENTSO-E dataset definition, Article 12.1.D](https://www.entsoe.eu/fileadmin/user_upload/_library/resources/Transparency/02_MoP%20Ref02%20-%20DDD_V2R5.pdf)

**load_forecast_fr.csv**

```csv
datetime,Forecasted Load
2021-01-01 00:00:00+01:00,66500.0
```

The `load_forecast_fr.csv` dataset contains the **day-ahead forecast of France's total electricity demand**, measured in **MW**. This row predicts an average demand of **66,500 MW**, or **66.5 GW**, during **00:00–01:00 on 1 January 2021**. The prediction was prepared for delivery ahead of time; midnight is the start of the predicted interval, not the time the forecast was created. Comparing it with the actual-load example of **65,761 MW** shows the forecast was **739 MW higher than actual demand** for that hour. That comparison requires matching timestamps and interval lengths. The file begins with hourly intervals and ends with 15-minute intervals in 2024. The CSV does not include forecast issue times or revision history, so it alone cannot establish the exact version available at a particular historical decision time.

**generation_actual_fr.csv**

```csv
,Fossil Hard coal,Hydro Pumped Storage,Biomass,Fossil Gas,Fossil Hard coal,Fossil Oil,Hydro Pumped Storage,Hydro Run-of-river and pondage,Hydro Water Reservoir,Nuclear,Solar,Waste,Wind Onshore,Hydro Water Reservoir,Wind Offshore,Wind Offshore
,Actual Consumption,Actual Consumption,Actual Aggregated,Actual Aggregated,Actual Aggregated,Actual Aggregated,Actual Aggregated,Actual Aggregated,Actual Aggregated,Actual Aggregated,Actual Aggregated,Actual Aggregated,Actual Aggregated,Actual Consumption,Actual Aggregated,Actual Consumption
datetime,,,,,,,,,,,,,,,,
2021-01-01 00:00:00+01:00,,,365.0,5292.0,9.0,198.0,630.0,4844.0,1107.0,50378.0,0.0,247.0,1082.0,,,
```

The `generation_actual_fr.csv` dataset records France's **actual electricity generation by production type**, measured in **MW**. This example covers **00:00–01:00 on 1 January 2021**. Among its values, nuclear generation averaged **50,378 MW**, gas generation **5,292 MW**, onshore wind **1,082 MW**, and solar **0 MW**. These are power outputs for that hour, not prices or installed capacities. The file has **two column-header levels**: the first identifies the source and the second distinguishes `Actual Aggregated` generation from `Actual Consumption`. For example, pumped storage can generate electricity when releasing water and consume electricity when pumping it uphill; consumption must not be added as positive generation. Empty fields mean missing or unreported values, not necessarily zero. The extra header lines are not data rows. Your file begins hourly and ends at 15-minute resolution. These are retrospective operating observations, rather than forecasts. RTE explains that actual generation uses measurements and, for some sources, estimates. [RTE actual-generation description](https://data.rte-france.org/web/guest/catalog/-/api/generation/Actual-Generation/v1.1)

**renewables_forecast_fr.csv**

```csv
datetime,Solar,Wind Onshore,Wind Offshore
2021-01-01 00:00:00+01:00,0.0,1280.8,
```

The `renewables_forecast_fr.csv` dataset contains **day-ahead forecasts of French solar and wind generation**, measured in **MW**. This row predicts **0 MW of solar output** and **1,280.8 MW of onshore wind output** during **00:00–01:00 on 1 January 2021**. Its offshore-wind field is blank, meaning no value is supplied in this row; it should not be interpreted automatically as zero. The timestamp marks the hour when generation was expected, not when the prediction was issued. Your file has hourly timestamps across the inspected endpoints. Comparing the wind forecast with actual onshore generation of **1,082 MW** for the same hour gives a forecast overestimate of **198.8 MW**. These forecasts are separate from total demand forecasts and actual renewable generation. The export does not carry an issue timestamp alongside each forecast. [ENTSO-E wind and solar forecast definition](https://eepublicdownloads.entsoe.eu/clean-documents/Transparency/MoP_Ref2_DDD_v3r4.pdf)

**imbalance_prices_fr.csv**

```csv
datetime,Long,Short
2021-01-01 00:00:00+01:00,62.28,67.88
```

The `imbalance_prices_fr.csv` dataset contains French **imbalance settlement prices**, measured in **€/MWh**. These are used to financially settle a market participant's deviation from its contracted electricity position. `Long` refers to a surplus position and `Short` to a shortfall position. This row reports **€62.28/MWh for a long position** and **€67.88/MWh for a short position** during **00:00–00:30 on 1 January 2021**. For illustration, a positive price of €67.88/MWh applied to a **2 MWh shortfall** gives an imbalance charge of **€135.76**. The row does not say that the whole French system was simultaneously long and short: it lists settlement rates for the two positions. The beginning of the file uses 30-minute intervals; later records use 15-minute intervals. The timestamp identifies the settlement period, not when its price became known. These are retrospective settlement data, distinct from day-ahead prices and individual balancing-service prices. [RTE imbalance-settlement explanation](https://www.services-rte.com/en/learn-more-about-our-services/becoming-a-balance-responsible-party/Imbalance-settlement-price.html)

**imbalance_volumes_fr.csv**

```csv
datetime,Imbalance Volume
2021-01-01 00:00:00+01:00,-499.43
```

The `imbalance_volumes_fr.csv` dataset reports the **aggregate imbalance energy for the French balancing area**, measured in **MWh per settlement interval**. This example records a signed imbalance of **−499.43 MWh** during **00:00–00:30 on 1 January 2021**. Its magnitude is **499.43 MWh**; expressed as an average power magnitude over half an hour, that is **998.86 MW**. This is neither a price nor total electricity demand. The sign records imbalance direction according to the source/library convention. RTE generally describes a negative system imbalance as a shortage, but the exported sign alone should not be treated as independently verified evidence of that direction without checking the original source encoding, because the library can apply signs from direction codes. The file begins at 30-minute resolution and ends with 15-minute intervals. Values describe an operating period and are published afterwards; they may be revised. [ENTSO-E imbalance-volume definition](https://www.entsoe.eu/fileadmin/user_upload/_library/resources/Transparency/02_MoP%20Ref02%20-%20DDD_V2R5.pdf), [library sign handling](https://github.com/EnergieID/entsoe-py/blob/master/entsoe/parsers.py#L652)

**physical_flows_CH_FR.csv — example for all physical-flow files**

```csv
datetime,0
2021-01-01 00:00:00+01:00,542.0
```

The `physical_flows_CH_FR.csv` dataset reports **actual cross-border electricity flow from Switzerland to France**, measured in **MW**. The filename order is **source to destination**, so `CH_FR` means imports into France from Switzerland; `FR_CH` describes the opposite direction. This row reports an average transfer of **542 MW from Switzerland to France** during **00:00–01:00 on 1 January 2021**, corresponding to **542 MWh** over that hour. The column named `0` contains the flow value. This measures actual transfer during the operating interval, rather than a forecast, price or scheduled trade. The example file has hourly timestamps at both ends, while other border files can have shorter intervals or change resolution across years. Zero means a reported zero in that direction; check the opposite-direction file too. Actual physical flows can differ from commercial schedules because electricity follows the interconnected network. The same explanation applies to every `physical_flows_*.csv` file after changing the areas and checking its interval length. [ENTSO-E physical-flow definition, Article 12.1.G](https://www.entsoe.eu/fileadmin/user_upload/_library/resources/Transparency/02_MoP%20Ref02%20-%20DDD_V2R5.pdf)

**scheduled_CH_FR.csv — example for all scheduled-exchange files**

```csv
datetime,0
2021-01-01 00:00:00+01:00,194.0
```

The `scheduled_CH_FR.csv` dataset records **planned commercial electricity exchanges from Switzerland to France**, measured in **MW**. Your downloader requests **day-ahead schedules** using `dayahead=True`. This row means **194 MW** was scheduled for transfer from Switzerland to France during **00:00–01:00 on 1 January 2021**, equivalent to **194 MWh over that hour**. The column named `0` contains the scheduled power. The timestamp refers to delivery, not when the schedule was created or the trade executed. Compare this with the physical-flow example of **542 MW**: the planned commercial transfer and actual physical transfer are different measurements and need not match. This example file is hourly; other scheduled files have 30-minute or 15-minute intervals in some periods. `scheduled_FR_CH.csv` describes the reverse direction. The same explanation applies to every `scheduled_*.csv` file after adjusting the source, destination and interval length. The CSV does not show the original schedule publication time. [ENTSO-E scheduled-exchange definition, Article 12.1.F](https://www.entsoe.eu/fileadmin/user_upload/_library/resources/Transparency/02_MoP%20Ref02%20-%20DDD_V2R5.pdf)

**Country and bidding-zone codes**

| Code | Area |
|---|---|
| FR | France |
| BE | Belgium |
| CH | Switzerland |
| DE_LU | Germany–Luxembourg bidding zone |
| IT_NORD | Northern Italy bidding zone |
| ES | Spain |
| GB | Great Britain |

**Monthly balancing files**

Files inside `balancing_monthly/` contain the same balancing-price columns, separated by month so downloads can resume. Their data is combined in `balancing_energy_prices_fr.csv`. For analysis, use the combined file or the monthly files, rather than appending both sets together and counting observations twice.
