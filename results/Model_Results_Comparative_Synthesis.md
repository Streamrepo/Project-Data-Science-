# Comparative results and discussion

All numbers below come from the files in `results/model_1` to `results/model_5` and are the same as in `report/report.ipynb`.

## A. Overview of the modelling approaches

The question is whether information available after the French day-ahead market clears, and before delivery, can rank half-hour intervals by the risk of severe imbalance settlement stress. Stress is directional. Short-side stress is the short imbalance price minus the day-ahead price. Long-side stress is the day-ahead price minus the long imbalance price. Each classifier predicts whether its stress is above the 2021-22 90th percentile, 77.83 EUR/MWh for short and 78.35 EUR/MWh for long. The size of an exceedance is not predicted.

| Model | Classifier for each side | Inputs |
|---|---|---|
| 1 | L2 logistic regression | 17 day-ahead and calendar predictors |
| 2 | L2 logistic regression | The 17 predictors plus two regime probabilities from a three-state Gaussian HMM |
| 3 | XGBoost | The same 17 predictors as Model 1 |
| 4 | XGBoost | The 17 predictors plus the same two regime probabilities |
| 5 | Logistic regression, boosted trees or their average, then Platt calibration | The 17 predictors plus 7 rolling features (price relative to the last 7 and 30 days, residual load, and the share of severe half-hours over days D-8 to D-2) |

The 17 base predictors are six standardised continuous variables (day-ahead price, load forecast, onshore wind forecast, solar forecast, scheduled imports and scheduled exports), two cyclical time-of-day terms, six weekday indicators and three season indicators. The HMM is an unsupervised feature generator shared by Models 2 and 4. The naive reference forecasts the 2021-22 event rate at every interval. Code: [Model 1](../src/models/model_1_logistic_regression.py), [Model 2](../src/models/model_2_logistic_hmm.py), [Model 3](../src/models/model_3_xgboost.py), [Model 4](../src/models/model_4_xgboost_hmm.py), [Model 5](../src/models/model_5_rolling.py), [shared definitions](../src/models/model_234_common.py).

Models 1 to 4 are fitted on 2021-22 (34,111 rows), choose their settings by average precision on 2023 (16,409 rows) and are scored once on 2024 (17,481 rows). Model 5 chooses its settings on two rolling folds (fit on 2021 and score 2022, then fit on 2021-22 and score 2023), is refitted on 2021-2023 and is scored once on 2024. Every model chooses its own settings for each target, so a comparison between two models includes any change in the selected settings.

## B. Comparative predictive performance

Average precision (AP, `pr_auc` in the CSV files) summarises precision across recall levels, and its no-skill value is the share of severe intervals in the period. Precision at 10% is the event rate among the highest-scoring tenth of intervals. Brier score and log loss measure the probabilities themselves, and lower is better. The constant forecast uses the 2021-22 event rate: Brier 0.0929 and log loss 0.3330 (short), 0.0743 and 0.2821 (long) on 2024.

| Target | Model | Train ROC-AUC / AP | Validation ROC-AUC / AP | Test ROC-AUC | Test AP | Test precision at 10% | Test Brier | Test log loss |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Short | 1 | 0.7151 / 0.2063 | 0.6041 / 0.1008 | 0.5302 | 0.1129 | 0.1167 | 0.0980 | 0.3955 |
| Short | 2 | 0.7169 / 0.2081 | 0.6070 / 0.1018 | 0.5357 | 0.1147 | 0.1190 | 0.0979 | 0.3947 |
| Short | 3 | 0.7575 / 0.2735 | 0.6005 / 0.0950 | 0.5542 | 0.1228 | 0.1419 | 0.0969 | 0.3772 |
| Short | 4 | 0.7583 / 0.2712 | 0.6005 / 0.0965 | 0.5646 | 0.1284 | **0.1573** | 0.0961 | 0.3637 |
| Short | 5 | 0.7286 / 0.2276 | 0.6226 / 0.1104 | **0.5719** | **0.1330** | 0.1568 | **0.0932** | **0.3362** |
| Long | 1 | 0.7985 / 0.2988 | 0.5948 / 0.0766 | 0.5829 | 0.1022 | 0.1098 | 0.0756 | 0.2975 |
| Long | 2 | 0.7989 / 0.2992 | 0.5912 / 0.0775 | 0.5847 | 0.1032 | 0.1018 | 0.0756 | 0.2978 |
| Long | 3 | 0.8171 / 0.3319 | 0.6880 / 0.1019 | 0.6437 | 0.1397 | 0.1590 | 0.0759 | 0.3111 |
| Long | 4 | 0.8171 / 0.3319 | 0.6880 / 0.1019 | 0.6437 | 0.1397 | 0.1590 | 0.0759 | 0.3111 |
| Long | 5 | 0.8211 / 0.3181 | 0.6907 / 0.1397 | **0.6938** | **0.1718** | **0.2054** | **0.0719** | **0.2681** |

Bold marks the best model on each test metric. For Model 5 the train and validation columns come from the model fitted on 2021-22, and the test column from the final model fitted on 2021-2023. Metrics per split are in [model_1](model_1/metrics.csv), [model_2](model_2/metrics.csv), [model_3](model_3/metrics.csv), [model_4](model_4/metrics.csv) and [model_5](model_5/metrics.csv).

**Ranking and screening.** Model 5 has the best 2024 ROC-AUC and AP on both sides. With a review budget of 1,748 intervals (10% of 2024), about 20.5% of Model 5's flagged long-side intervals were severe, against 8.0% overall, which is a lift of 2.6. That is 359 of the 1,406 long events (25.5%). On the short side Models 4 and 5 are level: about 15.7% of flagged intervals were severe against 10.4% overall (275 and 274 of 1,811 events). Most flagged intervals still do not cross the threshold, and no economic value or alert cost was evaluated.

**Probability accuracy.** On 2024 the constant forecast beats Models 1 to 4 on both Brier score and log loss, on both sides. Model 5 is the only model that beats it, and only on the long side (Brier 0.0719 against 0.0743, log loss 0.2681 against 0.2821). On the short side Model 5 is level with the constant on Brier score (0.0932 against 0.0929) and slightly worse on log loss. In 2023 every model beats the constant on both scores. Model 3 improves long AP from 0.1022 to 0.1397 relative to Model 1 while worsening log loss from 0.2975 to 0.3111, so a model can rank better and still give less accurate probabilities. The calibration step of Model 5 is what turns its gain in ranking into a gain in probabilities on the long side.

**Time dependence.** Every model has a lower ROC-AUC on each held-out year than in training. The share of severe intervals is 10% by construction in training, then 7.2% (short) and 5.3% (long) in 2023, and 10.4% and 8.0% in 2024, because the 2021-22 thresholds are kept fixed and price levels differ a lot between the 2022 energy crisis and later years. A higher AP in 2024 than in 2023 can therefore go together with a lower ROC-AUC. On the short side the logistic models lead the boosted trees in 2023 and trail them in 2024.

## C. Common findings across models

Scheduled imports and exports matter most on the short side. In Model 1 their standardised coefficients are +0.384 and -0.461, and Model 2 gives +0.393 and -0.458. In Model 3 the top three inputs by share of total gain are day-ahead price (0.202), scheduled exports (0.154) and scheduled imports (0.146). For long events the day-ahead price has the largest logistic coefficient (+0.563 in Model 1, +0.558 in Model 2) and the largest tree gain (0.380), followed by the load forecast (0.198) and the solar forecast (0.100). Coefficients express conditional associations on the log-odds scale. Tree gain measures how much each input improves the fit, and has no direction. [Model 1 coefficients](model_1/coefficients.csv), [Model 2 coefficients](model_2/coefficients.csv), [Model 3 importance](model_3/feature_importance.csv), [Model 4 importance](model_4/feature_importance.csv).

The day-ahead price also enters the target definition with opposite signs on the two sides, so its importance mixes an economic signal with a mechanical link to the quantity being predicted. Nothing here identifies what causes stress.

The two outcomes are kept apart because they behave almost as opposites. The correlation between the signed stresses is -0.988, and only 2 of 68,001 modelled intervals are severe on both sides. Their sum equals the short-long price spread, since the day-ahead price cancels. See sections 4 to 6 of [the report notebook](../report/report.ipynb).

## D. Main differences between models

**Learner (Model 1 to Model 3).** With the inputs fixed, the boosted trees raise 2024 ROC-AUC by 0.024 on the short side and 0.061 on the long side, and AP by 0.010 and 0.038. The long-side advantage appears in both 2023 and 2024. On the short side the trees trail the logistic models in 2023, so that advantage depends on the year.

**Rolling features and calibration (Model 3 to Model 5).** Model 5 adds 0.018 ROC-AUC and 0.010 AP on the short side, and 0.050 and 0.032 on the long side. A day-block bootstrap of 2024 (1,000 resamples) gives these intervals for Model 5 minus another model:

| Model 5 minus | Short AP | Short ROC-AUC | Long AP | Long ROC-AUC |
|---|---|---|---|---|
| Model 1 | +0.020 [+0.012, +0.029] | +0.041 [+0.026, +0.057] | +0.069 [+0.040, +0.105] | +0.110 [+0.082, +0.140] |
| Model 2 | +0.018 [+0.010, +0.027] | +0.036 [+0.020, +0.052] | +0.068 [+0.039, +0.104] | +0.108 [+0.079, +0.138] |
| Model 3 | +0.010 [-0.000, +0.021] | +0.018 [-0.004, +0.039] | +0.032 [+0.005, +0.059] | +0.050 [+0.027, +0.075] |
| Model 4 | +0.006 [-0.004, +0.017] | +0.009 [-0.014, +0.031] | +0.032 [+0.005, +0.059] | +0.050 [+0.027, +0.075] |

Against Models 1 and 2 every interval stays above zero. Against Models 3 and 4 the long-side gain is clear, while on the short side the intervals include zero and the data cannot separate the models. [Bootstrap file](model_5/bootstrap_differences.csv).

**Choices made by Model 5.** On the short side the logistic regression wins (C = 0.01, mean fold AP 0.1616, against 0.1598 for the average and 0.1457 for boosted trees). On the long side the average of the logistic regression and shallow boosted trees wins narrowly (0.2078 against 0.2077 for the logistic regression alone), with a very strong penalty (C = 0.0001). That choice is close to a tie. [Settings](model_5/selected.json).

Logistic regression gives signed, readable coefficients. Boosted trees give richer shapes but only importances, with no direction. The HMM adds a sequential fit and needs ordered inputs with gaps handled. No timing or memory benchmark was made.

All models use a binary target. An interval 1 EUR/MWh above the cut-off and one 1,000 EUR/MWh above count the same, so these results say nothing about sensitivity to the largest losses.

## E. Contribution and usefulness of each model

**Model 1** is the interpretable benchmark. It shows which fundamentals are associated with stress, but its short-side ROC-AUC of 0.530 is barely above chance, so it is weak as a screening tool.

**Model 2** tests whether regimes help a logistic model. It adds 0.006 ROC-AUC and 0.002 AP on the short side and almost nothing on the long side, where precision at 10% falls slightly. It works as an ablation, with little evidence of operational benefit.

**Model 3** shows what non-linear modelling gains from the same inputs. It is a clear step up on the long side (ROC-AUC 0.644, AP 0.140) and a smaller one on the short side.

**Model 4** tests regimes after allowing non-linear relationships. On the long side the trees never use the regime probabilities, so it gives exactly the same predictions as Model 3. On the short side it selects different settings (depth 3 and 100 trees, against depth 2 and 300 trees for Model 3), and the regime probabilities carry about 10% of the total gain.

**Model 5** is the best model on ranking on both sides and the only one with better probabilities than the constant forecast on any side. Its gain over Models 3 and 4 is clear on the long side and not separable from zero on the short side.

## F. Added value of the HMM regimes

| Difference on 2024 | Target | ROC-AUC | AP | Precision at 10% | Brier | Log loss |
|---|---|---:|---:|---:|---:|---:|
| Model 2 minus Model 1 | Short | +0.0055 | +0.0018 | +0.0023 | -0.0001 | -0.0008 |
| Model 2 minus Model 1 | Long | +0.0018 | +0.0010 | -0.0080 | +0.0000 | +0.0003 |
| Model 4 minus Model 3 | Short | +0.0104 | +0.0056 | +0.0154 | -0.0009 | -0.0135 |
| Model 4 minus Model 3 | Long | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

The short-side gains of Model 4 are modest, they come with changed tree settings, and no interval or test supports a claim that they persist across years. The long-side change is exactly zero for Model 4 and negligible for Model 2. The experiments cannot tell whether the short-side gain comes from latent regimes, from persistence in inputs the trees already see, or from a useful non-linear transformation. Model 5, which uses no HMM, matches Model 4 on short-side screening, which suggests the second or third reason, although this is not tested.

The HMM has three states with diagonal Gaussian emissions. It is fitted on 34,686 half-hours of 2021-22 inputs in 10 unbroken runs, with three random starts and the best training likelihood kept. Probabilities are forward-filtered, so each one depends only on the current and earlier observations. Gaps move the state forward with the transition matrix and no emission update. The mean day-ahead prices of the states are about 100, 182 and 305 EUR/MWh. The middle state has an almost zero solar forecast, so it mostly marks night. All three states stay put from one half-hour to the next with probability above 0.95, which is a run of about 10 to 12 hours. These are persistent conditions of the observable inputs, and not proven shortage regimes. [Model 2 regime summary](model_2/regimes.json), [Model 4 regime summary](model_4/regimes.json).

## G. Overall interpretation

Day-ahead information gives a measurable signal for ranking severe deviations in the French market, particularly on the long side, where Model 5 reaches a 2024 ROC-AUC of 0.694 and an AP of 0.172 against a base rate of 0.080. On the short side the best 2024 ROC-AUC is 0.572 and the best AP is 0.133 against a base rate of 0.104, which is a weak signal. For most models the gap between ranking and probability accuracy limits their use as raw probabilities in a decision rule. Severe short-side stress probably comes mostly from events after the auction, such as outages, forecast errors and intraday moves, that day-ahead data cannot see.

## H. Limitations

**Information timing.** The inputs are day-ahead prices, forecasts, scheduled exchanges and calendar terms. Same-period actual load, generation, physical flows and settlement outcomes are not inputs, and they do not decide which rows are modelled. A half-hour is modelled when all six day-ahead inputs and both stress scores exist. Scaling, thresholds, the HMM and the Model 5 rolling features use only past data, and 13 automated tests check this. The raw files have delivery timestamps but no publication timestamps, so we cannot prove that forecasts were not revised after the auction. Forecast gaps are filled forward for at most two hours.

**Scope of the evidence.** All final results rest on 2024 alone, and severe events come in clusters, so there are far fewer independent events than rows. Differences of a few thousandths in AP should not be read as real. The bootstrap intervals do not account for the uncertainty of fitting the models or for the number of comparisons. Model 5 is refitted on 2021-2023, so it has seen one more year than Models 1 to 4 before the 2024 test and the comparison is not strictly like for like. Model 4's short-side result moved between two full re-runs (2024 ROC-AUC 0.560, then 0.565), so differences of that size between Models 3 and 4 are within the variation of the HMM and tree fits. There is no 2025 holdout, and no cost-weighted outcome or interval-level comparison.

**Conclusion.** The logistic baseline gives the clearest account of the associations. Boosted trees give the largest single improvement for long-side ranking. Rolling features and calibration (Model 5) improve ranking on both sides and give the only probabilities better than a constant forecast, on the long side. The HMM regimes add little beyond what the day-ahead inputs and the time of day already contain. Natural next steps are intraday and outage information, more test years such as 2025, and modelling how large the stress is, not only whether it is severe.

## Reproducing the results

See `README.md` in the repository root for the setup, the API key and the order in which the scripts are run. The notebook `report/report.ipynb` reads the saved tables and lists the same steps in section 4.5.
