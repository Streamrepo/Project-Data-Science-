# Comparative results and discussion

## A. Overview of the modelling approaches

The experiments ask whether information available after the French day-ahead market clears and before delivery can rank half-hour intervals by the risk of severe imbalance settlement stress. The implemented outcomes are **directional**. Short-side stress is the short imbalance settlement price minus the day-ahead price. Long-side stress is the day-ahead price minus the long imbalance settlement price. Each classifier predicts whether its stress exceeds the corresponding 2021–2022 90th percentile, €77.86/MWh for short and €78.27/MWh for long. Absolute deviations would define a different event set. The reported results concern the probability of crossing these directional thresholds. The settlement price and the size of an exceedance were outside the prediction task. [Target construction and modelling plan](https://github.com/Streamrepo/Project-Data-Science-/blob/9b8183a/Modelling_Plan.md)

| Model | Classifier for each side | Inputs and comparison |
|---|---|---|
| 1 | L2-regularized logistic regression | Reference with 17 day-ahead and calendar predictors |
| 2 | L2-regularized logistic regression | Same 17 predictors plus two probabilities from a three-state Gaussian HMM |
| 3 | XGBoost | Same 17 predictors as Model 1, with nonlinear tree splits and interactions |
| 4 | XGBoost | Same XGBoost search protocol as Model 3, plus the same two probabilities as Model 2 |

The 17 base predictors comprise six standardized continuous variables (day-ahead price, load forecast, onshore wind forecast, solar forecast, scheduled imports and scheduled exports), two cyclical time-of-day terms, six weekday indicators and three season indicators. The HMM is an unsupervised feature generator shared by Models 2 and 4. The repository contains no standalone HMM classifier score. The naive reference forecasts the corresponding training event rate at every interval. [Model 1](https://github.com/Streamrepo/Project-Data-Science-/blob/9b8183a/src/models/model_1_logistic_regression.py) · [Model 2](../src/models/model_2_logistic_hmm.py) · [Model 3](../src/models/model_3_xgboost.py) · [Model 4](../src/models/model_4_xgboost_hmm.py)

All four classifiers use the same scored rows, labels, feature scaling for the base predictors and unweighted training classes. They fit on 2021–2022 (34,053 rows), select hyperparameters using average precision on 2023 (16,393 rows) and report final internal test scores on 2024 (17,481 rows). The logistic grid varies inverse regularization strength. The XGBoost grid varies depth, learning rate and number of trees. Each model selects its own settings for each outcome. The paired comparisons therefore measure the effect of adding regime features under a shared tuning protocol, including any resulting change in the selected settings. The experiments do not report a same-hyperparameter sensitivity check. [Original modelling plan, §§6, 9 and 20](https://github.com/Streamrepo/Project-Data-Science-/blob/9b8183a/Modelling_Plan.md) · [Shared feature and scoring definitions](../src/models/model_234_common.py)

## B. Comparative predictive performance

Average precision (AP), labeled `pr_auc` in the CSVs, summarizes precision across recall levels. Its no-skill reference is the event prevalence in the evaluated split. Precision at 10% is the event rate among the highest-scoring tenth of intervals. Brier score and log loss measure probability accuracy, with lower values preferred. The constant forecast uses the training prevalence for every split. Recall and F1 at a fixed probability threshold, calibration curves and confidence intervals were not reported.

| Target | Model | Train ROC-AUC / AP | Validation ROC-AUC / AP | Test ROC-AUC | Test AP | Test precision at 10% | Test Brier | Test log loss |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Short | 1 | 0.7153 / 0.2065 | 0.6045 / 0.1011 | 0.5319 | 0.1135 | 0.1196 | 0.0980 | 0.3951 |
| Short | 2 | 0.7171 / 0.2083 | 0.6071 / 0.1020 | 0.5369 | 0.1152 | 0.1230 | 0.0979 | 0.3946 |
| Short | 3 | 0.7600 / 0.2743 | 0.6002 / 0.0962 | 0.5552 | 0.1233 | 0.1453 | 0.0968 | 0.3764 |
| Short | 4 | 0.7588 / 0.2709 | 0.5987 / 0.0968 | **0.5661** | **0.1284** | **0.1579** | **0.0960** | **0.3624** |
| Long | 1 | 0.7981 / 0.2984 | 0.5949 / 0.0766 | 0.5821 | 0.1022 | 0.1098 | **0.0758** | **0.2981** |
| Long | 2 | 0.7985 / 0.2987 | 0.5912 / 0.0775 | 0.5840 | 0.1032 | 0.1018 | 0.0758 | 0.2985 |
| Long | 3 | 0.8173 / 0.3318 | 0.6908 / 0.1035 | 0.6473 | **0.1429** | **0.1630** | 0.0762 | 0.3115 |
| Long | 4 | 0.8171 / 0.3314 | 0.6897 / 0.1032 | **0.6477** | 0.1426 | 0.1590 | 0.0761 | 0.3112 |

Bold identifies the best classifier within each test metric, using unrounded values. Model 1 outputs are in `results/model_1/` at commit `9b8183a` on `Lucs-Magic-Tree`. Training and validation probability scores are available in the same CSVs. [Model 2 metrics](model_2/metrics.csv) · [Model 3 metrics](model_3/metrics.csv) · [Model 4 metrics](model_4/metrics.csv)

The independent implementations of Models 2–4 were run end to end in a temporary directory containing only their four Python files and the required input CSVs from commit `9b8183a`. All 11 regenerated result files matched the previously committed outputs after normalizing line endings. A separate check verified that the forward filter agrees with inference limited to the observations through each interval and advances by the transition probabilities alone when observations are missing.

**Ranking and screening.** On 2024 short events, Model 4 leads all five classifier metrics. At a 10% review budget, about 15.8% of its flagged intervals were events, against 10.4% in the full test sample. For long events, Models 3 and 4 have nearly identical ROC-AUC, while Model 3 leads AP and precision at 10%. Its flagged intervals contain about 16.3% events, against 8.1% overall. These figures imply roughly 1.52 and 2.02 times the respective test prevalences. Most flagged intervals still fail to cross the stress threshold, and no economic value or alert cost was evaluated.

**Probability accuracy.** The constant forecast outperforms every classifier on both test Brier and test log loss. Its short scores are 0.0929 and 0.3330, and its long scores are 0.0745 and 0.2826. For example, Model 3 improves long AP from 0.1022 to 0.1429 relative to Model 1 while worsening log loss from 0.2981 to 0.3115. The models can therefore add screening value through ranking even when their raw probabilities are less accurate under proper scoring rules. Aggregate scores alone cannot identify whether the gap comes from calibration, a few severe probability errors or other changes in the score distributions.

**Time dependence.** All four models have lower ROC-AUC on each held-out year than in training. For short events, the logistic models lead validation AP (0.1011–0.1020 versus 0.0962–0.0968 for the trees), while the tree models lead on test. For long events, the tree advantage is present in both held-out years. Prevalence changes from approximately 10% in training for each target to 7.17% short and 5.29% long in validation, then to 10.36% short and 8.07% long on test. Accordingly, higher test AP than validation AP can coincide with lower ROC-AUC. Validation was also used for selection, so it is distinct from the internal test.

## C. Common findings across models

Scheduled imports and exports are prominent short-side predictors. In Model 1, their standardized coefficients are about +0.385 and −0.458, respectively. Model 2 preserves the signs and similar magnitudes. In both tree models, day-ahead price and the two scheduled exchange variables are the top three short-side features by normalized gain. For long events, day-ahead price has the largest absolute non-intercept logistic coefficient in both specifications (about +0.562 and +0.558 per training standard deviation) and the highest tree gain (about 0.38 in both models). Load forecast ranks second by tree gain for long. Coefficients express conditional log-odds associations under the logistic specification. Tree gain measures the fitted utility of splits and carries no direction or inferential uncertainty. [Model 2 coefficients](model_2/coefficients.csv) · [Model 3 importance](model_3/feature_importance.csv) · [Model 4 importance](model_4/feature_importance.csv) · [Model 1 results summary](https://github.com/Streamrepo/Project-Data-Science-/blob/9b8183a/Modelling_Plan.md#20-model-1-logistic-regressions-for-severe-stress-2026-11-23)

The recurring test pattern is stronger ranking for long than short, alongside probability scores below the constant benchmark for both sides. The notebook provides context for keeping the two outcomes separate. It reports a Pearson correlation of about −0.988 between the signed stresses and only two simultaneous threshold exceedances among 67,927 retained intervals. Their sum equals the short–long settlement-price spread, since the day-ahead price cancels. This describes the outcomes themselves. The stored aggregate metrics cannot establish whether models flag the same timestamps, miss the same extreme intervals or make similar errors during specific seasons or market episodes. See §§2 and 4–6 of `report/report.ipynb` at commit `9b8183a` for the stress distributions, threshold table, class counts and joint-event analysis. [Notebook summary](https://github.com/Streamrepo/Project-Data-Science-/blob/9b8183a/Modelling_Plan.md#21-report-notebook-2026-11-23)

## D. Main differences between models

The Model 1 to Model 3 comparison holds the inputs fixed while changing the learner. Test ROC-AUC increases by 0.0234 for short and 0.0652 for long. Test AP increases by 0.0098 and 0.0407, respectively. The long advantage on both validation and test supports the usefulness of allowing nonlinear splits and interactions in these data. The scores cannot attribute the gain to any specific threshold or interaction. The reversal in short-side ordering between validation and test shows that the advantage of the trees depends on the evaluation year.

Logistic regression offers signed, comparatively transparent coefficients, subject to the chosen encoding and a linear form on the log-odds scale. XGBoost offers richer shapes, while its recorded importances give neither effect directions nor explanations for individual alerts. HMM-enhanced models add a sequential fit and require maintaining ordered day-ahead inputs and handling gaps. The boosted models also require fitting and selecting ensembles, and Model 4 combines both computational steps. This is a comparison of workflow complexity. The repository contains no controlled benchmark for training time, inference latency or memory use. [Logistic implementation](https://github.com/Streamrepo/Project-Data-Science-/blob/9b8183a/src/models/model_1_logistic_regression.py) · [HMM implementation](../src/models/model_2_logistic_hmm.py) · [XGBoost implementation](../src/models/model_3_xgboost.py)

All models use a binary exceedance objective. A marginal breach and a very large breach each contribute one positive label, so these results cannot rank the models by sensitivity to the largest financial losses. Fixed thresholds make comparisons consistent across years but also expose changes in event prevalence and ranking. A single final test year provides limited evidence of stability across market conditions.

## E. Specific contribution and usefulness of each model

**Model 1** is the interpretable benchmark for the relationship between available fundamentals and directional stress. It identifies associations worth inspecting and provides the best long-side proper scores among the four classifiers in 2024, although the constant forecast remains better. Its short-side ROC-AUC of 0.5319 limits its use as a stand-alone screening tool.

**Model 2** tests whether temporal context adds value to a logistic specification. Its comparison with Model 1 shows small increases in AP for both sides and reduced long precision at the 10% screening budget. It is useful as an ablation of regime information within a transparent learner, with weak evidence for operational benefit.

**Model 3** establishes what can be gained from nonlinear modelling with the same observable fundamentals. It offers the strongest long-side AP and precision at 10% on the internal test. It is the more parsimonious tree specification for long-side screening under the observed results, with worse raw long probabilities than the logistic baseline.

**Model 4** tests whether temporal context helps after allowing nonlinear relationships among current inputs. It offers the strongest observed short screening results, at the cost of fitting and maintaining the HMM. Its near-tie on long ROC-AUC and marginally lower long AP make the extra workflow difficult to justify for long-side ranking from this test alone.

## F. Added value of HMM regimes

| HMM addition | 2024 target | Δ ROC-AUC | Δ AP | Δ precision at 10% | Δ Brier | Δ log loss |
|---|---|---:|---:|---:|---:|---:|
| Model 2 − Model 1 | Short | +0.0051 | +0.0017 | +0.0034 | −0.0001 | −0.0005 |
| Model 2 − Model 1 | Long | +0.0019 | +0.0010 | −0.0080 | +0.0000 | +0.0003 |
| Model 4 − Model 3 | Short | +0.0109 | +0.0051 | +0.0126 | −0.0008 | −0.0140 |
| Model 4 − Model 3 | Long | +0.0004 | −0.0002 | −0.0040 | −0.0000 | −0.0003 |

Differences use unrounded CSV values before rounding to four decimal places. A signed zero indicates a change below 0.00005 in magnitude. The positive short AP gains appear in both classifier families on test. With a budget of 1,748 intervals, Model 4 identifies 276 short events versus 254 for Model 3. For long, it identifies 278 versus 285. Dividing these counts by the test totals of 1,811 short events and 1,410 long events gives best observed recall at this budget of 15.2% for short (Model 4) and 20.2% for long (Model 3). The short result is a modest screening gain in this sample. No paired test, resampling interval or external test supports a claim of statistical significance or persistence across years. [Model 2 metrics](model_2/metrics.csv) · [Model 3 metrics](model_3/metrics.csv) · [Model 4 metrics](model_4/metrics.csv)

The shared three-state diagonal Gaussian HMM fits complete contiguous runs of the six continuous day-ahead inputs in 2021–2022. It applies forward filtering on the full UTC grid, so the probability at a delivery slot depends on the current available observation and previous ones. A gap advances the transition without an emission update. The two retained probabilities encode the three-state distribution because the probabilities sum to one. Fitted state-average day-ahead prices are about €100, €182 and €305/MWh. The middle-price state has almost zero mean solar forecast, and all three states have one-step self-transition probabilities above 0.95. These fitted values describe persistent conditions in the observable inputs. They do not identify structural shortage regimes. [Model 2 regime summary](model_2/regimes.json) · [Model 4 regime summary](model_4/regimes.json)

The HMM extracts temporal structure from variables already available to the classifiers. It can therefore add context beyond the current interval's predictors without bringing in an independent information source. The experiments cannot distinguish whether the short gain reflects latent regimes, persistence in the same variables or a useful nonlinear transformation. The small and inconsistent long-side changes suggest that this regime representation adds little for that target once the day-ahead fundamentals are available. Testing simpler lagged or rolling summaries of the same inputs would help isolate what the HMM contributes.

## G. Overall interpretation and research implications

Day-ahead fundamentals contain a measurable signal for ranking severe deviations in the French market, particularly on the long side. The strongest long gains come from using XGBoost with the existing inputs. The strongest short result comes from adding temporal regime probabilities to XGBoost, but its discrimination remains modest. For both sides, the gap between ranking and probability accuracy limits the case for using raw classifier probabilities in a financial decision rule.

The importance of scheduled exchanges on the short side is consistent with cross-border schedules carrying information about market conditions before delivery. The relationship does not establish what causes imbalance settlement stress. The day-ahead price is available at forecast time and also enters the target definition with opposite signs on the two sides. Its importance therefore combines any economic signal with a mechanical link to the quantity being predicted. The results do not identify the individual effects of forecasts, dispatch, network constraints or balancing actions.

## H. Limitations and final conclusions

**Information timing and cohort selection.** The feature lists contain day-ahead prices, forecasts, scheduled exchanges and calendar information. Same-period realized load, generation, physical flows and settlement outcomes are excluded from the model inputs. Scaling, target thresholds and HMM parameters are estimated on training data, and the HMM forward filter avoids future observations. The raw exports carry delivery timestamps but lack publication timestamps, so the original vintages and precise availability at a common forecasting cutoff cannot be independently verified. Grid alignment can consult the next timestamp to infer coverage cadence, which warrants verification against the information available at the historical forecasting cutoff. Some forecast gaps are filled forward for up to two hours. The HMM runs on the full predictor grid, but the classifier cohort also requires available realized quantities to compute ex-post forecast errors. This selection is common to all four models and can bias the evaluated population relative to the set of intervals that could be scored live. [Data preparation](https://github.com/Streamrepo/Project-Data-Science-/blob/9b8183a/src/data_downloads_and_manipulation/build_dataset.py) · [Data audit and documented coverage](https://github.com/Streamrepo/Project-Data-Science-/blob/9b8183a/Modelling_Plan.md)

**Scope of the evidence.** Comparisons on the common 2024 rows are directly interpretable within each target. Short and long have different outcome definitions and event rates, so their AP values alone do not measure relative model quality. No standalone HMM predictive benchmark, absolute-deviation classifier, external 2025 evaluation, cost-weighted outcome or interval-level comparison is available. Calibration plots, threshold-dependent decision metrics and subgroup error analyses require saved scores or a reproducible scoring run. Dependence between adjacent half-hours also means uncertainty for paired performance differences should account for time, for example through block resampling.

**Conclusion.** The logistic baseline provides the clearest account of predictive associations, XGBoost provides the largest improvement for long-side ranking, and adding the HMM yields a modest further improvement for short-side screening in 2024. No classifier beats the constant forecast on test probability scores. Follow-up work should retain the same chronological discipline, evaluate all live-scoreable intervals with known outcomes, compare HMM features with simpler temporal summaries and assess calibration and decision value using a final external period after the modelling choices are fixed.

## Reproducing Models 2–4 on Gabriel

The implementations share one local scoring module and use Model 2's HMM code in Model 4 and Model 3's XGBoost code in Model 4. The source code for Model 1 and the original data builder can stay at commit `9b8183a`. All three runs require the design matrix from that commit. Models 2 and 4 additionally require the 15 raw day-ahead CSVs from that commit. These inputs are absent from the `Gabriel` branch.

From the repository root, use a Python environment with NumPy, pandas, scikit-learn, hmmlearn and XGBoost installed. The verification used Python 3.12.10 with NumPy 2.5.3, pandas 3.0.6, scikit-learn 1.9.1, hmmlearn 0.3.3 and XGBoost 3.4.1. The commands below extract the data into a temporary directory and run the modules from the current branch. Each command writes its model's CSV or JSON outputs under `results/`.

```powershell
$inputDir = Join-Path ([System.IO.Path]::GetTempPath()) ([guid]::NewGuid().ToString())
New-Item -ItemType Directory -Path $inputDir | Out-Null
git archive --format=zip --output="$inputDir/inputs.zip" 9b8183a Data/Processed/design_matrix.csv Data/Raw/ENSTOE/day_ahead_prices_fr.csv Data/Raw/ENSTOE/load_forecast_fr.csv Data/Raw/ENSTOE/renewables_forecast_fr.csv ':(glob)Data/Raw/ENSTOE/scheduled_*.csv'
Expand-Archive -LiteralPath "$inputDir/inputs.zip" -DestinationPath $inputDir
$matrix = "$inputDir/Data/Processed/design_matrix.csv"
$raw = "$inputDir/Data/Raw/ENSTOE"
$env:OMP_NUM_THREADS = '1'
$env:OPENBLAS_NUM_THREADS = '1'
.\.venv\Scripts\python.exe -m src.models.model_2_logistic_hmm --design-matrix $matrix --raw-dir $raw
.\.venv\Scripts\python.exe -m src.models.model_3_xgboost --design-matrix $matrix
.\.venv\Scripts\python.exe -m src.models.model_4_xgboost_hmm --design-matrix $matrix --raw-dir $raw
```
