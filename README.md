# Retail Sales Time-Series Forecasting: Baseline & Model Evolution

## Project Overview
This project establishes a highly optimized, temporal time-series forecasting baseline using over 1 million rows of historical daily sales data from the Rossmann retail dataset. The primary objective is to evaluate the mathematical limits of a traditional, local Pandas-based modeling approach before migrating the workflow to a cloud-native, agentic architecture (Snowflake + Claude Code) for exogenous feature engineering and production-grade LLM serving.

## Data Preprocessing & Strategy
Retail sales data is inherently volatile. Initial data exploration revealed that analyzing sales on a daily level introduced significant noise, primarily due to zero-sales days (Sundays and holidays). To isolate the true signal, the following steps were taken:

*   **Resampling:** The dataset was downsampled from a daily to a weekly frequency (**W-SUN**). This smoothed out artificial daily drops while maintaining the fast-paced cyclical heartbeat of retail operations. Monthly resampling was rejected as it overly smoothed the data, blinding the models to intra-month realities like paydays and short-term promotions.
*   **Decomposition:** The weekly sales data was mathematically separated into its underlying **Trend** and **Cyclic** components using a Hodrick-Prescott (HP) filter to visually confirm repeating 52-week seasonal patterns.
*   **Train/Test Split:** The dataset was split sequentially, withholding the final 12 weeks of data as a blind test set for Walk-Forward (Rolling Window) Cross-Validation.

## Statistical Testing & Validation
Before modeling, the dataset's statistical properties were rigorously tested to determine the required parameters for the ARIMA-family models:

*   **Augmented Dickey-Fuller (ADF) Test:** Used to prove stationarity (a constant mean and variance). The ADF test returned a p-value of **0.00000427**, conclusively rejecting the null hypothesis. Because the data lacked a wandering long-term trend, the differencing parameter was locked at zero (**d = 0**).
*   **Partial Autocorrelation Function (PACF):** Used to identify the AutoRegressive (**p**) parameter by isolating the direct impact of past weeks on the present, stripping out intermediate noise. Mathematically significant lags were found at 2, 12, 24, 39, 40, and 52 weeks.
*   **Autocorrelation Function (ACF):** Used to identify the Moving Average (**q**) parameter by measuring the total, compounded correlation of past forecasting errors rippling through the timeline.

## Modeling Evolution & Performance

### 1. AutoRegressive Model: AR(2, 0, 0)
*   **Concept:** Predicts future sales based exclusively on the actual sales momentum of the immediate past.
*   **Evaluation & Parameter Selection:** While the PACF showed a massive spike at lag 52, forcing an AR(52) model causes a dimensionality collapse and requires the model to ingest garbage weights for irrelevant intermediate weeks. Using the Akaike Information Criterion (AIC) to penalize complexity, **AR(2)** was selected as the mathematically superior, stable baseline.
*   **Performance:** Final Rolling MSE of **17,465,839**.
*   **Limitation:** AR models only look at past actuals and are completely blind to the model's own past prediction mistakes.

### 2. AutoRegressive Moving Average: ARMA(2, 0, 1)
*   **Concept:** Upgrades the AR engine by adding the Moving Average (MA) component. It uses the **q** parameter to calculate weights for past residuals (errors). If the model over-predicted last week, the MA component actively adjusts the steering wheel to correct today's prediction.
*   **Performance:** Final Rolling MSE of **8,901,762**.
*   **Limitation:** While highly effective at correcting short-term errors, the ARMA engine has no concept of a calendar. It cannot anticipate massive, recurring annual spikes (like the holiday season) until after the error has already occurred.

### 3. Seasonal ARIMA: SARIMA(2, 0, 1) x (1, 0, 0, 52)
*   **Concept:** Injects a 52-week seasonal calendar into the ARMA engine. This allows the algorithm to look back exactly one year to anticipate massive, localized foot-traffic events and seasonal behavior.
*   **Performance:** Final Rolling MSE of **5,573,085**.
*   **Limitation (The Compute Bottleneck):** While highly accurate, calculating a 52-week seasonal inversion matrix on a rolling window demands massive local CPU compute and memory. This traditional Python/Pandas approach hits a hard scalability wall in production environments.

## Strategic Next Steps
The SARIMA model represents the absolute mathematical ceiling of temporal-only forecasting in a local environment. To break past this ceiling and capture localized, non-cyclical events (like massive 50%-off promotions), the workflow will migrate to an **Agentic MLOps Architecture**. 

By utilizing **Claude Code** integrated with **Snowflake**, complex exogenous feature engineering will be automated via optimized SQL pushdown, bypassing local memory limits and preparing the pipeline for deployment behind a **vLLM** and **FastAPI** serving layer. 

*Note: This migration and the accompanying production-grade LLM serving stack will be covered in a separate repository under a broader project, detailing the required tool stack, methodologies, and architectural decisions.*
