# Retail Sales Time-Series Forecasting: Baseline & Model Evolution

## 1. Project Overview
This project establishes a highly optimized, temporal time-series forecasting baseline using over 1 million rows of historical daily sales data from the Rossmann retail dataset. The primary objective is to evaluate the strict mathematical limits of a traditional, local Pandas-based modeling approach. By pushing these local algorithms to their computational breaking point, we establish a baseline to justify migrating the workflow to a cloud-native, agentic architecture (Snowflake + Claude Code) for exogenous feature engineering and production-grade LLM serving.

---

## 2. Data Preprocessing Strategy
Time-series forecasting relies heavily on identifying pure, repeatable signals hidden inside chaotic data. Initial exploration revealed that daily sales data was too volatile, primarily due to zero-sales days like Sundays and holidays.

*   **Strategic Resampling (Weekly W-SUN):** Daily data hides the trend in noise, while monthly data overly smooths the signal, erasing critical intra-month realities like paydays and week-long promotions. Weekly resampling was selected to yield a smooth baseline that retains the fast-paced cyclical heartbeat of retail operations.
*   **Decomposition:** The weekly sales data was mathematically separated into its underlying **Trend**, **Seasonal**, and **Residual** components using a Hodrick-Prescott (HP) filter. This visually confirmed repeating 52-week seasonal patterns.
*   **Walk-Forward Validation:** The dataset was split sequentially, withholding the final 12 weeks for Rolling Window Cross-Validation. Rather than guessing 12 weeks blindly, the model predicts one week, measures the error, absorbs the true value into its history, and re-trains to predict the next week—mimicking real-world production.

---

## 3. Statistical Testing & Validation
Predictive algorithms require mathematical proof that the data's core statistical properties do not shift unpredictably over time. 

*   **Augmented Dickey-Fuller (ADF) Test:** Evaluates Stationarity. The test returned a p-value of **0.00000427**, conclusively rejecting the null hypothesis. Because the data proved to be stationary (no long-term wandering escalator trend), the integration/differencing parameter for all models was securely locked at zero (**`d = 0`**).
*   **Partial Autocorrelation Function (PACF):** Measures the direct, isolated impact of a past week on the current week. This identified the AutoRegressive (**`p`**) parameter.
*   **Autocorrelation Function (ACF):** Measures the compounded, cascading ripple effect of past forecasting errors over time. This identified the Moving Average (**`q`**) parameter.

---

## 4. Modeling Evolution & Performance

To prevent the models from simply memorizing the past (overfitting), all architectures were rigorously evaluated using the **Akaike Information Criterion (AIC)**, which penalizes the addition of useless variables. 

### Phase 1: The Baseline Momentum Engine — AR(2)
*   **What it is:** An AutoRegressive model that predicts future sales based exclusively on the actual sales numbers of the immediate past.
*   **Why we used it:** To establish a pure, momentum-based baseline. While the PACF showed a massive correlation spike at 52 weeks, forcing an AR(52) model triggers the Curse of Dimensionality—forcing the engine to ingest garbage weights for 49 irrelevant intermediate weeks. Using AIC to penalize complexity, AR(2) was selected as the most mathematically stable foundation.
*   **How it works:** It looks at the actual sales from 1 week ago and 2 weeks ago, assigns them optimized mathematical weights, and projects the line forward.
*   **The Improvement:** Established our fundamental baseline benchmark, yielding a Final Rolling MSE of **17,465,839**.
*   **The Limitation:** Blind trust in momentum. If the model makes a massive prediction error, it has no mathematical mechanism to realize it was wrong or correct itself in the next step.

### Phase 2: The Error-Correcting Engine — ARMA(2, 1)
*   **What it is:** Upgrades the AR engine by adding the Moving Average (MA) component, introducing an active error-correction feedback loop.
*   **Why we used it:** AR models do not learn from their mistakes. The MA component forces the algorithm to analyze its own past residuals (errors) to adjust the current prediction.
*   **How it works:** Alongside looking at the past 2 weeks of actual sales (`p=2`), it looks at the prediction error from exactly 1 week ago (`q=1`). If the model over-predicted by \$2,000 last week, it calculates a negative weight to steer today's prediction back down on course.
*   **The Improvement:** By learning from its mistakes rather than blindly trusting momentum, the model cut the error nearly in half. The MSE dropped to **8,901,762** (a **~49% improvement** over the baseline).
*   **The Limitation:** It assumes the overarching trend is completely stable. If the store begins a long-term growth phase (a macro trend), the ARMA model will consistently under-predict because it cannot adjust its baseline.

### Phase 3: The Trend-Flattening Engine — ARIMA(2, 0, 1)
*   **What it is:** AutoRegressive Integrated Moving Average. It introduces the "Integrated" (`I`) component, which utilizes differencing to flatten macro trends.
*   **Why we used it:** Real-world retail data often features long-term macro trends (like inflation or year-over-year store growth) that confuse standard ARMA models. 
*   **How it works:** It utilizes the differencing parameter (`d`) to look at the *change* between weeks rather than the raw sales numbers. However, because our initial ADF test conclusively proved our specific dataset was already stationary (p-value = 0.00000427), the differencing parameter was locked at zero (`d=0`). 
*   **The Improvement:** Conceptually, ARIMA prepares the architecture to handle non-stationary data. In this specific stationary dataset, the engine operates identically to ARMA, holding the MSE at **8,901,762**.
*   **The Limitation:** It is completely "calendar-blind." While it can handle momentum, errors, and trends, it cannot anticipate massive, recurring annual spikes (like the December holiday rush) until *after* the spike has occurred and generated a massive error.

### Phase 4: The Calendar-Aware Engine — SARIMA(2, 0, 1) x (1, 0, 0, 52)
*   **What it is:** Seasonal ARIMA. It injects a strict 52-week seasonal calendar directly into the ARIMA engine.
*   **Why we used it:** To cure the calendar-blindness of the ARIMA model. SARIMA allows the algorithm to mathematically anticipate localized foot-traffic events and repeating seasonal behavior rather than just reacting to the resulting errors.
*   **How it works:** It retains the ARIMA(2, 0, 1) setup but adds a seasonal lookback (`P=1`, `s=52`). It calculates the immediate momentum and errors today, but then cross-references what happened exactly 52 weeks ago to anticipate seasonal surges.
*   **The Improvement:** By anticipating spikes rather than reacting to them, the engine dropped the MSE to **5,573,085** (a further **~37% improvement** over the ARIMA model).
*   **The Limitation:** While highly accurate, calculating a 52-week seasonal inversion matrix on a rolling window demands massive local CPU compute and memory. Running Walk-Forward Validation with millions of rows causes standard local Pandas environments to hang. Furthermore, the model remains "context-blind"—it knows week 48 is historically busy, but it does not know if the store is throwing a massive 50%-off promotion *this* specific week.

---

## 5. Strategic Next Steps: The Agentic Stack

The SARIMA model represents the absolute mathematical ceiling of temporal-only forecasting in a local environment. To break past this ceiling and capture localized, non-cyclical events (like massive promotions or state holidays), the workflow will migrate to an **Agentic MLOps Architecture**. 

*   **Agentic SQL Pipelines:** Migrating the dataset to **Snowflake** and using **Claude Code (MCP)** to automate heavy, exogenous feature engineering via optimized SQL pushdown, entirely bypassing local compute bottlenecks.
*   **Production Serving:** Transitioning the predictive insights into a production-ready LLM serving stack using **vLLM, FastAPI, Redis, and Kubernetes**. 

*(Note: This migration and the accompanying production-grade LLM serving stack will be covered in a separate repository under a broader project, detailing the required tool stack, methodologies, and architectural decisions.)*
