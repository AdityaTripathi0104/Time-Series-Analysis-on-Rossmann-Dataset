#  Retail Sales Time-Series Forecasting: Baseline & Model Evolution

##  1. Project Overview
This project establishes a highly optimized, temporal time-series forecasting baseline using over 1 million rows of historical daily sales data from the Rossmann retail dataset. The primary objective is to evaluate the strict mathematical limits of a traditional, local Pandas-based modeling approach. By pushing these local algorithms to their computational breaking point, we establish a baseline to justify migrating the workflow to a cloud-native, agentic architecture (Snowflake + Claude Code) for exogenous feature engineering and production-grade LLM serving.

---

##  2. Data Preprocessing Strategy
Time-series forecasting relies heavily on identifying pure, repeatable signals hidden inside chaotic data. Initial exploration revealed that daily sales data was too volatile, primarily due to zero-sales days like Sundays and holidays.

* **Strategic Resampling (Weekly `W-SUN`):** Daily data hides the trend in noise, while monthly data overly smooths the signal, erasing critical intra-month realities like paydays and week-long promotions. Weekly resampling was selected to yield a smooth baseline that retains the fast-paced cyclical heartbeat of retail operations.
* **Decomposition:** The weekly sales data was mathematically separated into its underlying **Trend**, **Seasonal**, and **Residual** components using a Hodrick-Prescott (HP) filter. This visually confirmed repeating 52-week seasonal patterns.
* **Walk-Forward Validation:** The dataset was split sequentially, withholding the final 12 weeks for Rolling Window Cross-Validation. Rather than guessing 12 weeks blindly, the model predicts one week, measures the error, absorbs the true value into its history, and re-trains to predict the next week—mimicking real-world production.

---

##  3. Statistical Testing & Validation
Predictive algorithms require mathematical proof that the data's core statistical properties do not shift unpredictably over time. 

* **Augmented Dickey-Fuller (ADF) Test:** Evaluates Stationarity. The test returned a p-value of **0.00000427**, conclusively rejecting the null hypothesis. Because the data proved to be stationary (no long-term wandering escalator trend), the integration/differencing parameter for all models was securely locked at zero (**`d = 0`**).
* **Partial Autocorrelation Function (PACF):** Measures the direct, isolated impact of a past week on the current week. This identified the AutoRegressive (**`p`**) parameter.
* **Autocorrelation Function (ACF):** Measures the compounded, cascading ripple effect of past forecasting errors over time. This identified the Moving Average (**`q`**) parameter.

---

##  4. Modeling Evolution & Performance (What, Why, How)
To prevent the models from simply memorizing the past (overfitting), all architectures were rigorously evaluated using the **Akaike Information Criterion (AIC)**, which penalizes the addition of useless variables.

### Phase 1: AutoRegressive Model — AR(2, 0, 0)
* **What it is:** A model that predicts future sales based exclusively on the actual sales momentum of the immediate past.
* **Why we used it:** To establish a pure, momentum-based baseline. While the PACF showed a massive spike at 52 weeks, forcing an AR(52) model triggers the Curse of Dimensionality—forcing the engine to ingest garbage weights for 49 irrelevant intermediate weeks. Using AIC to penalize complexity, AR(2) was selected as the most mathematically stable foundation.
* **How it works:** It looks at the actual sales from 1 week ago and 2 weeks ago, assigns them optimized mathematical weights, and projects the line forward.
* **The Improvement:** This established our fundamental baseline, yielding a Final Rolling MSE of **17,465,839**.

### Phase 2: AutoRegressive Moving Average — ARMA(2, 0, 1)
* **What it is:** Upgrades the AR engine by adding the Moving Average (MA) component, which introduces an error-correction feedback loop.
* **Why we used it:** AR models blindly trust momentum. If the AR model overshoots reality, it doesn't learn from the mistake. The MA component forces the algorithm to analyze its own past residuals (errors) to adjust the current prediction.
* **How it works:** Alongside looking at the past 2 weeks of actual sales (`p=2`), it looks at the prediction error from 1 week ago (`q=1`). If it over-predicted by \$2,000 last week, it calculates a negative weight to steer today's prediction back on course.
* **The Improvement:** By learning from its mistakes rather than blindly trusting momentum, the model cut the error nearly in half. The MSE dropped to **8,901,762** (a **~49% improvement** over the baseline).

### Phase 3: Seasonal ARIMA — SARIMA(2, 0, 1) x (1, 0, 0, 52)
* **What it is:** Injects a 52-week seasonal calendar into the ARMA engine.
* **Why we used it:** The ARMA engine is "calendar-blind." It reacts well to short-term errors, but it cannot anticipate massive, recurring annual spikes (like the December holiday rush) until after it makes a massive error.
* **How it works:** It retains the ARMA(2, 0, 1) setup but adds a seasonal lookback (`P=1`, `s=52`). It calculates what the momentum is today, but then cross-references what happened exactly 52 weeks ago to anticipate seasonal surges.
* **The Improvement:** By anticipating spikes rather than reacting to them, the engine dropped the MSE to **5,573,085** (a further **~37% improvement** over the ARMA model).

---

## 5. Core Limitations & Bottlenecks

While the SARIMA model represents the mathematical ceiling of temporal-only forecasting, it hit two critical walls that prevent it from being a true enterprise solution:

### Limitation 1: The Exogenous Ceiling (Context-Blindness)
A calendar alone operates in a temporal vacuum. The SARIMA model knows that week 48 is historically busy, but it does not know if the store is throwing a massive 50%-off promotion *this* week, or if a local school holiday is driving unexpected foot traffic. Without external (exogenous) variables, the model cannot predict non-cyclical, localized revenue spikes. 

### Limitation 2: The Compute Bottleneck (Curse of Dimensionality)
The transition to SARIMA exposed the severe computational limits of a local Python/Pandas environment. To calculate a 52-week seasonal parameter, the CPU must hold an entire year of data points in active memory to compute massive inversion matrices just to step forward one single week. Running Walk-Forward Validation with millions of rows on a local machine causes the Pandas and Statsmodels engines to hang or crash entirely.

---

##  6. Strategic Next Steps: The Agentic Stack

To solve these limitations, this workflow has reached the end of its local lifecycle. The next phase of this project migrates the forecasting logic into a modern, cloud-native architecture:

1. **Agentic SQL Pipelines:** Migrating the raw data to **Snowflake** and leveraging **Claude Code (MCP)** to act as an autonomous data engineer. Claude will generate the complex exogenous feature engineering (Promos, State Holidays) directly inside the warehouse via optimized SQL pushdown, entirely bypassing local compute bottlenecks.
2. **Production Serving:** Transitioning the resulting insights into a highly scalable LLM serving architecture utilizing **vLLM, FastAPI, Redis (for caching), and Kubernetes**. 

*(Note: The MLOps migration, architectural diagrams, and infrastructure code will be detailed in a separate repository under this overarching project.)*
