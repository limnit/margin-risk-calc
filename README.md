# margin-risk-calc
computes volatility ranges for US listed equities and generates a CSV for risk systems
# Risk Assessment and Margin Calculation System for Equities and Indexes

**Author:** George Kledaras 
**Date:** October 11, 2024

---

## Table of Contents

1. [Abstract](#abstract)
2. [Introduction](#1-introduction)
3. [Regulatory Compliance](#2-regulatory-compliance)
   - 3.1. [Financial Regulations](#21-financial-regulations)
   - 3.2. [Margin Requirements per FINRA Rule 4210](#22-margin-requirements-per-finra-rule-4210)
4. [System Overview](#3-system-overview)
5. [Data Acquisition](#4-data-acquisition)
   - 5.1. [Ticker Classification](#41-ticker-classification)
   - 5.2. [Historical Data Retrieval](#42-historical-data-retrieval)
     - 5.2.1. [API Integration](#421-api-integration)
6. [Risk Metrics Calculation](#5-risk-metrics-calculation)
   - 6.1. [Volatility Calculation](#51-volatility-calculation)
     - 6.1.1. [Total Volatility](#511-total-volatility)
     - 6.1.2. [Downside and Upside Volatility](#512-downside-and-upside-volatility)
   - 6.2. [Value at Risk (VaR)](#52-value-at-risk-var)
   - 6.3. [Beta Calculation](#53-beta-calculation)
   - 6.4. [Kurtosis](#54-kurtosis)
7. [Margin Requirements](#6-margin-requirements)
   - 7.1. [Industry Standards](#61-industry-standards)
   - 7.2. [Margin Calculation Methodology](#62-margin-calculation-methodology)
     - 7.2.1. [Margin for Stocks](#621-margin-for-stocks)
     - 7.2.2. [Margin for Indexes](#622-margin-for-indexes)
     - 7.2.3. [Margin Caps and Floors](#623-margin-caps-and-floors)
     - 7.2.4. [Handling Low-Priced Stocks](#624-handling-low-priced-stocks)
8. [System Implementation](#7-system-implementation)
   - 8.1. [Technology Stack](#71-technology-stack)
   - 8.2. [Script Structure and Workflow](#72-script-structure-and-workflow)
     - 8.2.1. [Initialization](#721-initialization)
     - 8.2.2. [Data Acquisition](#722-data-acquisition)
     - 8.2.3. [Risk Calculation](#723-risk-calculation)
     - 8.2.4. [Data Storage and Update](#724-data-storage-and-update)
     - 8.2.5. [Output Generation](#725-output-generation)
   - 8.3. [Multithreading and Performance Optimization](#73-multithreading-and-performance-optimization)
9. [Usage Instructions](#8-usage-instructions)
   - 9.1. [Prerequisites](#81-prerequisites)
   - 9.2. [Installation](#82-installation)
   - 9.3. [Configuration](#83-configuration)
   - 9.4. [Execution](#84-execution)
   - 9.5. [Monitoring and Logs](#85-monitoring-and-logs)
   - 9.6. [Output](#86-output)
10. [Compliance and Regulatory Considerations](#9-compliance-and-regulatory-considerations)
    - 10.1. [Alignment with FINRA and SEC Regulations](#91-alignment-with-finra-and-sec-regulations)
    - 10.2. [Data Accuracy and Integrity](#92-data-accuracy-and-integrity)
    - 10.3. [Auditability](#93-auditability)
    - 10.4. [Limitations and Assumptions](#94-limitations-and-assumptions)
11. [Conclusion](#10-conclusion)
12. [References](#11-references)
13. [Appendices](#appendices)
    - A. [Mathematical Formulations](#appendix-a-mathematical-formulations)
    - B. [Code Snippets](#appendix-b-code-snippets)
    - C. [Sample Output](#appendix-c-sample-output)
    - D. [Limitations and Future Enhancements](#appendix-d-limitations-and-future-enhancements)

---

## Abstract

This white paper presents a comprehensive risk assessment and margin calculation system tailored for equities and indexes, designed specifically for risk managers. The system automates the collection of historical market data, calculates key risk metrics, and determines margin requirements based on industry standards and regulatory guidelines. By leveraging advanced statistical methods and adhering to regulatory compliance, the system aids risk managers in effectively assessing and managing financial risk.

---

## 1. Introduction

In today's dynamic financial markets, risk managers face the challenge of accurately assessing risk and determining appropriate margin requirements to safeguard against potential losses. The complexities of market movements necessitate a robust system that can automate data collection, perform sophisticated risk calculations, and comply with regulatory standards.

This paper details the development and implementation of a risk assessment and margin calculation system that addresses these needs. The system is built using Python and integrates with the Polygon.io API to fetch market data, employing advanced statistical techniques to compute risk metrics.

---

## 2. Regulatory Compliance

### 2.1. Financial Regulations

The system aligns with regulations set by key financial authorities, including:

- **Financial Industry Regulatory Authority (FINRA)**
- **Securities and Exchange Commission (SEC)**

These regulations mandate specific margin requirements and risk management practices to ensure market stability and protect investors.

### 2.2. Margin Requirements per FINRA Rule 4210

According to FINRA Rule 4210, margin requirements for securities are determined based on the type of security and its associated risk. The system adheres to these guidelines by:

- Implementing minimum margin percentages.
- Adjusting margin requirements based on volatility.
- Differentiating between long and short positions.

---

## 3. System Overview

The system encompasses the following components:

1. **Data Acquisition Module**
2. **Risk Metrics Calculation Engine**
3. **Margin Calculation Module**
4. **Reporting and Output Generation**

The architecture ensures modularity, allowing for easy updates and maintenance.

---

## 4. Data Acquisition

### 4.1. Ticker Classification

Tickers are classified into:

- **Stocks (Common Stock)**
- **Broad Indexes**
- **Sector Indexes**

Classification is essential for applying appropriate margin requirements and is based on data from the Polygon.io API.

### 4.2. Historical Data Retrieval

The system fetches daily closing prices for each ticker over the past 365 days (approximately 252 trading days). Data retrieval is optimized using multithreading to handle I/O-bound operations efficiently.

#### 4.2.1. API Integration

- **API Used:** Polygon.io Market Data API
- **Endpoints Accessed:**
  - `/v3/reference/tickers` for ticker classification.
  - `/v2/aggs/ticker/{ticker}/range/1/day/{start}/{end}` for historical prices.

---

## 5. Risk Metrics Calculation

The system calculates several key risk metrics:

- **Volatility (Total, Downside, Upside)**
- **Value at Risk (VaR)**
- **Beta**
- **Kurtosis**

### 5.1. Volatility Calculation

#### 5.1.1. Total Volatility

Total volatility ($\sigma_{\text{total}}$) is calculated using the weighted standard deviation of daily returns:

$$
\sigma_{\text{total}} = \sqrt{252} \times \sqrt{\frac{\sum_{i=1}^{n} w_i (r_i - \mu)^2}{\sum_{i=1}^{n} w_i}}
$$

- $r_i$: Daily return on day $i$.
- $\mu$: Weighted average return.
- $w_i$: Weight for day $i$ (recent days may have higher weights).
- $n$: Number of trading days (up to 252).

#### 5.1.2. Downside and Upside Volatility

- **Downside Volatility ($\sigma_{\text{down}}$):** Focuses on negative returns (used for long positions).
- **Upside Volatility ($\sigma_{\text{up}}$):** Focuses on positive returns (used for short positions).

Calculations are similar to total volatility but consider only negative or positive returns:

$$
\sigma_{\text{down}} = \sqrt{252} \times \sqrt{\frac{\sum_{i \in \{r_i < 0\}} w_i (r_i - \mu_{\text{down}})^2}{\sum_{i \in \{r_i < 0\}} w_i}}
$$

$$
\sigma_{\text{up}} = \sqrt{252} \times \sqrt{\frac{\sum_{i \in \{r_i > 0\}} w_i (r_i - \mu_{\text{up}})^2}{\sum_{i \in \{r_i > 0\}} w_i}}
$$

### 5.2. Value at Risk (VaR)

VaR estimates the maximum expected loss over a given time horizon at a specific confidence level ($\alpha$):

$$
\text{VaR} = - \left( \mu + z_{\alpha} \times \frac{\sigma_{\text{total}}}{\sqrt{252}} \right) \times \sqrt{252}
$$

- $z_{\alpha}$: Z-score corresponding to confidence level $\alpha$ (e.g., 1.645 for 95% confidence).

### 5.3. Beta Calculation

Beta ($\beta$) measures the sensitivity of a security's returns to market movements:

$$
\beta = \frac{\text{Cov}(r_{\text{security}}, r_{\text{market}})}{\text{Var}(r_{\text{market}})}
$$

- $r_{\text{security}}$: Returns of the security.
- $r_{\text{market}}$: Returns of the market index (e.g., SPY).
- **Covariance** and **Variance** are calculated over the same time period.

### 5.4. Kurtosis

Kurtosis measures the "tailedness" of the return distribution, indicating the presence of extreme values:

$$
\text{Kurtosis} = \frac{n \sum_{i=1}^{n} (r_i - \mu)^4}{\left( \sum_{i=1}^{n} (r_i - \mu)^2 \right)^2}
$$

- A higher kurtosis implies a higher probability of extreme returns.

---

## 6. Margin Requirements

### 6.1. Industry Standards

The system's margin requirements are based on:

- **FINRA Rule 4210**
- **Industry Best Practices**

### 6.2. Margin Calculation Methodology

### 6.2. Margin Calculation Methodology

The margin calculation methodology has been updated to account for highly volatile stocks that may require margin requirements exceeding 100%. This ensures that the system accurately reflects the increased risk associated with such securities.

#### 6.2.1. Margin for Stocks

**Long Positions:**

For long positions, the margin requirement is calculated as:

$$
\text{Margin}_{\text{long}} = \min\left( \text{MinMargin}_{\text{long}} + \left( \text{MaxMargin} - \text{MinMargin}_{\text{long}} \right) \times \left( \frac{\sigma_{\text{down}}}{\sigma_{\text{threshold}}} \right), \text{Margin}_{\text{cap}} \right)
$$

- \( \text{MinMargin}_{\text{long}} = 15\% \)
- \( \text{MaxMargin} = 150\% \) (allowing margin requirements up to 150%)
- \( \sigma_{\text{down}} \): Downside volatility
- \( \sigma_{\text{threshold}} = 100\% \): Volatility threshold for scaling margin requirements
- \( \text{Margin}_{\text{cap}} = 150\% \): Maximum allowable margin requirement

**Short Positions:**

For short positions, the margin requirement is calculated as:

$$
\text{Margin}_{\text{short}} = \min\left( \text{MinMargin}_{\text{short}} + \left( \text{MaxMargin} - \text{MinMargin}_{\text{short}} \right) \times \left( \frac{\sigma_{\text{up}}}{\sigma_{\text{threshold}}} \right), \text{Margin}_{\text{cap}} \right)
$$

- \( \text{MinMargin}_{\text{short}} = 20\% \)
- Other variables are as defined above.

#### 6.2.3. Margin Caps and Floors

- **Maximum Margin (\( \text{MaxMargin} \)):** Increased to **150%** to account for highly volatile stocks.
- **Margin Cap (\( \text{Margin}_{\text{cap}} \)):** Set at **150%**, representing the maximum margin requirement.
- **Minimum Margin:** As specified per position type and security classification.

#### 6.2.4. Handling Highly Volatile and Low-Priced Stocks

- **Stocks Priced Below \$1.00:** Fixed margin requirement of **100%** (trade on a cash basis).
- **Highly Volatile Stocks (\( \sigma > 100\% \)):** Margin requirements may exceed **100%**, up to the maximum of **150%**.


---

## 7. System Implementation

### 7.1. Technology Stack

- **Programming Language:** Python 3.x
- **Data Libraries:** Pandas, NumPy
- **Statistical Libraries:** SciPy
- **Database:** SQLite (managed via SQLAlchemy ORM)
- **API Integration:** Requests library for HTTP calls to Polygon.io API
- **Concurrency:** `concurrent.futures.ThreadPoolExecutor` for multithreading

### 7.2. Script Structure and Workflow

#### 7.2.1. Initialization

- **Database Setup:** Initializes SQLite database with appropriate schema.
- **API Key Configuration:** Sets up the Polygon.io API key for authentication.

#### 7.2.2. Data Acquisition

- **Ticker Fetching:** Retrieves all active tickers with classifications.
- **Historical Data Fetching:** Downloads historical price data for each ticker.

#### 7.2.3. Risk Calculation

- **Data Preparation:** Calculates daily returns and applies weights.
- **Metric Computation:** Calculates volatility, VaR, beta, and kurtosis.
- **Margin Calculation:** Determines margin requirements based on calculated volatilities and classifications.

#### 7.2.4. Data Storage and Update

- **Initial Data Storage:** Stores historical data in the database.
- **Daily Updates:** Updates the database with new data as it becomes available.

#### 7.2.5. Output Generation

- **Risk Profile Report:** Generates a CSV file containing all calculated metrics and margin requirements.

### 7.3. Multithreading and Performance Optimization

- **Concurrency Management:** Uses a semaphore to limit the number of concurrent API calls, respecting rate limits.
- **Thread Safety:** Ensures database operations are thread-safe by using SQLAlchemy's connection pooling.
- **Error Handling:** Implements retry logic and exception handling for network and API errors.

---

## 8. Usage Instructions

### 8.1. Prerequisites

- **Python Environment:** Ensure Python 3.x is installed.
- **API Key:** Obtain a Polygon.io API key with sufficient privileges.

### 8.2. Installation

Install the required Python libraries:

```bash
pip install pandas numpy requests scipy sqlalchemy
```
#### 8.2.1. Initialization

- **Database Setup:** Initializes SQLite database with appropriate schema.
- **API Key Configuration:** Sets up the Polygon.io API key for authentication.

#### 8.2.2. Data Acquisition

- **Ticker Fetching:** Retrieves all active tickers with classifications.
- **Historical Data Fetching:** Downloads historical price data for each ticker.

#### 8.2.3. Risk Calculation

- **Data Preparation:** Calculates daily returns and applies weights.
- **Metric Computation:** Calculates volatility, VaR, beta, and kurtosis.
- **Margin Calculation:** Determines margin requirements based on calculated volatilities and classifications.

#### 8.2.4. Data Storage and Update

- **Initial Data Storage:** Stores historical data in the database.
- **Daily Updates:** Updates the database with new data as it becomes available.

#### 8.2.5. Output Generation

- **Risk Profile Report:** Generates a CSV file containing all calculated metrics and margin requirements.

### 8.3. Multithreading and Performance Optimization

- **Concurrency Management:** Uses a semaphore to limit the number of concurrent API calls, respecting rate limits.
- **Thread Safety:** Ensures database operations are thread-safe by using SQLAlchemy's connection pooling.
- **Error Handling:** Implements retry logic and exception handling for network and API errors.

---

## 9. Usage Instructions

### 9.1. Prerequisites

- **Python Environment:** Ensure Python 3.x is installed.
- **API Key:** Obtain a Polygon.io API key with sufficient privileges.

### 9.2. Installation

Install the required Python libraries:

```bash
pip install pandas numpy requests scipy sqlalchemy
```
### 9.3. Configuration
Set your Polygon.io API key in the script:
```bash
API_KEY = 'YOUR_API_KEY'
```

### 9.4. Execution
Run the script:
```bash
python risk_assessment.py
```

### 9.5. Monitoring and Logs

- **Progress Indicators:** The script outputs progress messages to the console.
- **Error Logs:** Any errors encountered are printed to the console.

### 9.6. Output

- **Risk Profile Report:** Generated as `daily_risk_profiles.csv` in the script's directory.
- **Data Included:** Ticker, Classification, Latest Price, Volatilities, VaR, Beta, Kurtosis, Margin Percentages.

---

## 10. Compliance and Regulatory Considerations

### 10.1. Alignment with FINRA and SEC Regulations

- **Margin Requirements:** The system now accommodates higher margin requirements for highly volatile stocks, which may exceed 100%, aligning with FINRA Rule 4210 and brokerage industry practices.

### 10.2. Data Accuracy and Integrity

- **Reliable Data Source:** Uses Polygon.io, a reputable market data provider.
- **Data Validation:** Includes checks for data completeness and consistency.

### 10.3. Auditability

- **Transparency:** Detailed metrics allow for easy verification and audit.
- **Reproducibility:** Calculations can be replicated using the provided data and methodology.

### 10.4. Limitations and Assumptions

- **Historical Data Limitations:** Assumes past market behavior is indicative of future risk.
- **Normal Distribution Assumption:** VaR calculations assume returns are normally distributed.

---

## 11. Conclusion

The risk assessment and margin calculation system provides risk managers with a powerful tool to evaluate financial risk accurately. By automating data collection and employing advanced statistical methods, the system enhances the efficiency and effectiveness of risk management processes. Its compliance with regulatory standards ensures that organizations can confidently rely on it to meet their regulatory obligations.

---

## 12. References

1. **FINRA Margin Requirements:**
   - [FINRA Rule 4210](https://www.finra.org/rules-guidance/rulebooks/finra-rules/4210)

2. **Securities and Exchange Commission (SEC):**
   - [SEC Regulations](https://www.sec.gov/regulation)

3. **Polygon.io API Documentation:**
   - [Polygon.io API Docs](https://polygon.io/docs/)

4. **Python Libraries:**
   - [Pandas Documentation](https://pandas.pydata.org/docs/)
   - [NumPy Documentation](https://numpy.org/doc/)
   - [SciPy Documentation](https://docs.scipy.org/doc/)
   - [SQLAlchemy Documentation](https://docs.sqlalchemy.org/)

---

## 13. Appendices

### Appendix A: Mathematical Formulations

#### A.1. Weighted Standard Deviation

The weighted standard deviation is calculated as:

$$
\sigma = \sqrt{\frac{\sum_{i=1}^{n} w_i\ (x_i - \mu)^2}{\sum_{i=1}^{n} w_i}}
$$

- \( x_i \): Data point \( i \).
- \( w_i \): Weight for data point \( i \).
- \( \mu \): Weighted mean.

#### A.2. Z-Score for VaR

The Z-score corresponds to the inverse of the cumulative distribution function (CDF) of the standard normal distribution:

$$
z_\alpha = \Phi^{-1}(1 - \alpha)
$$

- \( \Phi^{-1} \): Inverse CDF (quantile function).
- \( \alpha \): Confidence level (e.g., 0.95 for 95%).

---

### Appendix B: Code Snippets

#### B.1. Volatility Calculation Code

```python
def calculate_volatilities(returns, weights):
    # Total Volatility
    total_volatility = weighted_std(returns, weights) * np.sqrt(252)

    # Separate positive and negative returns
    positive_mask = returns > 0
    negative_mask = returns < 0
    positive_returns = returns[positive_mask]
    negative_returns = returns[negative_mask]
    positive_weights = weights[positive_mask]
    negative_weights = weights[negative_mask]

    # Upside Volatility
    if len(positive_returns) > 0:
        upside_volatility = weighted_std(positive_returns, positive_weights) * np.sqrt(252)
    else:
        upside_volatility = 0.0

    # Downside Volatility
    if len(negative_returns) > 0:
        downside_volatility = weighted_std(negative_returns, negative_weights) * np.sqrt(252)
    else:
        downside_volatility = 0.0

    return total_volatility, downside_volatility, upside_volatility
```

#### B.2. Margin Calculation Code

```python
def get_margin_percentage(volatility, price, classification, position):
    # Define margin parameters
    min_margin_long = 0.15
    min_margin_short = 0.20
    max_margin = 1.50  # Allow margin requirements up to 150%
    margin_cap = 1.50  # Maximum allowable margin requirement
    volatility_threshold = 1.00  # 100% volatility threshold

    if classification == 'Broad Index':
        return 0.08  # 8% fixed margin
    elif classification == 'Sector Index':
        return 0.10  # 10% fixed margin
    elif classification == 'Stock':
        if price < 1.0:
            return 1.00  # 100% margin requirement for low-priced stocks
        else:
            if position == 'long':
                margin = min_margin_long + (max_margin - min_margin_long) * (volatility / volatility_threshold)
            elif position == 'short':
                margin = min_margin_short + (max_margin - min_margin_short) * (volatility / volatility_threshold)
            else:
                margin = max_margin  # Default to maximum margin

            # Ensure margin does not exceed margin_cap
            margin_percentage = min(margin, margin_cap)
            return margin_percentage
    else:
        return 1.00  # Default margin requirement

```
---
### Appendix C: Sample Output

| Ticker | Classification | Latest Price | Total Volatility | Downside Volatility | Upside Volatility | Kurtosis |   VaR   |  Beta  | Long Margin % | Short Margin % |
|--------|----------------|--------------|------------------|---------------------|-------------------|----------|---------|--------|----------------|-----------------|
| AAPL   | Stock          | \$150.00     | 25.00%           | 15.00%              | 10.00%            |   3.50   | -5.00%  |  1.20  |     26.25%     |     22.00%      |
| SPY    | Broad Index    | \$430.00     | 15.00%           | 8.00%               | 7.00%             |   2.80   | -4.00%  |  1.00  |     8.00%      |     8.00%       |
| XYZ    | Stock          | \$0.80       | 40.00%           | 30.00%              | 10.00%            |   4.20   | -8.00%  |  1.50  |    100.00%     |    100.00%      |
---
### Appendix D: Limitations and Future Enhancements

#### D.1. Limitations

- **Data Quality:** Relies on the accuracy of data provided by Polygon.io.
- **Model Assumptions:** Assumes normal distribution of returns, which may not capture extreme market events.

### Appendix D: Limitations and Future Enhancements

#### D.1. Limitations

- **Data Quality:** Relies on the accuracy of data provided by Polygon.io.
- **Model Assumptions:** Assumes normal distribution of returns, which may not capture extreme market events.
- **Margin Calculation Limits:** While the system now accounts for margin requirements up to 150%, extreme market conditions may necessitate further adjustments.

#### D.2. Future Enhancements

- **Dynamic Volatility Thresholds:** Implement adaptive volatility thresholds based on market conditions.
- **Enhanced Risk Models:** Incorporate stress testing and scenario analysis for extreme events.
- **Regulatory Updates:** Continuously monitor and integrate changes in regulatory requirements.


---
