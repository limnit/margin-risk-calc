#python
import pandas as pd
import numpy as np
import requests
import datetime
import time
import sqlite3
import threading
from scipy.stats import norm
import concurrent.futures
from threading import Lock
from sqlalchemy import create_engine, text
from sqlalchemy.dialects.sqlite import insert
import urllib.parse

API_KEY = 'YOUR_KEY'  # your Polygon.io API key


# Create an engine with connection pooling
engine = create_engine(
    'sqlite:///historical_data.db',
    echo=False,
    connect_args={'check_same_thread': False},
    pool_pre_ping=True
)

def is_broad_index(ticker):
    broad_indexes = ['SPX', 'SPY', 'DJIA', 'NDX', 'COMP', 'RUT']
    return ticker.upper() in broad_indexes

def get_all_tickers_with_classification():
    base_url = 'https://api.polygon.io/v3/reference/tickers'
    params = {
        'apiKey': API_KEY,
        'market': 'stocks',
        'active': 'true',
        'limit': 1000  # Maximum limit per API documentation
    }
    tickers = []
    url = base_url
    while True:
        response = requests.get(url, params=params)
        if response.status_code == 200:
            data = response.json()
            if 'results' in data and data['results']:
                for item in data['results']:
                    ticker = item['ticker']
                    asset_type = item.get('type', '')
                    if asset_type == 'CS':  # Common Stock
                        classification = 'Stock'
                    elif asset_type == 'Index':
                        if is_broad_index(ticker):
                            classification = 'Broad Index'
                        else:
                            classification = 'Sector Index'
                    else:
                        classification = 'Other'
                    tickers.append({'ticker': ticker, 'classification': classification})
                # Handle pagination using 'next_url'
                if 'next_url' in data and data['next_url']:
                    next_url = data['next_url']
                    # Ensure the API key is included in the next_url
                    parsed_url = urllib.parse.urlparse(next_url)
                    query_params = urllib.parse.parse_qs(parsed_url.query)
                    if 'apiKey' not in query_params:
                        # Append the apiKey to the next_url
                        separator = '&' if '?' in next_url else '?'
                        next_url = f"{next_url}{separator}apiKey={API_KEY}"
                    url = next_url
                    params = {}  # Clear params since all parameters are in next_url
                    # Optional: Log progress
                    print(f"Fetched {len(tickers)} tickers so far...")
                    # Optional: Delay between requests
                    time.sleep(0.2)  # Adjust as needed
                else:
                    break  # No more pages
            else:
                break  # No results
        elif response.status_code == 429:
            # Rate limit exceeded, wait and retry
            print("Rate limit exceeded. Waiting for 1 minute...")
            time.sleep(60)
            continue  # Retry the same request
        else:
            print(f"Error fetching tickers: {response.status_code}")
            print(response.text)  # Print response content for debugging
            break  # Exit on other errors
    return tickers

def get_historical_data(ticker, start_date, end_date):
    url = f'https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/day/{start_date}/{end_date}'
    params = {
        'apiKey': API_KEY,
        'adjusted': 'true',
        'sort': 'asc',
        'limit': 5000
    }
    max_retries = 5
    retry_delay = 1
    for attempt in range(max_retries):
        try:
            response = requests.get(url, params=params, timeout=30)
            if response.status_code == 200:
                data = response.json()
                if 'results' in data and data['results']:
                    df = pd.DataFrame(data['results'])
                    df['date'] = pd.to_datetime(df['t'], unit='ms')
                    df['ticker'] = ticker  # Add the 'ticker' column
                    df = df[['ticker', 'date', 'c']].rename(columns={'c': 'close'})
                    return df
                else:
                    print(f"No data found for {ticker}")
                    return pd.DataFrame()
            elif response.status_code == 429:
                print(f"Rate limit exceeded for {ticker}. Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
                retry_delay *= 2
            else:
                print(f"Error fetching data for {ticker}: {response.status_code}")
                return pd.DataFrame()
        except requests.exceptions.Timeout:
            print(f"Timeout occurred for {ticker}. Retrying in {retry_delay} seconds...")
            time.sleep(retry_delay)
            retry_delay *= 2
    print(f"Failed to fetch data for {ticker} after {max_retries} attempts.")
    return pd.DataFrame()

def initialize_database():
    conn = engine.connect()
    conn.execute(text('''
        CREATE TABLE IF NOT EXISTS stock_prices (
            ticker TEXT,
            date DATE,
            close REAL,
            PRIMARY KEY (ticker, date)
        )
    '''))
    conn.close()

def insert_or_replace(table, conn, keys, data_iter):
    data = [dict(zip(keys, row)) for row in data_iter]
    stmt = insert(table.table).values(data)
    stmt = stmt.prefix_with('OR REPLACE')
    conn.execute(stmt)

def weighted_std(series, weights):
    mean = np.average(series, weights=weights)
    variance = np.average((series - mean) ** 2, weights=weights)
    return np.sqrt(variance)

def calculate_dynamic_margin(volatility, position, max_margin):
    # Define minimum margin requirements
    min_margin_long = 0.15  # 15% for long positions
    min_margin_short = 0.20  # 20% for short positions
    volatility_threshold = 1.00  # 100% volatility threshold (can be adjusted)

    # Determine minimum margin based on position type
    if position == 'long':
        min_margin = min_margin_long
        sigma = volatility  # Use downside volatility for long positions
    elif position == 'short':
        min_margin = min_margin_short
        sigma = volatility  # Use upside volatility for short positions
    else:
        # Default to maximum margin if position type is unrecognized
        return max_margin

    # Calculate the margin percentage
    margin = min_margin + (max_margin - min_margin) * (sigma / volatility_threshold)

    # Ensure the margin does not exceed the maximum allowed margin
    margin_percentage = min(margin, max_margin)

    # Ensure margin percentage is at least the minimum margin
    margin_percentage = max(margin_percentage, min_margin)

    return margin_percentage

def get_margin_percentage(volatility, price, classification, position):
    if classification == 'Broad Index':
        return 0.08
    elif classification == 'Sector Index':
        return 0.10
    elif classification == 'Stock':
        if price < 1.0 or volatility > 1.0:
            return 1.00  # Trade on cash basis
        else:
            max_margin = 1.50  # Allow margin requirements up to 150%
            margin_percentage = calculate_dynamic_margin(volatility, position, max_margin)
            return min(margin_percentage, max_margin)
    else:
        return 1.00

def calculate_risk_profiles(historical_data, ticker_info):
    risk_profiles = {}
    market_ticker = 'SPY'
    market_date = datetime.datetime.now().strftime('%Y-%m-%d')

    # Prepare market returns
    market_df = historical_data[historical_data['ticker'] == market_ticker]
    market_df = market_df.sort_values('date')
    market_df['return'] = market_df['close'].pct_change()
    market_returns = market_df[['date', 'return']].dropna()

    for ticker_data in ticker_info:
        ticker = ticker_data['ticker']
        classification = ticker_data['classification']

        if ticker == market_ticker:
            continue
        df = historical_data[historical_data['ticker'] == ticker]
        df = df.sort_values('date')
        df['return'] = df['close'].pct_change()
        df = df.dropna(subset=['return'])

        # Keep only the last 252 trading days
        df = df.tail(252)
        if df.empty or len(df) < 2:
            print(f"Insufficient data for {ticker}. Skipping risk calculations.")
            continue

        returns = df['return'].values

        # Calculate Kurtosis
        kurt = df['return'].kurtosis()
        kurt = round(kurt, 5)

        # Calculate Weighted Volatility
        n = len(returns)
        weights = np.ones(n)
        weights[-5:] = 2  # Double the weight of the last 5 days
        weights /= weights.sum()

        # Separate returns into positive and negative
        positive_mask = returns > 0
        negative_mask = returns < 0
        positive_returns = returns[positive_mask]
        negative_returns = returns[negative_mask]
        positive_weights = weights[positive_mask]
        negative_weights = weights[negative_mask]

        # Calculate Upside Volatility for Short Positions
        if len(positive_returns) > 1:
            upside_volatility = weighted_std(positive_returns, positive_weights) * np.sqrt(252)
        else:
            upside_volatility = np.nan  # Not enough data to calculate volatility

        # Calculate Downside Volatility for Long Positions
        if len(negative_returns) > 1:
            downside_volatility = weighted_std(negative_returns, negative_weights) * np.sqrt(252)
        else:
            downside_volatility = np.nan  # Not enough data to calculate volatility

        # Total Volatility
        if len(returns) > 1:
            total_volatility = weighted_std(returns, weights) * np.sqrt(252)
        else:
            total_volatility = np.nan

        # VaR using Total Volatility
        if not np.isnan(total_volatility):
            confidence_level = 0.95
            z_score = norm.ppf(1 - confidence_level)
            mean_return = np.average(returns, weights=weights)
            var = -(mean_return + z_score * total_volatility / np.sqrt(252)) * np.sqrt(252)
            var = round(var, 5)
        else:
            var = np.nan

        # Beta Calculation
        merged_df = pd.merge(df[['date', 'return']], market_returns, on='date', suffixes=('_stock', '_market'))
        if len(merged_df) > 1:
            covariance = merged_df[['return_stock', 'return_market']].cov().iloc[0, 1]
            market_variance = merged_df['return_market'].var()
            if market_variance != 0:
                beta = covariance / market_variance
                beta = round(beta, 5)
            else:
                beta = np.nan
        else:
            beta = np.nan

        # Get Latest Price
        latest_price = df['close'].iloc[-1]

        # Calculate Margin Percentages
        long_margin_pct = get_margin_percentage(downside_volatility, latest_price, classification, position='long')
        short_margin_pct = get_margin_percentage(upside_volatility, latest_price, classification, position='short')

        # Calculate downRom and upRom
        down_rom = -long_margin_pct
        up_rom = short_margin_pct

        # Prepare security type
        security_type = 'CS' if classification == 'Stock' else classification

        risk_profiles[ticker] = {
            'marketDate': market_date,
            'securityType': security_type,
            'symbol': ticker,
            'downRange': round(down_rom, 5),
            'upRange': round(up_rom, 5),
            'Kurtosis': kurt,
            'VaR': var,
            'Beta': beta,
            'Latest_Price': latest_price,
            'Total_Volatility': total_volatility,
            'Downside_Volatility': downside_volatility,
            'Upside_Volatility': upside_volatility,
            'Classification': classification
        }

    risk_df = pd.DataFrame.from_dict(risk_profiles, orient='index')
    risk_df.reset_index(drop=True, inplace=True)
    return risk_df

def store_initial_data(tickers, start_date, end_date):
    semaphore = threading.Semaphore(5)
    success_count = 0
    failure_count = 0

    def fetch_and_store(ticker):
        nonlocal success_count, failure_count
        try:
            with semaphore:
                print(f"Fetching initial data for {ticker}")
                df = get_historical_data(ticker, start_date, end_date)
                if not df.empty:
                    try:
                        df.to_sql('stock_prices', engine, if_exists='append', index=False, method=insert_or_replace)
                        print(f"Data stored for {ticker}")
                        success_count += 1
                    except Exception as e:
                        print(f"Error inserting data for {ticker}: {e}")
                        failure_count += 1
                else:
                    print(f"No data for {ticker}")
                    failure_count += 1
        except Exception as e:
            print(f"Unhandled exception for {ticker}: {e}")
            failure_count += 1

    max_workers = min(10, len(tickers))
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(fetch_and_store, ticker): ticker for ticker in tickers}
        for future in concurrent.futures.as_completed(futures):
            pass  # Already handled in fetch_and_store

    print(f"Data fetching completed. Success: {success_count}, Failures: {failure_count}")

def update_daily_data(tickers):
    semaphore = threading.Semaphore(5)
    success_count = 0
    failure_count = 0

    def fetch_and_update(ticker):
        nonlocal success_count, failure_count
        try:
            with semaphore:
                print(f"Updating data for {ticker}")
                conn = engine.connect()
                result = conn.execute(text('SELECT MAX(date) FROM stock_prices WHERE ticker = :ticker'), {'ticker': ticker})
                last_date = result.scalar()
                conn.close()

                if last_date:
                    start_date = (datetime.datetime.strptime(last_date, '%Y-%m-%d') + datetime.timedelta(days=1)).strftime('%Y-%m-%d')
                else:
                    start_date = (datetime.datetime.now() - datetime.timedelta(days=365)).strftime('%Y-%m-%d')

                end_date = datetime.datetime.now().strftime('%Y-%m-%d')

                df = get_historical_data(ticker, start_date, end_date)
                if not df.empty:
                    try:
                        df.to_sql('stock_prices', engine, if_exists='append', index=False, method=insert_or_replace)
                        print(f"Data updated for {ticker}")
                        success_count += 1
                    except Exception as e:
                        print(f"Error updating data for {ticker}: {e}")
                        failure_count += 1
                else:
                    print(f"No new data for {ticker}")
                    failure_count += 1
        except Exception as e:
            print(f"Unhandled exception for {ticker}: {e}")
            failure_count += 1

    max_workers = min(10, len(tickers))
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(fetch_and_update, ticker): ticker for ticker in tickers}
        for future in concurrent.futures.as_completed(futures):
            pass  # Already handled in fetch_and_update

    print(f"Data update completed. Success: {success_count}, Failures: {failure_count}")

def load_historical_data(tickers):
    conn = engine.connect()
    start_calc_date = (datetime.datetime.now() - datetime.timedelta(days=365)).strftime('%Y-%m-%d')
    df_list = []
    batch_size = 900  # Less than 999 to stay within SQLite parameter limit
    for i in range(0, len(tickers), batch_size):
        tickers_batch = tickers[i:i+batch_size]
        placeholders = ','.join(f':ticker_{j}' for j in range(len(tickers_batch)))
        query = text(f'''
            SELECT ticker, date, close FROM stock_prices
            WHERE ticker IN ({placeholders}) AND date >= :start_date
        ''')
        params = {f'ticker_{j}': ticker for j, ticker in enumerate(tickers_batch)}
        params['start_date'] = start_calc_date
        try:
            df_batch = pd.read_sql_query(query, conn, params=params)
            print(f"Retrieved data for batch {i//batch_size + 1}: {len(df_batch)} records")
            df_list.append(df_batch)
        except Exception as e:
            print(f"Error retrieving data for batch starting at index {i}: {e}")
    conn.close()
    if df_list:
        df = pd.concat(df_list, ignore_index=True)
        print(f"Total records retrieved: {len(df)}")
        return df
    else:
        print("No data retrieved from the database.")
        return pd.DataFrame()

def main():
    global start_date, end_date
    initialize_database()

    end_date = datetime.datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.datetime.now() - datetime.timedelta(days=365)).strftime('%Y-%m-%d')

    # Fetch tickers with classifications
    print("Fetching all tickers with classification...")
    ticker_info = get_all_tickers_with_classification()
    tickers = [t['ticker'] for t in ticker_info]
    print(f"Total tickers fetched: {len(tickers)}")

    # Ensure 'SPY' is included for beta calculation
    if 'SPY' not in tickers:
        tickers.append('SPY')
        ticker_info.append({'ticker': 'SPY', 'classification': 'Broad Index'})

    # For testing, limit the number of tickers
    # tickers = tickers[:100]
    # ticker_info = ticker_info[:100]

    # Check if data exists
    conn = engine.connect()
    result = conn.execute(text('SELECT COUNT(DISTINCT ticker) FROM stock_prices'))
    count = result.scalar()
    conn.close()

    if count == 0:
        print("Storing initial data...")
        store_initial_data(tickers, start_date, end_date)
    else:
        print("Updating daily data...")
        update_daily_data(tickers)

    historical_data = load_historical_data(tickers)

    risk_df = calculate_risk_profiles(historical_data, ticker_info)

    # Select the columns to include in the output
    output_df = risk_df[['marketDate', 'securityType', 'symbol', 'downRange', 'upRange', 'Kurtosis', 'VaR', 'Beta']]

    # Save to CSV
    output_df.to_csv('daily_risk_profiles.csv', index=False)

    print("Risk profiles generated successfully.")
    print(output_df.head())

if __name__ == '__main__':
    main()
