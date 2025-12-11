import yfinance as yf
import pandas as pd

# Define a function to fetch data from Yahoo Finance
def fetch_stock_data(ticker_list, start_date, end_date):
    stock_data = {}

    # Loop through the list of tickers
    for ticker in ticker_list:
        # Fetch the stock data using yfinance for the given date range
        stock = yf.download(ticker, start=start_date, end=end_date)
        
        # Store the data in the dictionary, with the ticker as the key
        stock_data[ticker] = stock

        # Print a confirmation message
        print(f"Fetched data for {ticker}")
    
    return stock_data

# Convert the fetched stock data into a Pandas DataFrame
def store_data_in_dataframe(stock_data):
    df_list = []
    
    # Loop through the stock data and append each one to the list
    for ticker, data in stock_data.items():
        data['Ticker'] = ticker
        df_list.append(data)
    
    # Concatenate all the individual DataFrames into one
    final_df = pd.concat(df_list)
    
    return final_df

# Example usage
if __name__ == "__main__":
    tickers = ['AAPL', 'GOOGL', 'TSLA']  # List of stock tickers
    start_date = '2022-01-01'
    end_date = '2023-01-01'

    # Fetch the stock data
    stock_data = fetch_stock_data(tickers, start_date, end_date)

    # Store the data in a Pandas DataFrame
    final_dataframe = store_data_in_dataframe(stock_data)

    # Display the DataFrame
    print(final_dataframe.head())
