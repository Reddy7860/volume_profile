import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import skew, kurtosis
from scipy.stats import linregress
from collections import defaultdict
import requests
from io import StringIO
from datetime import datetime
import re
from sentence_transformers import SentenceTransformer, util
import openai
import talib
import math
import torch
from datetime import datetime, timedelta
from datetime import date
import numpy as np
import plotly.graph_objects as go
from scipy.signal import find_peaks
import json
import ast
import itertools

# Sidebar for inputs
st.sidebar.title("Trading Dashboard")
capital_per_trade = st.sidebar.number_input("Capital Per Trade", value=2000, min_value=100)
selected_strategy = st.sidebar.selectbox("Select Strategy", ['Momentum', 'Reversal', 'Breakout'])



# Initialize session state values if they don't exist
if 'xsrf_token' not in st.session_state:
    st.session_state.xsrf_token = ""
if 'laravel_token' not in st.session_state:
    st.session_state.laravel_token = ""
if 'xsrf_cookie' not in st.session_state:
    st.session_state.xsrf_cookie = ""
if 'laravel_session' not in st.session_state:
    st.session_state.laravel_session = ""
if 'predicted_trend' not in st.session_state:
    st.session_state.predicted_trend = ""
# Initialize session state for chat history
if 'history' not in st.session_state:
    st.session_state.history = []

# Section 1: Stock and Volume Profile Inputs
st.title("Real-time Volume Profile with Market Shape Detection")

ticker = st.text_input("Enter Stock Ticker", value="AAPL")

# Main interface with tab selection
tab = st.selectbox("Select Tab", ["Chat Interface", "Headers and Cookies","Volume Footprint Upload"])



if tab == "Volume Footprint Upload":
    # Title of the app
    st.title("File Upload and Timeframe Selection")

    # File uploader
    uploaded_file = st.file_uploader("Upload a text file")

    # Dropdown for timeframe selection
    timeframe_options = ["5 seconds", "5 minutes", "1 day"]
    timeframe = st.selectbox("Select Timeframe", timeframe_options, index=1)  # Default to "5 minutes"

    # Date and time picker for base timestamp
    # base_date = st.date_input("Select Base Date", datetime(2024, 6, 14).date())
    base_date = st.date_input("Select Base Date", date.today())
    base_time = st.time_input("Select Base Time", datetime.strptime("15:55:00", "%H:%M:%S").time())

    # Combine date and time into a single datetime object
    base_timestamp = datetime.combine(base_date, base_time)

    # Date picker for subsetting the data
    # subset_date = st.date_input("Select Subset Date", datetime(2024, 6, 1).date())
    subset_date = st.date_input("Select Subset Date", date.today())

    marker = st.text_input("Enter Marker", value="~m~98~m~")
    st_input = st.text_input("Enter st_input", value="st23")

    default_date = datetime.today().date()
    input_date = st.date_input("Start Date", value=default_date)

    # Fetch stock data in real-time
    def fetch_stock_data(ticker, start, interval='1m'):
        stock_data = yf.download(ticker, start=start, interval=interval)
        return stock_data

    data = fetch_stock_data(ticker, input_date)

    temp_data = data.copy()

    # Ensure the DataFrame index is a DatetimeIndex for VWAP calculations
    temp_data.reset_index(inplace=True)  # Reset index for column access
    temp_data.set_index(pd.DatetimeIndex(temp_data["Datetime"]), inplace=True)  # Use 'Datetime' as the index

    # Function to calculate Cumulative Volume Delta (CVD)
    def calculate_cvd(data):
        """
        Calculate the Cumulative Volume Delta (CVD) and additional metrics.
        """
        data['delta'] = data['Close'] - data['Open']  # Price delta
        data['buy_volume'] = data['Volume'] * (data['delta'] > 0).astype(int)
        data['sell_volume'] = data['Volume'] * (data['delta'] < 0).astype(int)
        data['cvd'] = (data['buy_volume'] - data['sell_volume']).cumsum()
        return data

    # Function to identify support and resistance levels
    def identify_support_resistance(data, start_time, end_time):
        """
        Identify support (most selling) and resistance (most buying) levels for a given time range.
        """
        time_frame = data.between_time(start_time, end_time).copy()
        time_frame = calculate_cvd(time_frame)
        
        if time_frame.empty:
            return {}

        # Support: Price level with most selling (most negative CVD)
        support_idx = time_frame['cvd'].idxmin()
        support_level = time_frame.loc[support_idx, 'Close']
        support_time = support_idx

        # Resistance: Price level with most buying (most positive CVD)
        resistance_idx = time_frame['cvd'].idxmax()
        resistance_level = time_frame.loc[resistance_idx, 'Close']
        resistance_time = resistance_idx

        return {
            "support_level": round(support_level, 2),
            "support_time": support_time.tz_localize(None).strftime('%Y-%m-%d %H:%M:%S'),
            "resistance_level": round(resistance_level, 2),
            "resistance_time": resistance_time.tz_localize(None).strftime('%Y-%m-%d %H:%M:%S'),
        }

    # Calculate CVD for the 09:30-16:00 timeframe
    cvd_data = temp_data.between_time("09:30", "16:00").copy()
    cvd_data = calculate_cvd(cvd_data)

    # Identify support and resistance for the 09:30-10:30 timeframe
    support_resistance_stats = identify_support_resistance(temp_data, "09:30", "10:30")
    support_level = support_resistance_stats["support_level"]
    support_time = support_resistance_stats["support_time"]
    resistance_level = support_resistance_stats["resistance_level"]
    resistance_time = support_resistance_stats["resistance_time"]

    # # Adding Buy/Sell signals to the data
    # cvd_data['signal'] = None
    # cvd_data['signal_type'] = None

    # # Logic for Buy/Sell signals
    # cvd_data['signal'] = cvd_data.apply(
    #     lambda row: row['Close'] if (row['Close'] > resistance_level and row['cvd'] > cvd_data.loc[resistance_time, 'cvd']) else (
    #         row['Close'] if (row['Close'] < support_level and row['cvd'] < cvd_data.loc[support_time, 'cvd']) else None),
    #     axis=1
    # )

    # cvd_data['signal_type'] = cvd_data.apply(
    #     lambda row: 'Buy' if (row['Close'] > resistance_level and row['cvd'] > cvd_data.loc[resistance_time, 'cvd']) else (
    #         'Sell' if (row['Close'] < support_level and row['cvd'] < cvd_data.loc[support_time, 'cvd']) else None),
    #     axis=1
    # )

    # Identify the first Buy signal
    first_buy_signal = cvd_data[(cvd_data['Close'] > resistance_level) & 
                                (cvd_data['cvd'] > cvd_data.loc[resistance_time, 'cvd'])].iloc[:1]

    # Identify the first Sell signal
    first_sell_signal = cvd_data[(cvd_data['Close'] < support_level) & 
                                (cvd_data['cvd'] < cvd_data.loc[support_time, 'cvd'])].iloc[:1]

    # Add first Buy and Sell timestamps if available
    first_buy_time = first_buy_signal.index[0].strftime('%Y-%m-%d %H:%M:%S') if not first_buy_signal.empty else "N/A"
    first_sell_time = first_sell_signal.index[0].strftime('%Y-%m-%d %H:%M:%S') if not first_sell_signal.empty else "N/A"

    st.write("Buy Time: ", first_buy_time)
    st.write("Sell Time: ", first_sell_time)

    # Update hovertext to include first Buy/Sell timestamps
    cvd_data['hovertext'] = (
        "Time: " + cvd_data.index.strftime('%Y-%m-%d %H:%M:%S') +
        "<br>Open: " + round(cvd_data['Open'], 2).astype(str) +
        "<br>High: " + round(cvd_data['High'], 2).astype(str) +
        "<br>Low: " + round(cvd_data['Low'], 2).astype(str) +
        "<br>Close: " + round(cvd_data['Close'], 2).astype(str) +
        "<br>CVD: " + round(cvd_data['cvd'], 2).astype(str) +
        f"<br>Support Level: {support_level}" +
        f"<br>Support Time: {support_time}" +
        f"<br>Resistance Level: {resistance_level}" +
        f"<br>Resistance Time: {resistance_time}" +
        f"<br>First Buy Time: {first_buy_time}" +
        f"<br>First Sell Time: {first_sell_time}"
    )

    # Create the candlestick chart with CVD
    fig = go.Figure()

    # Add candlestick trace
    fig.add_trace(go.Candlestick(
        x=cvd_data.index,
        open=cvd_data['Open'],
        high=cvd_data['High'],
        low=cvd_data['Low'],
        close=cvd_data['Close'],
        name='Candlestick',
        hovertext=cvd_data['hovertext'],
        hoverinfo='text'
    ))

    # Add CVD as a line trace on a secondary y-axis
    fig.add_trace(go.Scatter(
        x=cvd_data.index,
        y=cvd_data['cvd'],
        mode='lines',
        name='Cumulative Volume Delta (CVD)',
        line=dict(color='orange'),
        yaxis='y2',  # Use secondary y-axis
    ))

    # Add support line
    fig.add_shape(
        type="line",
        x0=cvd_data.index.min(),
        x1=cvd_data.index.max(),
        y0=support_level,
        y1=support_level,
        line=dict(color="blue", dash="dot"),
        name="Support Level",
    )

    # Add resistance line
    fig.add_shape(
        type="line",
        x0=cvd_data.index.min(),
        x1=cvd_data.index.max(),
        y0=resistance_level,
        y1=resistance_level,
        line=dict(color="red", dash="dot"),
        name="Resistance Level",
    )

    # # # Update layout to include a secondary y-axis for CVD
    # # fig.update_layout(
    # #     title="Candlestick Chart with CVD (09:30-16:00) and Support/Resistance (09:30-10:30)",
    # #     xaxis_title="Time",
    # #     yaxis_title="Price",
    # #     yaxis2=dict(
    # #         title="CVD",
    # #         overlaying='y',
    # #         side='right'
    # #     ),
    # #     template="plotly_dark",
    # #     hovermode="x unified"
    # # )

    # # Adding Buy signals (triangle-up)
    # fig.add_trace(go.Scatter(
    #     x=cvd_data[cvd_data['signal_type'] == 'Buy'].index,
    #     y=cvd_data[cvd_data['signal_type'] == 'Buy']['signal'],
    #     mode='markers',
    #     name='Buy Signal',
    #     marker=dict(symbol='triangle-up', color='green', size=10),
    #     hoverinfo='text',
    #     hovertext="Buy Signal"
    # ))

    # # Adding Sell signals (triangle-down)
    # fig.add_trace(go.Scatter(
    #     x=cvd_data[cvd_data['signal_type'] == 'Sell'].index,
    #     y=cvd_data[cvd_data['signal_type'] == 'Sell']['signal'],
    #     mode='markers',
    #     name='Sell Signal',
    #     marker=dict(symbol='triangle-down', color='red', size=10),
    #     hoverinfo='text',
    #     hovertext="Sell Signal"
    # ))

    # # Update layout to include the signals
    # fig.update_layout(
    #     title="Candlestick Chart with CVD and Buy/Sell Signals",
    #     xaxis_title="Time",
    #     yaxis_title="Price",
    #     yaxis2=dict(
    #         title="CVD",
    #         overlaying='y',
    #         side='right'
    #     ),
    #     template="plotly_dark",
    #     hovermode="x unified"
    # )

    # Add Buy signal (triangle-up) to the chart
    if not first_buy_signal.empty:
        fig.add_trace(go.Scatter(
            x=first_buy_signal.index,
            y=first_buy_signal['Close'],
            mode='markers',
            name='First Buy Signal',
            marker=dict(symbol='triangle-up', color='green', size=10),
            hoverinfo='text',
            hovertext="First Buy Signal"
        ))

    # Add Sell signal (triangle-down) to the chart
    if not first_sell_signal.empty:
        fig.add_trace(go.Scatter(
            x=first_sell_signal.index,
            y=first_sell_signal['Close'],
            mode='markers',
            name='First Sell Signal',
            marker=dict(symbol='triangle-down', color='red', size=10),
            hoverinfo='text',
            hovertext="First Sell Signal"
        ))

    
    # Update layout to include the filtered signals
    fig.update_layout(
        title="Candlestick Chart with CVD and First Buy/Sell Signals",
        xaxis_title="Time",
        yaxis_title="Price",
        yaxis2=dict(
            title="CVD",
            overlaying='y',
            side='right'
        ),
        template="plotly_dark",
        hovermode="x unified"
    )


    # Display the chart in Streamlit
    st.plotly_chart(fig)

    # Submit button
    if st.button("Submit"):
        if uploaded_file is not None:   
            # Read and process the uploaded file
            file_content = uploaded_file.read().decode("utf-8")

            # # Strip everything after the ~m~98~m~ marker
            # marker = '~m~98~m~'
            file_content = file_content.split(marker)[0]

            try:
                # Load the JSON data from the file content
                main_data = json.loads(file_content)
                
                data_section = main_data['p'][1][st_input]['ns']['d']
    #             data_section = main_data['p'][1]['st2']['ns']['d']
                nested_data = json.loads(data_section)
                footprint_levels = nested_data['graphicsCmds']['create']['footprintLevels']
                df = pd.DataFrame(footprint_levels[0]['data'])
                
                footprints = nested_data['graphicsCmds']['create']['footprints']
                df1 = pd.DataFrame(footprints[0]['data'])
                
                # Display the resulting DataFrame
                st.write("Footprint Levels DataFrame:")
                st.dataframe(df)
                
                st.write("Footprints DataFrame:")
                st.dataframe(df1)

                # Define the trading holidays
                trading_holidays = [
                    datetime(2024, 1, 1),   # Monday, January 1 - New Year's Day
                    datetime(2024, 1, 15),  # Monday, January 15 - Martin Luther King Jr. Day
                    datetime(2024, 2, 19),  # Monday, February 19 - Presidents' Day
                    datetime(2024, 3, 29),  # Friday, March 29 - Good Friday
                    datetime(2024, 5, 27),  # Monday, May 27 - Memorial Day
                    datetime(2024, 6, 19),  # Wednesday, June 19 - Juneteenth National Independence Day
                    datetime(2024, 7, 4),   # Thursday, July 4 - Independence Day
                    datetime(2024, 9, 2),   # Monday, September 2 - Labor Day
                    datetime(2024, 11, 28), # Thursday, November 28 - Thanksgiving Day
                    datetime(2024, 12, 25), # Wednesday, December 25 - Christmas Day
                    datetime(2025, 1, 1),   # Wednesday, January 1 - New Year's Day
                    datetime(2025, 1, 20),  # Monday, January 20 - Martin Luther King Jr. Day
                    datetime(2025, 2, 17),  # Monday, February 17 - Presidents' Day
                    datetime(2025, 4, 18),  # Friday, April 18 - Good Friday
                    datetime(2025, 5, 26),  # Monday, May 26 - Memorial Day
                    datetime(2025, 6, 19),  # Thursday, June 19 - Juneteenth National Independence Day
                    datetime(2025, 7, 4),   # Friday, July 4 - Independence Day
                    datetime(2025, 9, 1),   # Monday, September 1 - Labor Day
                    datetime(2025, 11, 27), # Thursday, November 27 - Thanksgiving Day
                    datetime(2025, 12, 25)  # Thursday, December 25 - Christmas Day
                ]

                # Initialize the base timestamp for index 2245
                # base_timestamp = datetime.strptime('2024-12-06 15:55:00', '%Y-%m-%d %H:%M:%S')
                # current_timestamp = base_timestamp

                current_timestamp = base_timestamp

                # Initialize a dictionary to store timestamps for each index
                # index_to_timestamp = {2245: current_timestamp}
                index_to_timestamp = {max(df1['index']): current_timestamp}

                # Market hours
                market_open = timedelta(hours=9, minutes=30)
                market_close = timedelta(hours=15, minutes=55)
                special_close = {
                    datetime(2024, 11, 29).date(): timedelta(hours=12, minutes=55)  # Special close time on 2024-11-29
                }
                day_increment = timedelta(days=1)
                weekend_days = [5, 6]  # Saturday and Sunday

                # Calculate the timestamps backward in 5-minute intervals, excluding weekends and outside market hours
                # for index in range(2244, -1, -1):
                for index in range(max(df1['index'])-1, -1, -1):
                    # Subtract 5 minutes
                    current_timestamp -= timedelta(minutes=5)
                    
                    # Check if current timestamp is before market open
                    while (current_timestamp.time() < (datetime.min + market_open).time() or
                        current_timestamp.time() > (datetime.min + special_close.get(current_timestamp.date(), market_close)).time() or
                        current_timestamp.weekday() in weekend_days or
                        current_timestamp.date() in [holiday.date() for holiday in trading_holidays]):
                        # Move to previous trading day if before market open
                        if current_timestamp.time() < (datetime.min + market_open).time():
                            current_timestamp = datetime.combine(current_timestamp.date() - day_increment, (datetime.min + special_close.get(current_timestamp.date() - day_increment, market_close)).time())
                        else:
                            # Otherwise, just subtract 5 minutes
                            current_timestamp -= timedelta(minutes=5)
                        
                        # Skip weekends and trading holidays
                        while current_timestamp.weekday() in weekend_days or current_timestamp.date() in [holiday.date() for holiday in trading_holidays]:
                            current_timestamp -= day_increment
                            current_timestamp = datetime.combine(current_timestamp.date(), (datetime.min + special_close.get(current_timestamp.date(), market_close)).time())
                    
                    # Assign the calculated timestamp to the index
                    index_to_timestamp[index] = current_timestamp

                # Create a list to hold the time series data
                time_series_data = []

                # Iterate over df1 and extract levels data
                for i, row in df1.iterrows():
                    timestamp = index_to_timestamp.get(row['index'])
                    
                    if timestamp:
                        levels = row['levels']
                        for level in levels:
                            time_series_data.append({
                                'timestamp': timestamp,
                                'price': level['price'],
                                'buyVolume': level['buyVolume'],
                                'sellVolume': level['sellVolume'],
                                'imbalance': level['imbalance'],
                                'index': row['index']
                            })

                # Create the dataframe from the time series data
                series_df = pd.DataFrame(time_series_data)

                series_df['timestamp'] = pd.to_datetime(series_df['timestamp'])
                series_df['date'] = series_df['timestamp'].dt.date

                # Subset the data based on the selected subset date
                subset_df = series_df[series_df['date'] == subset_date]

                filtered_df = subset_df.copy()

                # Sort by timestamp and price ascending
                filtered_df = filtered_df.sort_values(by=['timestamp', 'price']).reset_index(drop=True)

                # Calculate total volume at each price level
                filtered_df['totalVolume'] = filtered_df['buyVolume'] + filtered_df['sellVolume']

                # Group by timestamp and identify the Point of Control (POC) for each 5-minute interval
                def calculate_poc(group):
                    poc_price = group.loc[group['totalVolume'].idxmax(), 'price']
                    group['poc'] = poc_price
                    
                    # Calculate highest bid stacked imbalance and ask stacked imbalance
                    group['highest_bid_stacked_imbalance'] = group['buyVolume'].max()
                    group['highest_ask_stacked_imbalance'] = group['sellVolume'].max()
                    
                    # Calculate highest ask imbalance stack price (consider imbalance as 'sell' or 'both')
                    ask_imbalance_filter = group[(group['imbalance'] == 'sell') | (group['imbalance'] == 'both')]
                    if not ask_imbalance_filter.empty:
                        highest_ask_imbalance_stack_price = ask_imbalance_filter.loc[ask_imbalance_filter['sellVolume'].idxmax(), 'price']
                    else:
                        highest_ask_imbalance_stack_price = None
                    group['highest_ask_imbalance_stack_price'] = highest_ask_imbalance_stack_price
                    
                    # Calculate highest bid imbalance stack price (consider imbalance as 'buy' or 'both')
                    bid_imbalance_filter = group[(group['imbalance'] == 'buy') | (group['imbalance'] == 'both')]
                    if not bid_imbalance_filter.empty:
                        highest_bid_imbalance_stack_price = bid_imbalance_filter.loc[bid_imbalance_filter['buyVolume'].idxmax(), 'price']
                    else:
                        highest_bid_imbalance_stack_price = None
                    group['highest_bid_imbalance_stack_price'] = highest_bid_imbalance_stack_price
                    
                    # Calculate lowest ask imbalance price (consider imbalance as 'sell' or 'both')
                    if not ask_imbalance_filter.empty:
                        lowest_ask_imbalance_price = ask_imbalance_filter['price'].min()
                    else:
                        lowest_ask_imbalance_price = None
                    group['lowest_ask_imbalance_price'] = lowest_ask_imbalance_price
                    
                    # Calculate highest bid imbalance price (consider imbalance as 'buy' or 'both')
                    if not bid_imbalance_filter.empty:
                        highest_bid_imbalance_price = bid_imbalance_filter['price'].max()
                    else:
                        highest_bid_imbalance_price = None
                    group['highest_bid_imbalance_price'] = highest_bid_imbalance_price
                    
                    return group

                filtered_df = filtered_df.groupby('timestamp', group_keys=False).apply(calculate_poc)



                # Calculate delta (buyVolume - sellVolume)
                filtered_df['delta'] = filtered_df['buyVolume'] - filtered_df['sellVolume']

                # Calculate total ask imbalance count and highest stacked imbalance count
                def calculate_imbalances(group):
                    # Total ask imbalance count (where imbalance is 'sell' or 'both')
                    ask_imbalance_count = ((group['imbalance'] == 'sell') | (group['imbalance'] == 'both')).sum()
                    group['total_ask_imbalance_count'] = ask_imbalance_count

                    # Highest stacked ask imbalance count (consecutive 'sell' or 'both' imbalance)
                    max_stacked_ask_imbalance = ((group['imbalance'] == 'sell') | (group['imbalance'] == 'both')).astype(int).groupby(((group['imbalance'] != 'sell') & (group['imbalance'] != 'both')).cumsum()).cumsum().max()
                    group['highest_stacked_ask_imbalance'] = max_stacked_ask_imbalance

                    # Total bid imbalance count (where imbalance is 'buy' or 'both')
                    bid_imbalance_count = ((group['imbalance'] == 'buy') | (group['imbalance'] == 'both')).sum()
                    group['total_bid_imbalance_count'] = bid_imbalance_count

                    # Highest stacked bid imbalance count (consecutive 'buy' or 'both' imbalance)
                    max_stacked_bid_imbalance = ((group['imbalance'] == 'buy') | (group['imbalance'] == 'both')).astype(int).groupby(((group['imbalance'] != 'buy') & (group['imbalance'] != 'both')).cumsum()).cumsum().max()
                    group['highest_stacked_bid_imbalance'] = max_stacked_bid_imbalance

                    return group

                filtered_df = filtered_df.groupby('timestamp', group_keys=False).apply(calculate_imbalances)

                # Calculate total delta
                def calculate_delta(group):
                    group['candle_delta'] = group['buyVolume'].sum() - group['sellVolume'].sum()
                    return group

                filtered_df = filtered_df.groupby('timestamp', group_keys=False).apply(calculate_delta)

                filtered_df = filtered_df.sort_values(by=['timestamp', 'price']).reset_index(drop=True)

                def add_support_resistance_by_timestamp(df):
                    # Initialize lists for final results
                    support_levels_per_timestamp = []
                    resistance_levels_per_timestamp = []

                    # Group by timestamp
                    grouped = df.groupby('timestamp')

                    for timestamp, group in grouped:
                        support_levels = []
                        resistance_levels = []
                        
                        # Group consecutive rows with the same imbalance
                        for _, sub_group in group.groupby((group['imbalance'] != group['imbalance'].shift()).cumsum()):
                            if len(sub_group) >= 3:  # Ensure the sub-group has at least 3 rows
                                prices = sub_group['price'].tolist()
                                imbalance_type = sub_group['imbalance'].iloc[0]
                                
                                # Identify support and resistance levels based on imbalance type
                                if all(sub_group['imbalance'].isin(['buy', 'both'])):
                                    support_levels.append([round(p, 2) for p in prices])
                                elif all(sub_group['imbalance'].isin(['sell', 'both'])):
                                    resistance_levels.append([round(p, 2) for p in prices])

                        # Store levels for the current timestamp
                        support_levels_per_timestamp.append((timestamp, support_levels))
                        resistance_levels_per_timestamp.append((timestamp, resistance_levels))
                    
                    # Create new DataFrame columns
                    df['support_imbalance'] = df['timestamp'].map(
                        dict((timestamp, levels) for timestamp, levels in support_levels_per_timestamp)
                    )
                    df['resistance_imbalance'] = df['timestamp'].map(
                        dict((timestamp, levels) for timestamp, levels in resistance_levels_per_timestamp)
                    )

                    return df

                # Apply the function to the filtered DataFrame
                filtered_df = add_support_resistance_by_timestamp(filtered_df)

                # Sort by timestamp and price
                filtered_df = filtered_df.sort_values(by=['timestamp', 'price']).reset_index(drop=True)

                # Group by timestamp and check for unfinished bid and ask auctions
                def check_auctions(group):
                    min_price_row = group.loc[group['price'].idxmin()]
                    max_price_row = group.loc[group['price'].idxmax()]
                    
                    buy_auction_status = 'incomplete' if min_price_row['buyVolume'] > 0 and min_price_row['sellVolume'] > 0 else 'complete'
                    sell_auction_status = 'incomplete' if max_price_row['buyVolume'] > 0 and max_price_row['sellVolume'] > 0 else 'complete'
                    
                    group['buy_auction_status'] = buy_auction_status
                    group['sell_auction_status'] = sell_auction_status
                    
                    return group

                # Apply the auction check function to each group
                filtered_df = filtered_df.groupby('timestamp').apply(check_auctions).reset_index(drop=True)

                

                # Define a function that converts the value to a list if needed
                def safe_literal_eval(val):
                    if isinstance(val, str):
                        try:
                            return ast.literal_eval(val)
                        except ValueError:
                            print(f"Error in evaluating: {val}")
                            return val  # Optionally handle bad strings gracefully
                    return val

                # Apply to the columns
                filtered_df['support_imbalance'] = filtered_df['support_imbalance'].apply(safe_literal_eval)
                filtered_df['resistance_imbalance'] = filtered_df['resistance_imbalance'].apply(safe_literal_eval)

                # Flatten the nested list and then take the set of unique elements
                filtered_df['support_imbalance_count'] = filtered_df['support_imbalance'].apply(
                    lambda x: len(set(itertools.chain.from_iterable(x))) if isinstance(x, list) else 0
                )

                filtered_df['resistance_imbalance_count'] = filtered_df['resistance_imbalance'].apply(
                    lambda x: len(set(itertools.chain.from_iterable(x))) if isinstance(x, list) else 0
                )

                # Initialize active support and resistance levels
                active_support_levels = [item for sublist in filtered_df.loc[0, 'support_imbalance'] for item in sublist]
                active_resistance_levels = [item for sublist in filtered_df.loc[0, 'resistance_imbalance'] for item in sublist]

                # Function to update active support and resistance levels
                def update_active_levels(active_levels, traded_price):
                    # Remove levels that have been breached
                    return [level for level in active_levels if level != traded_price]

                # Group by timestamp and update active levels
                active_levels_df = []
                for timestamp, group in filtered_df.groupby('timestamp'):
                    # print(timestamp)
                    # print(active_support_levels)
                    for idx, row in group.iterrows():
                        traded_price = round(row['price'],2)
                        # print(traded_price)
                        # Add new support and resistance levels from the current row
                        new_support_levels = [item for sublist in row['support_imbalance'] for item in sublist]
                        new_resistance_levels = [item for sublist in row['resistance_imbalance'] for item in sublist]
                        
                        # Update active support and resistance levels with new levels
                        active_support_levels = list(set(active_support_levels + new_support_levels))
                        active_resistance_levels = list(set(active_resistance_levels + new_resistance_levels))
                        
                        # Update support levels
                        active_support_levels = update_active_levels(active_support_levels, traded_price)
                        
                        # Update resistance levels
                        active_resistance_levels = update_active_levels(active_resistance_levels, traded_price)
                    
                    # Sort the active support and resistance levels
                    active_support_levels = sorted(active_support_levels)
                    active_resistance_levels = sorted(active_resistance_levels)
                    
                    # Append the updated levels to the dataframe
                    active_levels_df.append({
                        'timestamp': timestamp,
                        'active_support_levels': active_support_levels,
                        'active_resistance_levels': active_resistance_levels
                    })

                # Create a DataFrame for active levels
                active_levels_df = pd.DataFrame(active_levels_df)

                temp_df = active_levels_df

                temp_df['active_support_levels'] = temp_df['active_support_levels'].apply(str)
                temp_df['active_resistance_levels'] = temp_df['active_resistance_levels'].apply(str)
                temp_df = temp_df.drop_duplicates().reset_index()

                
                # Convert 'support_imbalance' and 'resistance_imbalance' columns to strings
                filtered_df['support_imbalance'] = filtered_df['support_imbalance'].apply(str)
                filtered_df['resistance_imbalance'] = filtered_df['resistance_imbalance'].apply(str)

                # Reset index and filter required columns
                temp_filter_df = filtered_df[['timestamp','poc','highest_bid_stacked_imbalance','highest_ask_stacked_imbalance','highest_ask_imbalance_stack_price','highest_bid_imbalance_stack_price',
                                            'lowest_ask_imbalance_price','highest_bid_imbalance_price',
                                            'total_ask_imbalance_count','highest_stacked_ask_imbalance',
                                            'total_bid_imbalance_count','highest_stacked_bid_imbalance', 
                                            # 'active_support_levels','active_resistance_levels',
                                            'support_imbalance_count','resistance_imbalance_count',
                                            'support_imbalance', 'resistance_imbalance','candle_delta']].drop_duplicates().reset_index(drop=True)

                # Create consecutive POC flag and count highest consecutive POC
                temp_filter_df['consecutive_poc_flag'] = temp_filter_df['poc'].eq(temp_filter_df['poc'].shift())

                # Calculate the highest consecutive POC count
                temp_filter_df['highest_consecutive_poc_count'] = temp_filter_df['poc'].groupby((temp_filter_df['poc'] != temp_filter_df['poc'].shift()).cumsum()).transform('count')

                st.dataframe(temp_filter_df)

                # ticker = 'WMT'
                temp_filter_df['timestamp'] = pd.to_datetime(temp_filter_df['timestamp'])

                # Downloading NKE data from yfinance in 5-minute intervals (only available for the last 60 days)
                stock_data = yf.download(ticker, interval='5m', period='5d', progress=False)

                # Resetting the index of downloaded NKE data and renaming columns
                stock_data.reset_index(inplace=True)
                stock_data['timestamp'] = pd.to_datetime(stock_data['Datetime']).dt.tz_localize(None)

                # Rounding Open, High, Low, Close, and Adj Close columns to two decimals
                stock_data[['Open', 'High', 'Low', 'Close', 'Adj Close']] = stock_data[['Open', 'High', 'Low', 'Close', 'Adj Close']].round(2)

                # Adding a new column to indicate the trend as Bullish or Bearish
                stock_data['Candle_Trend'] = stock_data.apply(lambda row: 'Bullish' if row['Adj Close'] > row['Open'] else 'Bearish', axis=1)

                # Left join the existing dataframe with NKE data
                merged_df = pd.merge(temp_filter_df, stock_data, how='left', on='timestamp')

                # Adding a new column 'poc_direction'
                merged_df['poc_direction'] = merged_df.apply(lambda row: 'bullish' if row['Adj Close'] >= row['poc'] else 'bearish', axis=1)
                merged_df['highest_ask_imbalance_price_direction'] = merged_df.apply(
                    lambda row: '' if np.isnan(row['highest_ask_imbalance_stack_price']) else 
                                ('bullish' if row['Adj Close'] >= row['highest_ask_imbalance_stack_price'] else 'bearish'),
                    axis=1
                )
                merged_df['highest_bid_imbalance_price_direction'] = merged_df.apply(
                    lambda row: '' if np.isnan(row['highest_bid_imbalance_stack_price']) else 
                                ('bullish' if row['Adj Close'] >= row['highest_bid_imbalance_stack_price'] else 'bearish'),
                    axis=1
                )

                merged_df['total_bid_ask_count_direction'] = merged_df.apply(
                    lambda row: 'bearish' if row['total_ask_imbalance_count'] > row['total_bid_imbalance_count'] else 
                                ('neutral' if row['total_ask_imbalance_count'] == row['total_bid_imbalance_count'] else 'bullish'),
                    axis=1
                )

                merged_df['imbalance_support_resistance_direction'] = merged_df.apply(
                    lambda row: 'bearish' if row['resistance_imbalance_count'] > row['support_imbalance_count'] else 
                                ('neutral' if row['resistance_imbalance_count'] == row['support_imbalance_count'] else 'bullish'),
                    axis=1
                )

                # Add 'selling_activity' column
                merged_df['selling_activity'] = merged_df.apply(
                    lambda row: 'selling absorption' if row['Adj Close'] > row['lowest_ask_imbalance_price']
                                else ('selling initiation' if row['Adj Close'] <= row['lowest_ask_imbalance_price'] else 'neutral'),
                    axis=1
                )

                # Add 'buying_activity' column
                merged_df['buying_activity'] = merged_df.apply(
                    lambda row: 'buying absorption' if row['Adj Close'] < row['highest_bid_imbalance_price']
                                else ('buying initiation' if row['Adj Close'] >= row['highest_bid_imbalance_price'] else 'neutral'),
                    axis=1
                )

                # Adding a new column called 'activity_type'
                def determine_activity_type(row):
                    if row['Candle_Trend'] == 'Bullish' and row['candle_delta'] < 0:
                        return 'absorption'
                    elif row['Candle_Trend'] == 'Bearish' and row['candle_delta'] > 0:
                        return 'absorption'
                    else:
                        return 'neutral'

                merged_df['candle_delta_divergence_type'] = merged_df.apply(determine_activity_type, axis=1)

                st.dataframe(merged_df[['timestamp','poc','Adj Close','poc_direction','highest_ask_imbalance_price_direction','highest_bid_imbalance_price_direction','total_bid_ask_count_direction','imbalance_support_resistance_direction','buying_activity','selling_activity','candle_delta_divergence_type','Candle_Trend','candle_delta','support_imbalance_count','resistance_imbalance_count','highest_ask_imbalance_stack_price','highest_bid_imbalance_stack_price']])

                # st.dataframe(merged_df)

                st.write("Time Series DataFrame (last 20 entries):")
                st.dataframe(series_df.tail(20))

                st.write(f"Subset DataFrame (entries on {subset_date}):")
                st.dataframe(subset_df)

                # Group by timestamp and sum the buyVolume and sellVolume
                grouped_df = subset_df.groupby('timestamp').agg({
                    'buyVolume': 'sum',
                    'sellVolume': 'sum',
                    'price': ['min', 'max']
                }).reset_index()

                # Rename columns for clarity
                grouped_df.columns = ['timestamp', 'total_buy_volume', 'total_sell_volume', 'min_price', 'max_price']

                # Sort values by timestamp
                grouped_df = grouped_df.sort_values(by='timestamp', ascending=True)

                # Calculate VWAP
                def calculate_vwap(df):
                    df['cum_volume'] = df['total_buy_volume'] + df['total_sell_volume']
                    df['cum_vwap'] = (df['min_price'] * df['total_buy_volume'] + df['max_price'] * df['total_sell_volume']).cumsum() / df['cum_volume'].cumsum()
                    return df['cum_vwap']

                # Calculate Support and Resistance Levels
                def calc_support_resistance(df, window_size=5):
                    support_levels = []
                    resistance_levels = []

                    for i in range(len(df)):
                        if i < window_size:
                            support_levels.append(np.nan)
                            resistance_levels.append(np.nan)
                            continue

                        window = df.iloc[i-window_size:i]

                        # Identify support and resistance based on volume peaks and price reversals
                        buy_peak = window.loc[window['total_buy_volume'].idxmax()]
                        sell_peak = window.loc[window['total_sell_volume'].idxmax()]

                        support = buy_peak['min_price'] if buy_peak['min_price'] < df['min_price'].iloc[i] else np.nan
                        resistance = sell_peak['max_price'] if sell_peak['max_price'] > df['max_price'].iloc[i] else np.nan

                        support_levels.append(support)
                        resistance_levels.append(resistance)

                    return support_levels, resistance_levels

                # Calculate VWAP
                grouped_df['VWAP'] = calculate_vwap(grouped_df)

                # Calculate support and resistance levels
                grouped_df['Support'], grouped_df['Resistance'] = calc_support_resistance(grouped_df)

                # Plotting with Plotly
                fig = go.Figure()

                # Add price line
                fig.add_trace(go.Scatter(x=grouped_df['timestamp'], y=(grouped_df['min_price'] + grouped_df['max_price'])/2, mode='lines', name='Price'))

                # Add VWAP line
                fig.add_trace(go.Scatter(x=grouped_df['timestamp'], y=grouped_df['VWAP'], mode='lines', name='VWAP', line=dict(dash='dash')))

                # Add support levels
                fig.add_trace(go.Scatter(x=grouped_df['timestamp'], y=grouped_df['Support'], mode='markers', name='Support',
                                        marker=dict(color='green', size=5, symbol='triangle-up')))

                # Add resistance levels
                fig.add_trace(go.Scatter(x=grouped_df['timestamp'], y=grouped_df['Resistance'], mode='markers', name='Resistance',
                                        marker=dict(color='red', size=5, symbol='triangle-down')))

                # Add buy imbalance markers
                buy_imbalance = subset_df[subset_df['imbalance'] == 'buy']
                fig.add_trace(go.Scatter(x=buy_imbalance['timestamp'], y=buy_imbalance['price'], mode='markers', name='Buy Imbalance',
                                        marker=dict(color='blue', size=5, symbol='circle')))

                # Add sell imbalance markers
                sell_imbalance = subset_df[subset_df['imbalance'] == 'sell']
                fig.add_trace(go.Scatter(x=sell_imbalance['timestamp'], y=sell_imbalance['price'], mode='markers', name='Sell Imbalance',
                                        marker=dict(color='orange', size=5, symbol='circle')))

                # Update layout
                fig.update_layout(
                    title='Price with Support and Resistance Levels based on Volume and Imbalance',
                    xaxis_title='Time',
                    yaxis_title='Price',
                    xaxis=dict(
                        tickformat='%H:%M\n%b %d',
                        tickmode='linear',
                        dtick=300000  # 5 minutes in milliseconds
                    ),
                    template='plotly_dark'
                )

                # Show the plot
                st.plotly_chart(fig)

                # Additional chart for buy and sell volumes at each price point
                # Sum up the buy and sell volumes at each price point
                volume_df = subset_df.groupby(['timestamp', 'price']).agg({
                    'buyVolume': 'sum',
                    'sellVolume': 'sum'
                }).reset_index()

                # Calculate the sum of buy and sell volumes at each timestamp
                volume_sum_df = subset_df.groupby('timestamp').agg({
                    'buyVolume': 'sum',
                    'sellVolume': 'sum'
                }).reset_index()

                # Create the figure
                fig2 = go.Figure()

                # Add the price line
                fig2.add_trace(go.Scatter(x=volume_df['timestamp'], y=volume_df['price'], mode='lines', name='Price', line=dict(color='blue')))

                # Add buy volumes as green markers
                fig2.add_trace(go.Scatter(x=volume_df['timestamp'], y=volume_df['price'], mode='markers', name='Buy Volume',
                                        marker=dict(color='green', size=volume_df['buyVolume'] / 1000, symbol='circle'), opacity=0.6))

                # Add sell volumes as red markers
                fig2.add_trace(go.Scatter(x=volume_df['timestamp'], y=volume_df['price'], mode='markers', name='Sell Volume',
                                        marker=dict(color='red', size=volume_df['sellVolume'] / 1000, symbol='circle'), opacity=0.6))

                # Add secondary y-axis for volume sums
                fig2.add_trace(go.Bar(x=volume_sum_df['timestamp'], y=volume_sum_df['buyVolume'], name='Total Buy Volume',
                                    marker=dict(color='green'), yaxis='y2'))
                fig2.add_trace(go.Bar(x=volume_sum_df['timestamp'], y=volume_sum_df['sellVolume'], name='Total Sell Volume',
                                    marker=dict(color='red'), yaxis='y2'))

                # Update layout for secondary y-axis
                fig2.update_layout(
                    title='Buy and Sell Volumes at Each Price Point',
                    xaxis_title='Time',
                    yaxis_title='Price',
                    yaxis2=dict(title='Volume', overlaying='y', side='right'),
                    template='plotly_dark',
                    barmode='stack',
                    bargap=0.2
                )

                # Show the plot
                st.plotly_chart(fig2)

                # Third chart: Candlestick with highlighted candles

                # Sum up the buy and sell volumes at each price point
                volume_df = subset_df.groupby(['timestamp', 'price']).agg({
                    'buyVolume': 'sum',
                    'sellVolume': 'sum'
                }).reset_index()

                # Calculate the sum of buy and sell volumes at each timestamp
                volume_sum_df = subset_df.groupby('timestamp').agg({
                    'buyVolume': 'sum',
                    'sellVolume': 'sum'
                }).reset_index()

                temp_volume_df = volume_df
                temp_volume_df['timestamp'] = pd.to_datetime(temp_volume_df['timestamp'])

                # Group by timestamp and calculate the min and max price for each timestamp
                grouped_prices = temp_volume_df.groupby('timestamp').agg(
                    maxprice=('price', 'max'),
                    minprice=('price', 'min')
                ).reset_index()

                # Merge the grouped prices back into the original dataframe
                temp_volume_df = temp_volume_df.merge(grouped_prices, on='timestamp', how='left')

                # Function to calculate volumes at min and max prices for each timestamp
                def calculate_volumes(group):
                    min_price = group['minprice'].iloc[0]
                    max_price = group['maxprice'].iloc[0]

                    min_price_data = group[group['price'] == min_price]
                    max_price_data = group[group['price'] == max_price]

                    min_buy_volume = min_price_data['buyVolume'].sum()
                    min_sell_volume = min_price_data['sellVolume'].sum()
                    max_buy_volume = max_price_data['buyVolume'].sum()
                    max_sell_volume = max_price_data['sellVolume'].sum()

                    group['minPricebuyVolume'] = min_buy_volume
                    group['minPricesellVolume'] = min_sell_volume
                    group['maxPricebuyVolume'] = max_buy_volume
                    group['maxPricesellVolume'] = max_sell_volume

                    return group

                # Apply the function to calculate volumes for each group
                temp_volume_df = temp_volume_df.groupby('timestamp').apply(calculate_volumes).reset_index(drop=True)

                # Calculate the min and max price for each timestamp and the total buy/sell volume
                candlestick_data = temp_volume_df.groupby('timestamp').agg({
                    'price': ['min', 'max'],
                    'buyVolume': 'sum',
                    'sellVolume': 'sum',
                    'minPricebuyVolume': 'first',
                    'minPricesellVolume': 'first',
                    'maxPricebuyVolume': 'first',
                    'maxPricesellVolume': 'first'
                })

                candlestick_data.columns = [
                    'low', 'high', 'totalBuyVolume', 'totalSellVolume',
                    'MinPricebuyVolume', 'MinPricesellVolume',
                    'MaxPricebuyVolume', 'MaxPricesellVolume'
                ]
                candlestick_data.reset_index(inplace=True)

                # Create the candlestick chart
                fig3 = go.Figure()

                for i, row in candlestick_data.iterrows():
                    color = 'red' if (row['MinPricebuyVolume'] != 0) else 'blue'
                    color = 'green' if (row['MaxPricesellVolume'] != 0) else 'blue'

                    fig3.add_trace(go.Candlestick(
                        x=[row['timestamp']],
                        open=[row['low']],
                        high=[row['high']],
                        low=[row['low']],
                        close=[row['high']],
                        increasing_line_color=color,
                        decreasing_line_color=color,
                        showlegend=False
                    ))

                fig3.update_layout(
                    title='Candlestick Chart with Highlighted Candles',
                    xaxis_title='Timestamp',
                    yaxis_title='Price'
                )

                st.plotly_chart(fig3)

                # Fourth chart: Volume Clusters

                volume_df['timestamp'] = pd.to_datetime(volume_df['timestamp'])

                # Group by price to calculate total buy and sell volumes at each price level
                volume_by_price = volume_df.groupby('price').agg(
                    totalBuyVolume=('buyVolume', 'sum'),
                    totalSellVolume=('sellVolume', 'sum')
                ).reset_index()

                # Calculate the min and max price for each timestamp for the candlestick chart
                candlestick_data = volume_df.groupby('timestamp').agg(
                    open=('price', 'first'),
                    high=('price', 'max'),
                    low=('price', 'min'),
                    close=('price', 'last')
                ).reset_index()

                # Plot the candlestick chart
                fig4 = go.Figure(data=[go.Candlestick(
                    x=candlestick_data['timestamp'],
                    open=candlestick_data['open'],
                    high=candlestick_data['high'],
                    low=candlestick_data['low'],
                    close=candlestick_data['close']
                )])

                # Add buy volume clusters
                fig4.add_trace(go.Bar(
                    x=volume_by_price['totalBuyVolume'],
                    y=volume_by_price['price'],
                    orientation='h',
                    marker=dict(color='green', opacity=0.5),
                    name='Buy Volume',
                    xaxis='x2'
                ))

                # Add sell volume clusters
                fig4.add_trace(go.Bar(
                    x=volume_by_price['totalSellVolume'],
                    y=volume_by_price['price'],
                    orientation='h',
                    marker=dict(color='red', opacity=0.5),
                    name='Sell Volume',
                    xaxis='x2'
                ))

                # Update layout to include secondary x-axis and adjust the size
                fig4.update_layout(
                    title='Candlestick Chart with Volume Clusters',
                    xaxis_title='Timestamp',
                    yaxis_title='Price',
                    xaxis2=dict(title='Volume', overlaying='x', side='top'),
                    barmode='overlay',
                    width=1200,
                    height=800
                )

                st.plotly_chart(fig4)

                # Fifth chart: Candlestick with HVNs and LVNs

                volume_df['timestamp'] = pd.to_datetime(volume_df['timestamp'])

                # Group by price to calculate total buy and sell volumes at each price level
                volume_by_price = volume_df.groupby('price').agg(
                    totalBuyVolume=('buyVolume', 'sum'),
                    totalSellVolume=('sellVolume', 'sum')
                ).reset_index()

                # Identify High Volume Nodes (HVNs) and Low Volume Nodes (LVNs)
                threshold_high = volume_by_price['totalBuyVolume'].quantile(0.75)
                threshold_low = volume_by_price['totalBuyVolume'].quantile(0.25)

                hvns = volume_by_price[volume_by_price['totalBuyVolume'] >= threshold_high]
                lvns = volume_by_price[volume_by_price['totalBuyVolume'] <= threshold_low]

                # Calculate the min and max price for each timestamp for the candlestick chart
                candlestick_data = volume_df.groupby('timestamp').agg(
                    open=('price', 'first'),
                    high=('price', 'max'),
                    low=('price', 'min'),
                    close=('price', 'last')
                ).reset_index()

                # Plot the candlestick chart
                fig5 = go.Figure(data=[go.Candlestick(
                    x=candlestick_data['timestamp'],
                    open=candlestick_data['open'],
                    high=candlestick_data['high'],
                    low=candlestick_data['low'],
                    close=candlestick_data['close']
                )])

                # Add HVNs
                fig5.add_trace(go.Scatter(
                    x=hvns['totalBuyVolume'],
                    y=hvns['price'],
                    mode='markers',
                    marker=dict(color='blue', size=10),
                    name='High Volume Nodes (HVNs)',
                    xaxis='x2'
                ))

                # Add LVNs
                fig5.add_trace(go.Scatter(
                    x=lvns['totalBuyVolume'],
                    y=lvns['price'],
                    mode='markers',
                    marker=dict(color='yellow', size=10),
                    name='Low Volume Nodes (LVNs)',
                    xaxis='x2'
                ))

                # Add support and resistance lines based on HVNs
                for price in hvns['price']:
                    fig5.add_hline(y=price, line=dict(color='green', dash='dash'), name=f'Resistance {price}')

                # Add potential breakout/breakdown zones based on LVNs
                for price in lvns['price']:
                    fig5.add_hline(y=price, line=dict(color='red', dash='dash'), name=f'Breakout/Breakdown {price}')

                # Update layout to include secondary x-axis and adjust the size
                fig5.update_layout(
                    title='Candlestick Chart with HVNs, LVNs, Support and Resistance',
                    xaxis_title='Timestamp',
                    yaxis_title='Price',
                    xaxis2=dict(title='Volume', overlaying='x', side='top'),
                    barmode='overlay',
                    width=1200,
                    height=800
                )

                st.plotly_chart(fig5)

                # Sixth chart: Delta (Buy Volume - Sell Volume) by Price and Timestamp

                volume_df['timestamp'] = pd.to_datetime(volume_df['timestamp'])

                # Calculate delta (buy volume - sell volume) for each price level
                volume_df['delta'] = volume_df['buyVolume'] - volume_df['sellVolume']

                # Group by timestamp and price to calculate total delta at each price level
                delta_by_price = volume_df.groupby(['timestamp', 'price']).agg(
                    totalDelta=('delta', 'sum')
                ).reset_index()

                # Plot delta with x axis as timestamp, y axis as price and delta at those prices
                fig6 = go.Figure(data=go.Heatmap(
                    x=delta_by_price['timestamp'],
                    y=delta_by_price['price'],
                    z=delta_by_price['totalDelta'],
                    colorscale='RdYlGn',
                    colorbar=dict(title='Delta')
                ))

                # Update layout to adjust the size
                fig6.update_layout(
                    title='Delta (Buy Volume - Sell Volume) by Price and Timestamp',
                    xaxis_title='Timestamp',
                    yaxis_title='Price',
                    width=1200,
                    height=800
                )

                st.plotly_chart(fig6)

                # Seventh chart: Candlestick Chart with Top 5 Market Absorptions

                volume_df['timestamp'] = pd.to_datetime(volume_df['timestamp'])

                # Calculate the min and max price for each timestamp
                candlestick_data = volume_df.groupby('timestamp').agg({'price': ['min', 'max']})
                candlestick_data.columns = ['low', 'high']
                candlestick_data.reset_index(inplace=True)

                # Calculate total volume (buyVolume + sellVolume) for each price level
                volume_df['totalVolume'] = volume_df['buyVolume'] + volume_df['sellVolume']

                # Identify the top 5 prices with the highest market absorption (totalVolume) for the entire day
                top_absorptions = volume_df.nlargest(5, 'totalVolume')

                # Create the candlestick chart
                fig7 = go.Figure(data=[go.Candlestick(
                    x=candlestick_data['timestamp'],
                    low=candlestick_data['low'],
                    high=candlestick_data['high'],
                    open=candlestick_data['low'],
                    close=candlestick_data['high']
                )])

                # Plot top 5 market absorptions as lines
                for _, row in top_absorptions.iterrows():
                    fig7.add_shape(
                        type="line",
                        x0=row['timestamp'],
                        y0=row['price'],
                        x1=row['timestamp'] + pd.Timedelta(minutes=30),  # Adjust the end point as needed
                        y1=row['price'],
                        line=dict(color="Purple", width=2),
                        name='Top Market Absorption'
                    )

                fig7.update_layout(
                    title='Candlestick Chart with Top 5 Market Absorptions',
                    xaxis_title='Timestamp',
                    yaxis_title='Price',
                    width=1200,
                    height=800
                )

                st.plotly_chart(fig7)

                # Eighth chart: Aggressive Orders + Delta Setup

                volume_df['timestamp'] = pd.to_datetime(volume_df['timestamp'])
                volume_df['delta'] = volume_df['buyVolume'] - volume_df['sellVolume']

                # Step 1: Identify strong Support/Resistance zones (for simplicity, we'll set static S/R zones)
                support_zone = 250.59
                resistance_zone = 253.40

                # Step 2: Analyze Order Flow data to identify aggressive orders near S/R zones
                aggressive_orders = volume_df[
                    ((volume_df['price'] <= support_zone) & (volume_df['buyVolume'] > volume_df['sellVolume'])) |
                    ((volume_df['price'] >= resistance_zone) & (volume_df['sellVolume'] > volume_df['buyVolume']))
                ]

                # Step 3: Confirm trades with Delta values
                trades = []
                for _, row in aggressive_orders.iterrows():
                    if row['price'] <= support_zone and row['delta'] > 0:
                        trades.append((row['timestamp'], row['price'], 'Long'))
                    elif row['price'] >= resistance_zone and row['delta'] < 0:
                        trades.append((row['timestamp'], row['price'], 'Short'))

                trades_df = pd.DataFrame(trades, columns=['timestamp', 'price', 'direction'])

                # Create the candlestick chart
                candlestick_data = volume_df.groupby('timestamp').agg({'price': ['min', 'max']})
                candlestick_data.columns = ['low', 'high']
                candlestick_data.reset_index(inplace=True)

                fig8 = go.Figure(data=[go.Candlestick(
                    x=candlestick_data['timestamp'],
                    low=candlestick_data['low'],
                    high=candlestick_data['high'],
                    open=candlestick_data['low'],
                    close=candlestick_data['high']
                )])

                # Add Support/Resistance zones
                fig8.add_shape(
                    type="rect",
                    x0=volume_df['timestamp'].min(),
                    y0=support_zone - 0.05,
                    x1=volume_df['timestamp'].max(),
                    y1=support_zone + 0.05,
                    fillcolor="Green",
                    opacity=0.2,
                    line_width=0,
                    name='Support Zone'
                )
                fig8.add_shape(
                    type="rect",
                    x0=volume_df['timestamp'].min(),
                    y0=resistance_zone - 0.05,
                    x1=volume_df['timestamp'].max(),
                    y1=resistance_zone + 0.05,
                    fillcolor="Red",
                    opacity=0.2,
                    line_width=0,
                    name='Resistance Zone'
                )

                # Add trades to the chart
                for _, trade in trades_df.iterrows():
                    color = 'green' if trade['direction'] == 'Long' else 'red'
                    fig8.add_trace(go.Scatter(
                        x=[trade['timestamp']],
                        y=[trade['price']],
                        mode='markers+text',
                        marker=dict(color=color, size=10),
                        text=trade['direction'],
                        textposition='top center',
                        name=trade['direction']
                    ))

                fig8.update_layout(
                    title='Aggressive Orders + Delta Setup',
                    xaxis_title='Timestamp',
                    yaxis_title='Price',
                    width=1200,
                    height=800
                )

                st.plotly_chart(fig8)

                # Ninth chart: Cumulative Delta Confirmation Setup

                volume_df['timestamp'] = pd.to_datetime(volume_df['timestamp'])
                volume_df['delta'] = volume_df['buyVolume'] - volume_df['sellVolume']
                volume_df['cumDelta'] = volume_df['delta'].cumsum()

                # Step 1: Identify strong Support/Resistance zones (for simplicity, we'll set static S/R zones)
                support_zone = 253.10
                resistance_zone = 253.50

                # Step 2: Create a Cumulative Delta line chart
                cum_delta_trace = go.Scatter(
                    x=volume_df['timestamp'],
                    y=volume_df['cumDelta'],
                    mode='lines',
                    name='Cumulative Delta',
                    line=dict(color='blue')
                )

                # Step 3: Identify divergences between Price and Cum. Delta
                divergences = []
                for i in range(1, len(volume_df)):
                    if volume_df['price'].iloc[i] > volume_df['price'].iloc[i-1] and volume_df['cumDelta'].iloc[i] < volume_df['cumDelta'].iloc[i-1]:
                        divergences.append((volume_df['timestamp'].iloc[i], volume_df['price'].iloc[i], 'Short'))
                    elif volume_df['price'].iloc[i] < volume_df['price'].iloc[i-1] and volume_df['cumDelta'].iloc[i] > volume_df['cumDelta'].iloc[i-1]:
                        divergences.append((volume_df['timestamp'].iloc[i], volume_df['price'].iloc[i], 'Long'))

                divergences_df = pd.DataFrame(divergences, columns=['timestamp', 'price', 'direction'])

                # Create the candlestick chart
                candlestick_data = volume_df.groupby('timestamp').agg({'price': ['min', 'max']})
                candlestick_data.columns = ['low', 'high']
                candlestick_data.reset_index(inplace=True)

                fig9 = go.Figure(data=[go.Candlestick(
                    x=candlestick_data['timestamp'],
                    low=candlestick_data['low'],
                    high=candlestick_data['high'],
                    open=candlestick_data['low'],
                    close=candlestick_data['high'],
                    name='Price'
                )])

                # Add Cumulative Delta line chart
                fig9.add_trace(cum_delta_trace)

                # Add Support/Resistance zones
                fig9.add_shape(
                    type="rect",
                    x0=volume_df['timestamp'].min(),
                    y0=support_zone - 0.05,
                    x1=volume_df['timestamp'].max(),
                    y1=support_zone + 0.05,
                    fillcolor="Green",
                    opacity=0.2,
                    line_width=0,
                    name='Support Zone'
                )
                fig9.add_shape(
                    type="rect",
                    x0=volume_df['timestamp'].min(),
                    y0=resistance_zone - 0.05,
                    x1=volume_df['timestamp'].max(),
                    y1=resistance_zone + 0.05,
                    fillcolor="Red",
                    opacity=0.2,
                    line_width=0,
                    name='Resistance Zone'
                )

                # Add divergences to the chart
                for _, divergence in divergences_df.iterrows():
                    color = 'green' if divergence['direction'] == 'Long' else 'red'
                    fig9.add_trace(go.Scatter(
                        x=[divergence['timestamp']],
                        y=[divergence['price']],
                        mode='markers+text',
                        marker=dict(color=color, size=10),
                        text=divergence['direction'],
                        textposition='top center',
                        name=divergence['direction']
                    ))

                fig9.update_layout(
                    title='Cumulative Delta Confirmation Setup',
                    xaxis_title='Timestamp',
                    yaxis_title='Price',
                    width=1200,
                    height=800
                )

                st.plotly_chart(fig9)

                # Tenth chart: Volume Profile Shape Identification

                volume_df['timestamp'] = pd.to_datetime(volume_df['timestamp'])
                volume_df['totalVolume'] = volume_df['buyVolume'] + volume_df['sellVolume']

                # Group by price to get the total volume for each price level
                volume_profile = volume_df.groupby('price').agg(totalVolume=('totalVolume', 'sum')).reset_index()

                # Identify peaks in the volume profile
                volumes = volume_profile['totalVolume'].values
                prices = volume_profile['price'].values

                peaks, _ = find_peaks(volumes, distance=1)

                # Determine the shape
                shape = ""
                if len(peaks) == 1:
                    shape = "I-shape"
                elif len(peaks) == 2:
                    shape = "B-shape"
                else:
                    if peaks[0] < len(volumes) / 3:
                        shape = "P-shape"
                    elif peaks[-1] > 2 * len(volumes) / 3:
                        shape = "B-shape"
                    else:
                        shape = "D-shape"

                # Create the volume profile bar chart
                fig10 = go.Figure()
                fig10.add_trace(go.Bar(
                    x=volume_profile['totalVolume'],
                    y=volume_profile['price'],
                    orientation='h',
                    marker=dict(color='blue'),
                    name='Volume Profile'
                ))

                # Add peaks to the chart
                fig10.add_trace(go.Scatter(
                    x=volumes[peaks],
                    y=prices[peaks],
                    mode='markers',
                    marker=dict(color='red', size=10),
                    name='Peaks'
                ))

                fig10.update_layout(
                    title=f'Volume Profile Shape: {shape}',
                    xaxis_title='Volume',
                    yaxis_title='Price',
                    width=800,
                    height=600
                )

                st.plotly_chart(fig10)

            except json.JSONDecodeError:
                st.write("Error: The uploaded file does not contain valid JSON data.")
            except KeyError as e:
                st.write(f"Error: Key {e} not found in the JSON data.")
        else:
            st.write("Please upload a file before submitting.")

elif tab == "Headers and Cookies":
    # Input boxes for the headers and cookies
    xsrf_token = st.text_input("Enter X-Xsrf-Token:", st.session_state.xsrf_token)
    laravel_token = st.text_input("Enter laravel_token:", st.session_state.laravel_token)
    xsrf_cookie = st.text_input("Enter XSRF-TOKEN:", st.session_state.xsrf_cookie)
    laravel_session = st.text_input("Enter laravel_session:", st.session_state.laravel_session)
    
    # Button to update the headers and cookies
    if st.button("Update Headers and Cookies"):
        st.session_state.xsrf_token = xsrf_token
        st.session_state.laravel_token = laravel_token
        st.session_state.xsrf_cookie = xsrf_cookie
        st.session_state.laravel_session = laravel_session

elif tab == "Chat Interface":

    
    default_date = datetime.today().date()
    input_date = st.date_input("Start Date", value=default_date)

    # Fetch stock data in real-time
    def fetch_stock_data(ticker, start, interval='1m'):
        stock_data = yf.download(ticker, start=start, interval=interval)
        return stock_data

    data = fetch_stock_data(ticker, input_date)

    temp_data = data.copy()

    # Ensure the DataFrame index is a DatetimeIndex for VWAP calculations
    temp_data.reset_index(inplace=True)  # Reset index for column access
    temp_data.set_index(pd.DatetimeIndex(temp_data["Datetime"]), inplace=True)  # Use 'Datetime' as the index

    # Function to calculate Cumulative Volume Delta (CVD)
    def calculate_cvd(data):
        """
        Calculate the Cumulative Volume Delta (CVD) and additional metrics.
        """
        data['delta'] = data['Close'] - data['Open']  # Price delta
        data['buy_volume'] = data['Volume'] * (data['delta'] > 0).astype(int)
        data['sell_volume'] = data['Volume'] * (data['delta'] < 0).astype(int)
        data['cvd'] = (data['buy_volume'] - data['sell_volume']).cumsum()
        return data

    # Function to identify support and resistance levels
    def identify_support_resistance(data, start_time, end_time):
        """
        Identify support (most selling) and resistance (most buying) levels for a given time range.
        """
        time_frame = data.between_time(start_time, end_time).copy()
        time_frame = calculate_cvd(time_frame)
        
        if time_frame.empty:
            return {}

        # Support: Price level with most selling (most negative CVD)
        support_idx = time_frame['cvd'].idxmin()
        support_level = time_frame.loc[support_idx, 'Close']
        support_time = support_idx

        # Resistance: Price level with most buying (most positive CVD)
        resistance_idx = time_frame['cvd'].idxmax()
        resistance_level = time_frame.loc[resistance_idx, 'Close']
        resistance_time = resistance_idx

        return {
            "support_level": round(support_level, 2),
            "support_time": support_time.tz_localize(None).strftime('%Y-%m-%d %H:%M:%S'),
            "resistance_level": round(resistance_level, 2),
            "resistance_time": resistance_time.tz_localize(None).strftime('%Y-%m-%d %H:%M:%S'),
        }

    # Calculate CVD for the 09:30-16:00 timeframe
    cvd_data = temp_data.between_time("09:30", "16:00").copy()
    cvd_data = calculate_cvd(cvd_data)

    # Identify support and resistance for the 09:30-10:30 timeframe
    support_resistance_stats = identify_support_resistance(temp_data, "09:30", "10:30")
    support_level = support_resistance_stats["support_level"]
    support_time = support_resistance_stats["support_time"]
    resistance_level = support_resistance_stats["resistance_level"]
    resistance_time = support_resistance_stats["resistance_time"]

    # # Adding Buy/Sell signals to the data
    # cvd_data['signal'] = None
    # cvd_data['signal_type'] = None

    # # Logic for Buy/Sell signals
    # cvd_data['signal'] = cvd_data.apply(
    #     lambda row: row['Close'] if (row['Close'] > resistance_level and row['cvd'] > cvd_data.loc[resistance_time, 'cvd']) else (
    #         row['Close'] if (row['Close'] < support_level and row['cvd'] < cvd_data.loc[support_time, 'cvd']) else None),
    #     axis=1
    # )

    # cvd_data['signal_type'] = cvd_data.apply(
    #     lambda row: 'Buy' if (row['Close'] > resistance_level and row['cvd'] > cvd_data.loc[resistance_time, 'cvd']) else (
    #         'Sell' if (row['Close'] < support_level and row['cvd'] < cvd_data.loc[support_time, 'cvd']) else None),
    #     axis=1
    # )

    # Identify the first Buy signal
    first_buy_signal = cvd_data[(cvd_data['Close'] > resistance_level) & 
                                (cvd_data['cvd'] > cvd_data.loc[resistance_time, 'cvd'])].iloc[:1]

    # Identify the first Sell signal
    first_sell_signal = cvd_data[(cvd_data['Close'] < support_level) & 
                                (cvd_data['cvd'] < cvd_data.loc[support_time, 'cvd'])].iloc[:1]

    # Add first Buy and Sell timestamps if available
    first_buy_time = first_buy_signal.index[0].strftime('%Y-%m-%d %H:%M:%S') if not first_buy_signal.empty else "N/A"
    first_sell_time = first_sell_signal.index[0].strftime('%Y-%m-%d %H:%M:%S') if not first_sell_signal.empty else "N/A"

    st.write("Buy Time: ", first_buy_time)
    st.write("Sell Time: ", first_sell_time)

    # Update hovertext to include first Buy/Sell timestamps
    cvd_data['hovertext'] = (
        "Time: " + cvd_data.index.strftime('%Y-%m-%d %H:%M:%S') +
        "<br>Open: " + round(cvd_data['Open'], 2).astype(str) +
        "<br>High: " + round(cvd_data['High'], 2).astype(str) +
        "<br>Low: " + round(cvd_data['Low'], 2).astype(str) +
        "<br>Close: " + round(cvd_data['Close'], 2).astype(str) +
        "<br>CVD: " + round(cvd_data['cvd'], 2).astype(str) +
        f"<br>Support Level: {support_level}" +
        f"<br>Support Time: {support_time}" +
        f"<br>Resistance Level: {resistance_level}" +
        f"<br>Resistance Time: {resistance_time}" +
        f"<br>First Buy Time: {first_buy_time}" +
        f"<br>First Sell Time: {first_sell_time}"
    )

    # Create the candlestick chart with CVD
    fig = go.Figure()

    # Add candlestick trace
    fig.add_trace(go.Candlestick(
        x=cvd_data.index,
        open=cvd_data['Open'],
        high=cvd_data['High'],
        low=cvd_data['Low'],
        close=cvd_data['Close'],
        name='Candlestick',
        hovertext=cvd_data['hovertext'],
        hoverinfo='text'
    ))

    # Add CVD as a line trace on a secondary y-axis
    fig.add_trace(go.Scatter(
        x=cvd_data.index,
        y=cvd_data['cvd'],
        mode='lines',
        name='Cumulative Volume Delta (CVD)',
        line=dict(color='orange'),
        yaxis='y2',  # Use secondary y-axis
    ))

    # Add support line
    fig.add_shape(
        type="line",
        x0=cvd_data.index.min(),
        x1=cvd_data.index.max(),
        y0=support_level,
        y1=support_level,
        line=dict(color="blue", dash="dot"),
        name="Support Level",
    )

    # Add resistance line
    fig.add_shape(
        type="line",
        x0=cvd_data.index.min(),
        x1=cvd_data.index.max(),
        y0=resistance_level,
        y1=resistance_level,
        line=dict(color="red", dash="dot"),
        name="Resistance Level",
    )

    # # # Update layout to include a secondary y-axis for CVD
    # # fig.update_layout(
    # #     title="Candlestick Chart with CVD (09:30-16:00) and Support/Resistance (09:30-10:30)",
    # #     xaxis_title="Time",
    # #     yaxis_title="Price",
    # #     yaxis2=dict(
    # #         title="CVD",
    # #         overlaying='y',
    # #         side='right'
    # #     ),
    # #     template="plotly_dark",
    # #     hovermode="x unified"
    # # )

    # # Adding Buy signals (triangle-up)
    # fig.add_trace(go.Scatter(
    #     x=cvd_data[cvd_data['signal_type'] == 'Buy'].index,
    #     y=cvd_data[cvd_data['signal_type'] == 'Buy']['signal'],
    #     mode='markers',
    #     name='Buy Signal',
    #     marker=dict(symbol='triangle-up', color='green', size=10),
    #     hoverinfo='text',
    #     hovertext="Buy Signal"
    # ))

    # # Adding Sell signals (triangle-down)
    # fig.add_trace(go.Scatter(
    #     x=cvd_data[cvd_data['signal_type'] == 'Sell'].index,
    #     y=cvd_data[cvd_data['signal_type'] == 'Sell']['signal'],
    #     mode='markers',
    #     name='Sell Signal',
    #     marker=dict(symbol='triangle-down', color='red', size=10),
    #     hoverinfo='text',
    #     hovertext="Sell Signal"
    # ))

    # # Update layout to include the signals
    # fig.update_layout(
    #     title="Candlestick Chart with CVD and Buy/Sell Signals",
    #     xaxis_title="Time",
    #     yaxis_title="Price",
    #     yaxis2=dict(
    #         title="CVD",
    #         overlaying='y',
    #         side='right'
    #     ),
    #     template="plotly_dark",
    #     hovermode="x unified"
    # )

    # Add Buy signal (triangle-up) to the chart
    if not first_buy_signal.empty:
        fig.add_trace(go.Scatter(
            x=first_buy_signal.index,
            y=first_buy_signal['Close'],
            mode='markers',
            name='First Buy Signal',
            marker=dict(symbol='triangle-up', color='green', size=10),
            hoverinfo='text',
            hovertext="First Buy Signal"
        ))

    # Add Sell signal (triangle-down) to the chart
    if not first_sell_signal.empty:
        fig.add_trace(go.Scatter(
            x=first_sell_signal.index,
            y=first_sell_signal['Close'],
            mode='markers',
            name='First Sell Signal',
            marker=dict(symbol='triangle-down', color='red', size=10),
            hoverinfo='text',
            hovertext="First Sell Signal"
        ))

    
    # Update layout to include the filtered signals
    fig.update_layout(
        title="Candlestick Chart with CVD and First Buy/Sell Signals",
        xaxis_title="Time",
        yaxis_title="Price",
        yaxis2=dict(
            title="CVD",
            overlaying='y',
            side='right'
        ),
        template="plotly_dark",
        hovermode="x unified"
    )


    # Display the chart in Streamlit
    st.plotly_chart(fig)

    
    

    # Calculate the volume profile with buy and sell volumes
    def calculate_volume_profile(data, row_layout):
        price_min = data['Low'].min()
        price_max = data['High'].max()

        bins = row_layout
        bin_edges = np.linspace(price_min, price_max, bins)

        volume_profile = pd.DataFrame(index=bin_edges[:-1], columns=['Total Volume'])
        volume_profile['Total Volume'] = 0

        for index, row in data.iterrows():
            bin_indices = np.digitize([row['Low'], row['High']], bin_edges) - 1
            bin_indices = [max(0, min(bins-2, b)) for b in bin_indices]

            volume_profile.iloc[bin_indices[0]:bin_indices[1] + 1, volume_profile.columns.get_loc('Total Volume')] += row['Volume']

        return volume_profile

    # Function to calculate VAH, VAL, POC
    def calculate_vah_val_poc(volume_profile):
        total_volume = volume_profile['Total Volume'].sum()
        cumulative_volume = volume_profile['Total Volume'].cumsum()

        poc = volume_profile['Total Volume'].idxmax()  # Price level with highest volume (POC)

        vah_threshold = 0.7 * total_volume
        val_threshold = 0.3 * total_volume

        vah = volume_profile.index[cumulative_volume >= vah_threshold].min()
        val = volume_profile.index[cumulative_volume <= val_threshold].max()

        return vah, val, poc

    # Initial quick identification of market profile shape based on POC, VAH, and VAL
    def quick_identify_profile_shape(vah, val, poc):
        if poc > vah:
            return "P-shape (Bullish Accumulation)"
        elif poc < val:
            return "b-shape (Bearish Accumulation)"
        elif vah > poc > val:
            return "D-shape (Balanced Market)"
        else:
            return "B-shape (Double Distribution)"

    # Refine the initial guess with skewness and kurtosis
    def refine_with_skew_kurtosis(volume_profile, shape_guess):
        volumes = volume_profile['Total Volume'].values
        skewness = skew(volumes)
        kurt = kurtosis(volumes)

        if shape_guess == "P-shape" and skewness < 0:
            return "b-shape (Bearish Accumulation)"
        if shape_guess == "b-shape" and skewness > 0:
            return "P-shape (Bullish Accumulation)"

        if shape_guess == "D-shape" and abs(skewness) > 0.5 and kurt > 0:
            return "B-shape (Double Distribution)"

        return shape_guess

    # Calculate the volume profile
    volume_profile = calculate_volume_profile(data, row_layout=24)
    vah, val, poc = calculate_vah_val_poc(volume_profile)

    # Initial shape identification
    initial_shape = quick_identify_profile_shape(vah, val, poc)

    # Refined shape identification
    refined_shape = refine_with_skew_kurtosis(volume_profile, initial_shape)

    # Display the initial and refined market shapes
    st.write(f"Initial Market Profile Shape: {initial_shape}")
    st.write(f"Refined Market Profile Shape: {refined_shape}")

    # Plot the volume profile and VAH
    def plot_volume_profile(volume_profile, vah, val, poc):
        fig = go.Figure()

        fig.add_trace(go.Bar(
            y=volume_profile.index,
            x=volume_profile['Total Volume'],
            orientation='h',
            name='Total Volume',
            marker=dict(color='blue', opacity=0.6)
        ))

        # Highlight VAH, VAL, and POC
        fig.add_shape(type="line", y0=vah, y1=vah, x0=0, x1=1, line=dict(color="green", dash="dash"))
        fig.add_shape(type="line", y0=val, y1=val, x0=0, x1=1, line=dict(color="red", dash="dash"))
        fig.add_shape(type="line", y0=poc, y1=poc, x0=0, x1=1, line=dict(color="orange", dash="dash"))

        # Add annotations for VAH, VAL, and POC
        fig.add_annotation(xref="paper", yref="y", x=1, y=vah, text=f"VAH at {vah:.2f}", showarrow=False)
        fig.add_annotation(xref="paper", yref="y", x=1, y=val, text=f"VAL at {val:.2f}", showarrow=False)
        fig.add_annotation(xref="paper", yref="y", x=1, y=poc, text=f"POC at {poc:.2f}", showarrow=False)

        fig.update_layout(title='Volume Profile with Initial and Refined Market Shape Detection', xaxis_title='Volume', yaxis_title='Price')
        st.plotly_chart(fig)

    plot_volume_profile(volume_profile, vah, val, poc)

    # # Section 2: 5-Minute Stock Prices for the selected date
    # st.title("5-Minute Stock Price Data for Selected Date")

    # def fetch_five_minute_data(ticker, selected_date):
    #     start_date_str = selected_date.strftime("%Y-%m-%d")
    #     data = yf.download(ticker, start=start_date_str, end=start_date_str, interval="5m")
    #     return data

    # five_min_data = fetch_five_minute_data(ticker, start_date)

    # if not five_min_data.empty:
    #     five_min_data = five_min_data.reset_index()
    #     st.write("5-Minute Interval Data", five_min_data)
    # else:
    #     st.write("No 5-minute data available for the selected date.")

    # # Section 3: 30-Minute Data Table for the selected date
    # st.title("30-Minute Data Table for Selected Date")

    # def fetch_thirty_minute_data(ticker, selected_date):
    #     start_date_str = selected_date.strftime("%Y-%m-%d")
    #     data = yf.download(ticker, start=start_date_str, end=start_date_str, interval="30m")
    #     return data

    # thirty_min_data = fetch_thirty_minute_data(ticker, start_date)

    # if not thirty_min_data.empty:
    #     thirty_min_data = thirty_min_data.reset_index()
    #     st.write("30-Minute Interval Data", thirty_min_data)
    # else:
    #     st.write("No 30-minute data available for the selected date.")

    # # Section 4: IB Range Signal and Last Day VAL Signal
    # st.title("IB Range and Last Day's VAL Signal")

    # # Generate a signal for IB Range for today based on mock conditions
    # ib_range_signal = "IB Range Signal: Small" if thirty_min_data['High'].iloc[0] - thirty_min_data['Low'].iloc[0] < 2 else "IB Range Signal: Large"
    # st.write(ib_range_signal)

    # # Mock signal based on the previous day's VAL
    # val_signal = "Last Day's VAL Signal: Bullish" if vah > val else "Last Day's VAL Signal: Bearish"
    # st.write(val_signal)

    # Section 1: Fetch stock data in real-time
    # @st.cache_data
    def fetch_stock_data(ticker, interval='30m'):
        try:
            data = yf.download(ticker, period="60d", interval=interval)
            if data.empty:
                st.warning(f"No data found for {ticker} with {interval} interval. Please try a different date.")
            return data.reset_index()
        except Exception as e:
            st.error(f"Error fetching data for {ticker}: {e}")
            return pd.DataFrame()

    data = fetch_stock_data(ticker)

    dji_data = fetch_stock_data('^DJI')
    nasdaq_data = fetch_stock_data('^IXIC')
    s_and_p_500_data = fetch_stock_data('^GSPC')

    # Calculate percentage changes in 'Close' prices
    data['pct_change'] = data['Close'].pct_change()
    dji_data['DJI_pct_change'] = dji_data['Close'].pct_change()
    nasdaq_data['NASDAQ_pct_change'] = nasdaq_data['Close'].pct_change()
    s_and_p_500_data['S_AND_P_500_pct_change'] = s_and_p_500_data['Close'].pct_change()

    # Align all dataframes to match by timestamp
    merged_data = pd.merge(data[['Datetime', 'pct_change']],
                        dji_data[['Datetime', 'DJI_pct_change']],
                        on='Datetime', how='inner')

    merged_data = pd.merge(merged_data,
                        nasdaq_data[['Datetime', 'NASDAQ_pct_change']],
                        on='Datetime', how='inner')

    merged_data = pd.merge(merged_data,
                        s_and_p_500_data[['Datetime', 'S_AND_P_500_pct_change']],
                        on='Datetime', how='inner')

    # Drop NaN values resulting from the percentage change calculation
    merged_data.dropna(inplace=True)

    # Calculate correlations
    correlation_dji = merged_data['pct_change'].corr(merged_data['DJI_pct_change'])
    correlation_nasdaq = merged_data['pct_change'].corr(merged_data['NASDAQ_pct_change'])
    correlation_s_and_p_500 = merged_data['pct_change'].corr(merged_data['S_AND_P_500_pct_change'])

    # Display correlations using Streamlit
    st.write(f"Correlation between {ticker} and DJI movements: {correlation_dji:.2f}")
    st.write(f"Correlation between {ticker} and NASDAQ movements: {correlation_nasdaq:.2f}")
    st.write(f"Correlation between {ticker} and S&P 500 movements: {correlation_s_and_p_500:.2f}")

    temp_data = data.copy()


    # Define the main function
    def calculate_indicators_and_scores(temp_data):
        # Helper Functions
        def interpolate(value, value_high, value_low, range_high, range_low):
            """Interpolate a value to a new range."""
            return range_low + (value - value_low) * (range_high - range_low) / (value_high - value_low)

        def calculate_bbp_score(bbp, upper, lower):
            if bbp > upper:
                if bbp > 1.5 * upper:
                    return 100
                return interpolate(bbp, 1.5 * upper, upper, 100, 75)
            elif bbp > 0:
                return interpolate(bbp, upper, 0, 75, 50)
            elif bbp < lower:
                if bbp < 1.5 * lower:
                    return 0
                return interpolate(bbp, lower, 1.5 * lower, 25, 0)
            elif bbp < 0:
                return interpolate(bbp, 0, lower, 50, 25)
            else:
                return 50

        def normalize(buy, sell, smooth, close_values):
            os = 0
            max_val = None
            min_val = None
            normalized_values = []

            for i in range(len(close_values)):
                previous_os = os

                if buy[i]:
                    os = 1
                elif sell[i]:
                    os = -1

                if os > previous_os:
                    max_val = close_values[i]
                elif os < previous_os:
                    pass
                else:
                    max_val = max(close_values[i], max_val) if max_val is not None else close_values[i]

                if os < previous_os:
                    min_val = close_values[i]
                elif os > previous_os:
                    pass
                else:
                    min_val = min(close_values[i], min_val) if min_val is not None else close_values[i]

                if max_val is not None and min_val is not None and max_val != min_val:
                    normalized_value = (close_values[i] - min_val) / (max_val - min_val) * 100
                else:
                    normalized_value = 0
                normalized_values.append(normalized_value)

            normalized_values_np = np.array(normalized_values, dtype=float)
            smoothed_values = talib.SMA(normalized_values_np, timeperiod=smooth)
            return smoothed_values.tolist()

        def moving_average_value(source, length, ma_type):
            if ma_type == "SMA":
                return talib.SMA(source, timeperiod=length)
            elif ma_type == "EMA":
                return talib.EMA(source, timeperiod=length)
            elif ma_type == "HMA":
                half_length = int(length / 2)
                sqrt_length = int(np.sqrt(length))
                return talib.WMA(2 * talib.WMA(source, timeperiod=half_length) - talib.WMA(source, timeperiod=length), timeperiod=sqrt_length)
            elif ma_type == "WMA":
                return talib.WMA(source, timeperiod=length)
            elif ma_type == "VWMA":
                return (source * temp_data["Volume"]).rolling(window=length).sum() / temp_data["Volume"].rolling(window=length).sum()
            else:
                raise ValueError(f"Unsupported MA type: {ma_type}")

        def linear_regression_score(close_values, lr_length):
            lr_scores = []
            for i in range(len(close_values)):
                if i < lr_length:
                    lr_scores.append(np.nan)
                else:
                    y = close_values[i - lr_length + 1 : i + 1]
                    x = np.arange(len(y))
                    slope, intercept, r_value, _, _ = linregress(x, y)
                    lr_scores.append(50 * r_value + 50)
            return lr_scores

        # Indicator Calculations
        # 1. RSI
        rsi_length = 14
        temp_data["RSI"] = talib.RSI(temp_data["Close"], timeperiod=rsi_length)
        temp_data["RSI_Score"] = temp_data["RSI"].apply(lambda x: 
            interpolate(x, 100, 70, 100, 75) if x > 70 else
            interpolate(x, 70, 50, 75, 50) if x > 50 else
            interpolate(x, 50, 30, 50, 25) if x > 30 else
            interpolate(x, 30, 0, 25, 0))

        # 2. Stochastic Oscillator (%K)
        stoch_length_k = 14
        stoch_smoothing_k = 3
        temp_data["%K"], _ = talib.STOCH(
            temp_data["High"], temp_data["Low"], temp_data["Close"],
            fastk_period=stoch_length_k, slowk_period=stoch_smoothing_k, slowk_matype=0,
            slowd_period=stoch_smoothing_k, slowd_matype=0
        )
        temp_data["%K_Score"] = temp_data["%K"].apply(lambda x: 
            interpolate(x, 100, 80, 100, 75) if x > 80 else
            interpolate(x, 80, 50, 75, 50) if x > 50 else
            interpolate(x, 50, 20, 50, 25) if x > 20 else
            interpolate(x, 20, 0, 25, 0))

        # 3. Stochastic RSI
        stoch_rsi_length = 14
        stoch_rsi_smoothing_k = 3
        rsi_values = talib.RSI(temp_data["Close"], timeperiod=stoch_rsi_length)
        stoch_rsi_high = rsi_values.rolling(window=stoch_rsi_length).max()
        stoch_rsi_low = rsi_values.rolling(window=stoch_rsi_length).min()
        temp_data["Stoch_RSI_K"] = 100 * (rsi_values - stoch_rsi_low) / (stoch_rsi_high - stoch_rsi_low)
        temp_data["Stoch_RSI_K"] = temp_data["Stoch_RSI_K"].rolling(window=stoch_rsi_smoothing_k, min_periods=1).mean()
        temp_data["Stoch_RSI_Score"] = temp_data["Stoch_RSI_K"].apply(lambda x: 
            interpolate(x, 100, 80, 100, 75) if x > 80 else
            interpolate(x, 80, 50, 75, 50) if x > 50 else
            interpolate(x, 50, 20, 50, 25) if x > 20 else
            interpolate(x, 20, 0, 25, 0))

        # 4. Commodity Channel Index (CCI)
        cci_length = 20
        temp_data["CCI"] = talib.CCI(temp_data["High"], temp_data["Low"], temp_data["Close"], timeperiod=cci_length)
        temp_data["CCI_Score"] = temp_data["CCI"].apply(lambda x: 
            interpolate(x, 300, 100, 100, 75) if x > 100 else
            interpolate(x, 100, 0, 75, 50) if x >= 0 else
            interpolate(x, -100, -300, 25, 0) if x < -100 else
            interpolate(x, 0, -100, 50, 25))

        # 5. Bull-Bear Power (BBP)
        sma_length = 13
        sma_13 = talib.SMA(temp_data["Close"], timeperiod=sma_length)
        temp_data["BBP"] = (temp_data["High"] + temp_data["Low"]) - 2 * sma_13
        bbp_std = temp_data["BBP"].rolling(window=sma_length).std()
        bbp_sma = temp_data["BBP"].rolling(window=sma_length).mean()
        temp_data["BBP_Upper"] = bbp_sma + 2 * bbp_std
        temp_data["BBP_Lower"] = bbp_sma - 2 * bbp_std
        temp_data["BBP_Score"] = temp_data.apply(
            lambda row: calculate_bbp_score(row["BBP"], row["BBP_Upper"], row["BBP_Lower"]),
            axis=1
        )

        # 6. Moving Average
        ma_length = 20
        ma_type = "SMA"
        norm_smooth = 3
        temp_data["MA"] = moving_average_value(temp_data["Close"], ma_length, ma_type)
        buy_signal = temp_data["Close"] > temp_data["MA"]
        sell_signal = temp_data["Close"] < temp_data["MA"]
        temp_data["MA_Score"] = normalize(buy_signal, sell_signal, norm_smooth, temp_data["Close"])

        # 7. VWAP
        stdev_multiplier = 1.5
        temp_data["Typical Price"] = (temp_data["High"] + temp_data["Low"] + temp_data["Close"]) / 3
        temp_data["Cumulative TPxVolume"] = (temp_data["Typical Price"] * temp_data["Volume"]).cumsum()
        temp_data["Cumulative Volume"] = temp_data["Volume"].cumsum()
        temp_data["VWAP"] = temp_data["Cumulative TPxVolume"] / temp_data["Cumulative Volume"]
        rolling_std = temp_data["Typical Price"].rolling(window=len(temp_data)).std()
        temp_data["VWAP Upper"] = temp_data["VWAP"] + (rolling_std * stdev_multiplier)
        temp_data["VWAP Lower"] = temp_data["VWAP"] - (rolling_std * stdev_multiplier)
        buy_signals = temp_data["Close"] > temp_data["VWAP Upper"]
        sell_signals = temp_data["Close"] < temp_data["VWAP Lower"]
        temp_data["VWAP_Score"] = normalize(buy_signals, sell_signals, norm_smooth, temp_data["Close"].tolist())

        # 8. Bollinger Bands
        bb_length = 20
        bb_mult = 2.0
        temp_data["BB_Middle"], temp_data["BB_Upper"], temp_data["BB_Lower"] = talib.BBANDS(
            temp_data["Close"], timeperiod=bb_length, nbdevup=bb_mult, nbdevdn=bb_mult, matype=0
        )
        buy_signals = temp_data["Close"] > temp_data["BB_Upper"]
        sell_signals = temp_data["Close"] < temp_data["BB_Lower"]
        temp_data["BB_Score"] = normalize(buy_signals, sell_signals, norm_smooth, temp_data["Close"].tolist())

        # 9. Supertrend
        atr_length = 10
        st_factor = 3
        atr = talib.ATR(temp_data["High"], temp_data["Low"], temp_data["Close"], timeperiod=atr_length)
        hl2 = (temp_data["High"] + temp_data["Low"]) / 2
        temp_data["Supertrend_Upper"] = hl2 + (st_factor * atr)
        temp_data["Supertrend_Lower"] = hl2 - (st_factor * atr)
        supertrend = [0] * len(temp_data)
        direction = [0] * len(temp_data)
        for i in range(len(temp_data)):
            if i == 0:
                supertrend[i] = temp_data["Supertrend_Upper"][i]
                direction[i] = 1
            else:
                if direction[i - 1] == 1:
                    if temp_data["Close"][i] < temp_data["Supertrend_Lower"][i]:
                        direction[i] = -1
                        supertrend[i] = temp_data["Supertrend_Lower"][i]
                    else:
                        direction[i] = 1
                        supertrend[i] = min(temp_data["Supertrend_Upper"][i], supertrend[i - 1])
                elif direction[i - 1] == -1:
                    if temp_data["Close"][i] > temp_data["Supertrend_Upper"][i]:
                        direction[i] = 1
                        supertrend[i] = temp_data["Supertrend_Upper"][i]
                    else:
                        direction[i] = -1
                        supertrend[i] = max(temp_data["Supertrend_Lower"][i], supertrend[i - 1])
        temp_data["Supertrend"] = supertrend
        temp_data["Direction"] = direction
        temp_data["Buy_Signal"] = temp_data["Close"] > temp_data["Supertrend"]
        temp_data["Sell_Signal"] = temp_data["Close"] < temp_data["Supertrend"]
        temp_data["Supertrend_Score"] = normalize(
            buy=temp_data["Buy_Signal"],
            sell=temp_data["Sell_Signal"],
            smooth=norm_smooth,
            close_values=temp_data["Close"].tolist(),
        )

        # 10. Linear Regression
        lr_length = 25
        temp_data["Linear_Regression_Score"] = linear_regression_score(temp_data["Close"].values, lr_length)

        # --- Overall Sentiment ---
        pivot_length = 5
        norm_smooth = 3

        def pivot_high(prices, length):
            return prices.rolling(window=length, center=True).apply(lambda x: x.argmax() == length // 2, raw=False)

        def pivot_low(prices, length):
            return prices.rolling(window=length, center=True).apply(lambda x: x.argmin() == length // 2, raw=False)

        temp_data["Pivot_High"] = pivot_high(temp_data["Close"], pivot_length)
        temp_data["Pivot_Low"] = pivot_low(temp_data["Close"], pivot_length)

        ph_y = np.nan
        pl_y = np.nan
        ph_cross = False
        pl_cross = False
        bullish = []
        bearish = []

        for i in range(len(temp_data)):
            bull = False
            bear = False

            if temp_data["Pivot_High"][i]:
                ph_y = temp_data["Close"][i]
                ph_cross = False

            if temp_data["Pivot_Low"][i]:
                pl_y = temp_data["Close"][i]
                pl_cross = False

            if temp_data["Close"][i] > ph_y and not ph_cross:
                ph_cross = True
                bull = True

            if temp_data["Close"][i] < pl_y and not pl_cross:
                pl_cross = True
                bear = True

            bullish.append(bull)
            bearish.append(bear)

        temp_data["Bullish"] = bullish
        temp_data["Bearish"] = bearish

        temp_data["Sentiment"] = normalize(temp_data["Bullish"], temp_data["Bearish"], norm_smooth, temp_data["Close"])

        return temp_data
    
    temp_data = calculate_indicators_and_scores(temp_data)

    dji_data = calculate_indicators_and_scores(dji_data)
    nasdaq_data = calculate_indicators_and_scores(nasdaq_data)
    s_and_p_500_data = calculate_indicators_and_scores(s_and_p_500_data)

    # # Display final sentiment scores
    # st.write(temp_data[["Datetime", "RSI_Score", "%K_Score", "Stoch_RSI_Score", "CCI_Score", "BBP_Score",
    #                 "MA_Score", "VWAP_Score", "BB_Score", "Supertrend_Score", "Linear_Regression_Score", "Sentiment"]].tail(21))

    st.write("Below is the Market sentiment with the latest time on 30 mins")
    
    temp_data_summary = {
        "RSI_Score": temp_data.iloc[-1]['RSI_Score'],
        "%K_Score": temp_data.iloc[-1]['%K_Score'],
        "Stoch_RSI_Score": temp_data.iloc[-1]['Stoch_RSI_Score'],
        "CCI_Score": temp_data.iloc[-1]['CCI_Score'],
        "BBP_Score": temp_data.iloc[-1]['BBP_Score'],
        "MA_Score": temp_data.iloc[-1]['MA_Score'],
        "VWAP_Score": temp_data.iloc[-1]['VWAP_Score'],
        "BB_Score": temp_data.iloc[-1]['BB_Score'],
        "Supertrend_Score": temp_data.iloc[-1]['Supertrend_Score'],
        "Linear_Regression_Score": temp_data.iloc[-1]['Linear_Regression_Score'],
        "Sentiment": temp_data.iloc[-1]['Sentiment'],
    }

    # Extract scalar values directly
    bar_labels = [
        'RSI', '%K', 'Stoch RSI', 'CCI', 'BBP', 'MA', 'VWAP', 'BB', 'ST', 'REG'
    ]
    bar_values = [
        temp_data_summary["RSI_Score"], temp_data_summary["%K_Score"], temp_data_summary["Stoch_RSI_Score"],
        temp_data_summary["CCI_Score"], temp_data_summary["BBP_Score"], temp_data_summary["MA_Score"],
        temp_data_summary["VWAP_Score"], temp_data_summary["BB_Score"], temp_data_summary["Supertrend_Score"],
        temp_data_summary["Linear_Regression_Score"]
    ]
    sentiment_value = temp_data_summary["Sentiment"]

    # Create the bar chart figure
    fig = go.Figure()

    # Bar chart for indicators
    fig.add_trace(go.Bar(
        x=bar_labels,
        y=bar_values,
        marker_color=[
            'red' if v < 50 else 'blue' if v > 50 else 'gray' for v in bar_values
        ],
        name='Indicators'
    ))

    # Adding threshold lines
    fig.add_hline(y=75, line_dash='dash', line_color='blue', annotation_text="Overbought (75)", annotation_position="top left")
    fig.add_hline(y=50, line_dash='dash', line_color='gray', annotation_text="Neutral (50)", annotation_position="bottom right")
    fig.add_hline(y=25, line_dash='dash', line_color='red', annotation_text="Oversold (25)", annotation_position="bottom left")

    # Adding text labels for each bar
    for i, v in enumerate(bar_values):
        fig.add_trace(go.Scatter(
            x=[bar_labels[i]],
            y=[v + 2],  # Position slightly above the bar
            text=[f"{v:.1f}"],
            mode='text',
            showlegend=False
        ))

    # Update layout for the bar chart
    fig.update_layout(
        title_text='Market Sentiment Indicators',
        yaxis_title='Scores',
        height=400,
        width=1200,
        bargap=0.2,
        template="plotly_dark"
    )

    # Display the bar chart in Streamlit
    st.plotly_chart(fig, use_container_width=True)

    # Create the sentiment gauge figure
    gauge_fig = go.Figure()

    # Sentiment gauge
    gauge_fig.add_trace(go.Indicator(
        mode="gauge+number",
        value=sentiment_value,
        title={'text': "Market Sentiment"},
        gauge={
            'axis': {'range': [-90, 90]},
            'bar': {'color': "blue" if sentiment_value > 50 else "red"},
            'steps': [
                {'range': [-90, -30], 'color': "red"},
                {'range': [-30, 30], 'color': "gray"},
                {'range': [30, 90], 'color': "blue"}
            ],
            'threshold': {
                'line': {'color': "black", 'width': 4},
                'thickness': 0.75,
                'value': sentiment_value
            }
        }
    ))

    # Update layout for the sentiment gauge
    gauge_fig.update_layout(
        height=400,
        width=600,
        template="plotly_dark",
        title_text="Market Sentiment Gauge"
    )

    # Display the sentiment gauge in Streamlit
    st.plotly_chart(gauge_fig, use_container_width=True)

    # st.write(data)

    # # Section 2: True Range and ATR Calculations
    # def calculate_atr(data, atr_period=14):
    #     data['H-L'] = data['High'] - data['Low']
    #     data['H-PC'] = np.abs(data['High'] - data['Adj Close'].shift(1))
    #     data['L-PC'] = np.abs(data['Low'] - data['Adj Close'].shift(1))
    #     data['TR'] = data[['H-L', 'H-PC', 'L-PC']].max(axis=1)
    #     data['ATR_14'] = data['TR'].ewm(alpha=1/atr_period, adjust=False).mean()
    #     return data

    # data = calculate_atr(data)

    # The above code snippet is written in Python and it appears to be fetching daily data using the
    # `yf.download` function to calculate the Average True Range (ATR) for daily intervals. The
    # `ticker` variable is likely representing the stock symbol or financial instrument for which the
    # data is being fetched. The data is being fetched for a period of 60 days with a daily interval
    # of 1 day. The fetched data is then reset to have a clean index.
    # Fetch daily data to calculate ATR for daily intervals
    daily_data = yf.download(ticker, period="60d", interval="1d").reset_index()
    # daily_data['H-L'] = daily_data['High'] - daily_data['Low']
    # daily_data['H-PC'] = np.abs(daily_data['High'] - daily_data['Adj Close'].shift(1))
    # daily_data['L-PC'] = np.abs(daily_data['Low'] - daily_data['Adj Close'].shift(1))
    # daily_data['TR'] = daily_data[['H-L', 'H-PC', 'L-PC']].max(axis=1)
    # daily_data['ATR_14_1_day'] = daily_data['TR'].ewm(alpha=1/14, adjust=False).mean()
    # daily_data['Prev_Day_ATR_14_1_Day'] = daily_data['ATR_14_1_day'].shift(1)
    # daily_data['Date'] = pd.to_datetime(daily_data['Date']).dt.date
    # st.write(data)
    data['Date'] = pd.to_datetime(data['Datetime']).dt.date
    # # Convert 'Date' column in both dataframes to datetime format
    data['Date'] = pd.to_datetime(data['Date'])
    daily_data['Date'] = pd.to_datetime(daily_data['Date'])

    # Calculate ATR for 30-minute and daily intervals using TA-Lib
    atr_period = 14

    # ATR for daily data using TA-Lib
    daily_data['ATR_14_1_day'] = talib.ATR(daily_data['High'], daily_data['Low'], daily_data['Close'], timeperiod=atr_period)
    daily_data['Prev_Day_ATR_14_1_Day'] = daily_data['ATR_14_1_day'].shift(1)

    # Merge ATR from daily data into 30-minute data
    final_data = pd.merge(data, daily_data[['Date', 'ATR_14_1_day', 'Prev_Day_ATR_14_1_Day']], on='Date', how='left')

    # Calculate ATR for 30-minute data using TA-Lib
    final_data['ATR_14_30_mins'] = talib.ATR(final_data['High'], final_data['Low'], final_data['Close'], timeperiod=atr_period)

    final_data['ATR'] = final_data['ATR_14_30_mins']

    # Drop unnecessary columns from earlier calculations
    final_data = final_data.drop(['H-L', 'H-PC', 'L-PC', 'TR'], axis=1, errors='ignore')

    # Merge ATR into 30-minute data
    
    final_data = pd.merge(data, daily_data[['Date', 'ATR_14_1_day', 'Prev_Day_ATR_14_1_Day']], on='Date', how='left')

    # st.write(final_data)

    # Ensure 'Datetime' is in datetime format
    final_data['Datetime'] = pd.to_datetime(final_data['Datetime'])
    final_data = final_data.sort_values('Datetime',ascending=True)

    final_data['Close'] = pd.to_numeric(final_data['Close'], errors='coerce')

    # # Convert 'Close' to NumPy array of floats
    # close_prices = final_data['Close'].astype(float).values

    # # Calculate SMA using TA-Lib
    # final_data['MA_20'] = talib.SMA(close_prices, timeperiod=20)

    # Calculate Moving Average (MA)
    final_data['MA_20'] = final_data['Close'].rolling(window=20).mean()
    # # Calculate Moving Average (MA) using TA-Lib
    # final_data['MA_20'] = talib.SMA(final_data['Close'], timeperiod=20)

    # # Calculate Relative Strength Index (RSI)
    # delta = final_data['Close'].diff()
    # gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    # loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    # final_data['RSI'] = 100 - (100 / (1 + gain / loss))

    # Calculate RSI using TA-Lib with a 14-day period
    final_data['RSI'] = talib.RSI(final_data['Close'], timeperiod=14)


    # # Calculate Moving Average Convergence Divergence (MACD)
    # short_ema = final_data['Close'].ewm(span=12, adjust=False).mean()
    # long_ema = final_data['Close'].ewm(span=26, adjust=False).mean()
    # final_data['MACD'] = short_ema - long_ema
    # final_data['Signal Line'] = final_data['MACD'].ewm(span=9, adjust=False).mean()

    # # Calculate Bollinger Bands
    # final_data['MA_20'] = final_data['Close'].rolling(window=20).mean()
    # final_data['Bollinger_Upper'] = final_data['MA_20'] + (final_data['Close'].rolling(window=20).std() * 2)
    # final_data['Bollinger_Lower'] = final_data['MA_20'] - (final_data['Close'].rolling(window=20).std() * 2)

    # Calculate Moving Average Convergence Divergence (MACD) using TA-Lib
    final_data['MACD'], final_data['Signal Line'], _ = talib.MACD(final_data['Close'], fastperiod=12, slowperiod=26, signalperiod=9)

    # Calculate Bollinger Bands using TA-Lib
    final_data['MA_20_BB'], final_data['Bollinger_Upper'], final_data['Bollinger_Lower'] = talib.BBANDS(
        final_data['Close'], timeperiod=20, nbdevup=2, nbdevdn=2, matype=0
    )


    final_data['Date'] = pd.to_datetime(final_data['Date'])

    # # Calculate Volume Weighted Average Price (VWAP)
    # final_data['VWAP'] = (final_data['Volume'] * (final_data['High'] + final_data['Low'] + final_data['Close']) / 3).cumsum() / final_data['Volume'].cumsum()

    # # Calculate Fibonacci Retracement Levels (use high and low from a specific range if applicable)
    # highest = final_data['High'].max()
    # lowest = final_data['Low'].min()
    # final_data['Fib_38.2'] = highest - (highest - lowest) * 0.382
    # final_data['Fib_50'] = (highest + lowest) / 2
    # final_data['Fib_61.8'] = highest - (highest - lowest) * 0.618

    # Define a function to calculate VWAP for each group
    def calculate_vwap(df):
        typical_price = (df['High'] + df['Low'] + df['Close']) / 3
        vwap = (typical_price * df['Volume']).cumsum() / df['Volume'].cumsum()
        return vwap

    grouped = final_data.groupby(final_data['Date'].dt.date)

    # Apply the VWAP calculation for each group
    final_data['VWAP'] = grouped.apply(lambda x: calculate_vwap(x)).reset_index(level=0, drop=True)

    # Extract the date part to group the data by day
    final_data['Only_Date'] = final_data['Date'].dt.date

    # Calculate daily high and low values from the 30-minute data
    daily_high_low = final_data.groupby('Only_Date').agg({'High': 'max', 'Low': 'min'}).reset_index()

    # Add the previous day's high and low to the 30-minute data
    daily_high_low['Previous_High'] = daily_high_low['High'].shift(1)
    daily_high_low['Previous_Low'] = daily_high_low['Low'].shift(1)

    # Merge the previous high and low values into the original 30-minute data
    final_data = final_data.merge(daily_high_low[['Only_Date', 'Previous_High', 'Previous_Low']], 
                                        left_on='Only_Date', right_on='Only_Date', how='left')

    # Calculate Fibonacci retracement levels for each 30-minute interval based on the previous day's high and low
    final_data['Fib_38.2'] = final_data['Previous_High'] - (final_data['Previous_High'] - final_data['Previous_Low']) * 0.382
    final_data['Fib_50'] = (final_data['Previous_High'] + final_data['Previous_Low']) / 2
    final_data['Fib_61.8'] = final_data['Previous_High'] - (final_data['Previous_High'] - final_data['Previous_Low']) * 0.618

    # Drop unnecessary columns
    final_data.drop(['Previous_High', 'Previous_Low'], axis=1, inplace=True)

    # Calculate Average True Range (ATR) using TA-Lib (already done above)
    # final_poc_data['ATR'] is equivalent to 'ATR_14_30_mins'

    

    # Calculate Average True Range (ATR)
    high_low = final_data['High'] - final_data['Low']
    high_close = np.abs(final_data['High'] - final_data['Close'].shift())
    low_close = np.abs(final_data['Low'] - final_data['Close'].shift())
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    final_data['ATR'] = true_range.rolling(window=14).mean()

    # Calculate Stochastic Oscillator
    final_data['14-high'] = final_data['High'].rolling(window=14).max()
    final_data['14-low'] = final_data['Low'].rolling(window=14).min()


    # final_data['%K'] = (final_data['Close'] - final_data['14-low']) * 100 / (final_data['14-high'] - final_data['14-low'])
    # final_data['%D'] = final_data['%K'].rolling(window=3).mean()

    # # Calculate Parabolic SAR (for simplicity, this example uses a fixed acceleration factor)
    # final_data['PSAR'] = final_data['Close'].shift() + (0.02 * (final_data['High'] - final_data['Low']))

    # Calculate Stochastic Oscillator using TA-Lib
    final_data['%K'], final_data['%D'] = talib.STOCH(
        final_data['High'], final_data['Low'], final_data['Close'],
        fastk_period=14, slowk_period=3, slowk_matype=0, slowd_period=3, slowd_matype=0
    )

    # Calculate Parabolic SAR using TA-Lib
    final_data['PSAR'] = talib.SAR(final_data['High'], final_data['Low'], acceleration=0.02, maximum=0.2)

    # st.write(final_data)

    # Section 3: TPO Profile Calculation
    def calculate_tpo(data, tick_size=0.01, value_area_percent=70):
        price_levels = np.arange(data['Low'].min(), data['High'].max(), tick_size)
        tpo_counts = defaultdict(list)
        letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
        letter_idx = 0

        for _, row in data.iterrows():
            current_letter = letters[letter_idx % len(letters)]
            for price in price_levels:
                if row['Low'] <= price <= row['High']:
                    tpo_counts[price].append(current_letter)
            letter_idx += 1

        total_tpo = sum(len(counts) for counts in tpo_counts.values())
        value_area_target = total_tpo * value_area_percent / 100

        sorted_tpo = sorted(tpo_counts.items(), key=lambda x: len(x[1]), reverse=True)
        value_area_tpo = 0
        vah = 0
        val = float('inf')

        for price, counts in sorted_tpo:
            if value_area_tpo + len(counts) <= value_area_target:
                value_area_tpo += len(counts)
                vah = max(vah, price)
                val = min(val, price)
            else:
                break

        poc = sorted_tpo[0][0]  # Price with highest TPO count
        return tpo_counts, poc, vah, val

    # Section 4: IB Range, Market Classification, and Signals
    def calculate_market_profile(data):
        daily_tpo_profiles = []
        value_area_percent = 70
        tick_size = 0.01

        for date, group in data.groupby('Date'):
            tpo_counts, poc, vah, val = calculate_tpo(group, tick_size, value_area_percent)

            initial_balance_high = group['High'].iloc[:2].max()
            initial_balance_low = group['Low'].iloc[:2].min()
            initial_balance_range = initial_balance_high - initial_balance_low

            day_range = group['High'].max() - group['Low'].min()
            range_extension = day_range - initial_balance_range

            last_row = group.iloc[-1]
            last_row_close = last_row['Close']
            last_row_open = last_row['Open']

    #         st.write(last_row)

            if day_range <= initial_balance_range * 1.15:
                day_type = 'Normal Day'
            elif initial_balance_range < day_range <= initial_balance_range * 2:
                if last_row_open >= last_row_close:
                    day_type = 'Negative Normal Variation Day'
                elif last_row_open <= last_row_close:
                    day_type = 'Positive Normal Variation Day'
                else:
                    day_type = 'Normal Variation Day'
            elif day_range > initial_balance_range * 2:
                day_type = 'Trend Day'
            else:
                day_type = 'Neutral Day'

            if last_row['Close'] >= initial_balance_high:
                close_type = 'Closed above Initial High'
            elif last_row['Close'] <= initial_balance_low:
                close_type = 'Closed below Initial Low'
            else:
                close_type = 'Closed between Initial High and Low'

    #         if last_row['Close'] >= vah:
    #             close_type_va = 'Closed above VAH'
    #         elif last_row['Close'] <= initial_balance_low:
    #             close_type_va = 'Closed below VAL'
    #         else:
    #             close_type_va = 'Closed between VAH and VAL'

            tpo_profile = {
                'Date': date,
                'POC': round(poc, 2),
                'VAH': round(vah, 2),
                'VAL': round(val, 2),
                'Initial Balance High': round(initial_balance_high, 2),
                'Initial Balance Low': round(initial_balance_low, 2),
                'Initial Balance Range': round(initial_balance_range, 2),
                'Day Range': round(day_range, 2),
                'Range Extension': round(range_extension, 2),
                'Day Type': day_type,
                'Close Type' : close_type
    #             ,
    #             'Close Type VA':close_type_va
            }
            daily_tpo_profiles.append(tpo_profile)

        return pd.DataFrame(daily_tpo_profiles)

    market_profile_df = calculate_market_profile(final_data)

    # Merge TPO profile data into final_data based on the 'Date'
    final_data = pd.merge(final_data, market_profile_df, on='Date', how='left')

    # st.write(market_profile_df)

    # Section 5: Generate Signals based on Market Profile
    def generate_signals(market_profile_df):
        trends = []
        for i in range(1, len(market_profile_df)):
            prev_day = market_profile_df.iloc[i - 1]
            curr_day = market_profile_df.iloc[i]

            if curr_day['Initial Balance High'] > prev_day['VAH']:
                trend = 'Bullish'
            elif curr_day['Initial Balance Low'] < prev_day['VAL']:
                trend = 'Bearish'
            else:
                trend = 'Neutral'

            trends.append({
                'Date': curr_day['Date'],
                'Trend': trend,
                'Previous Day VAH': prev_day['VAH'],
                'Previous Day VAL': prev_day['VAL'],
                'Previous Day POC': prev_day['POC'],
            })

        return pd.DataFrame(trends)

    signals_df = generate_signals(market_profile_df)

    # Merge trend data into final_data
    final_data = pd.merge(final_data, signals_df, on='Date', how='left')

    # st.write(final_data)

    # Define the conditions for Initial Balance Range classification
    conditions = [
        final_data['Initial Balance Range'] < final_data['Prev_Day_ATR_14_1_Day'] / 3,
        (final_data['Initial Balance Range'] >= final_data['Prev_Day_ATR_14_1_Day'] / 3) & 
        (final_data['Initial Balance Range'] <= final_data['Prev_Day_ATR_14_1_Day']),
        final_data['Initial Balance Range'] > final_data['Prev_Day_ATR_14_1_Day']
    ]

    # Define the corresponding values for each condition
    choices = ['Small', 'Medium', 'Large']

    # Create the IB Range column using np.select()
    final_data['IB Range'] = np.select(conditions, choices, default='Unknown')

    # Round all values in final_data to 2 decimals
    final_data = final_data.round(2)

    # Display the final merged DataFrame
    # st.write(final_data)

    # Get the unique dates and sort them
    sorted_dates = sorted(set(final_data['Date']))
    final_data['2 Day VAH and VAL'] = ''
    # final_data['Previous Initial Market Profile Shape'] = ''
    # final_data['Previous Refined Market Profile Shape'] = ''

    # Use a for loop with range() to iterate over the sorted dates by index
    for i in range(2, len(sorted_dates)):
        date = sorted_dates[i]
        previous_date = sorted_dates[i - 1]

        # print(f"Current Date: {date}")
        # print(f"Previous Date: {previous_date}")

        # Extract data for the previous date
        previous_data = final_data[final_data['Date'] == previous_date]

        day_high = previous_data['High'].max()

        day_low = previous_data['Low'].max()

        # Initialize an empty list for actions
        actions = []
    #     actions.append(date)

        # Ensure previous_data has rows before accessing
        if not previous_data.empty:
            # Get the last row of the previous day's data
            last_row = previous_data.iloc[-1]

            # Compare 'Close' with 'VAH' and 'VAL'
            if last_row['Close'] >= last_row['VAH']:
                actions.append('Previous Day Close Above VAH')
                actions.append('Previous Day Close Bullish')
            elif last_row['Close'] <= last_row['VAL']:
                actions.append('Previous Day Close Below VAL')
                actions.append('Previous Day Close Bearish')
            else:
                actions.append('Previous Day Close Neutral')

            if last_row['Previous Day VAH'] >= last_row['VAH'] and last_row['Previous Day VAL'] <= last_row['VAL']:
                actions.append('Insider Neutral')
            elif last_row['Previous Day VAH'] <= last_row['VAH'] and last_row['Previous Day VAL'] >= last_row['VAL']:
                actions.append('Outsider Neutral')

            if last_row['IB Range'] == 'Large' and last_row['Close'] <= last_row['Initial Balance High']:
                final_day_type = 'Large Range Normal Day'
            elif last_row['IB Range'] == 'Medium' and day_high >= last_row['Initial Balance High'] and day_low <= last_row['Initial Balance Low']:
                final_day_type = 'Medium Range Neutral Day'
            elif last_row['IB Range'] == 'Medium' and last_row['Close'] >= last_row['Initial Balance High']:
                final_day_type = 'Medium Range +ve Normal Variation Day'
            elif last_row['IB Range'] == 'Medium' and last_row['Close'] <= last_row['Initial Balance Low']:
                final_day_type = 'Medium Range -ve Normal Variation Day'
            elif last_row['IB Range'] == 'Small' and last_row['Close'] >= last_row['Initial Balance High']:
                final_day_type = 'Small Range +ve Trend Variation Day'
            elif last_row['IB Range'] == 'Small' and last_row['Close'] <= last_row['Initial Balance Low']:
                final_day_type = 'Small Range -ve Trend Variation Day'
            elif last_row['IB Range'] == 'Small' and last_row['Close'] <= last_row['Initial Balance High'] and last_row['Close'] >= last_row['Initial Balance Low']:
                final_day_type = 'Small Range Non Trend Variation Day'
            else:
                final_day_type = ''
            
            # if last_row['POC'] > last_row['VAH']:
            #     previous_day_shape = "P-shape (Bullish Accumulation)"
            # elif last_row['POC'] < last_row['VAL']:
            #     previous_day_shape = "b-shape (Bearish Accumulation)"
            # elif last_row['VAH'] > last_row['POC'] > last_row['VAL']:
            #     previous_day_shape = "D-shape (Balanced Market)"
            # else:
            #     previous_day_shape = "B-shape (Double Distribution)"



        # # Print or perform further operations with actions
        # print(actions)

        final_data.loc[final_data['Date'] == previous_date, '2 Day VAH and VAL'] = str(actions)
        final_data.loc[final_data['Date'] == previous_date, 'Adjusted Day Type'] = str(final_day_type)
        # final_data.loc[final_data['Date'] == previous_date, 'Previous Initial Market Profile Shape'] = str(previous_day_shape)

    st.write("final_data")

    # st.write(dji_data.columns)

    dji_data = dji_data[['Datetime','Open','High','Low','Close','Volume','RSI_Score','%K_Score','Stoch_RSI_Score','CCI_Score','BBP_Score','MA_Score','VWAP_Score','BB_Score','Supertrend_Score','Linear_Regression_Score','Sentiment']]
    nasdaq_data = nasdaq_data[['Datetime','Open','High','Low','Close','Volume','RSI_Score','%K_Score','Stoch_RSI_Score','CCI_Score','BBP_Score','MA_Score','VWAP_Score','BB_Score','Supertrend_Score','Linear_Regression_Score','Sentiment']]
    s_and_p_500_data = s_and_p_500_data[['Datetime','Open','High','Low','Close','Volume','RSI_Score','%K_Score','Stoch_RSI_Score','CCI_Score','BBP_Score','MA_Score','VWAP_Score','BB_Score','Supertrend_Score','Linear_Regression_Score','Sentiment']]

    # Rename columns in dji_data to add '_dji' suffix except for the 'Datetime' column
    dji_data = dji_data.rename(columns={col: f"{col}_dji" for col in dji_data.columns if col != 'Datetime'})
    nasdaq_data = nasdaq_data.rename(columns={col: f"{col}_nasdaq" for col in nasdaq_data.columns if col != 'Datetime'})
    s_and_p_500_data = s_and_p_500_data.rename(columns={col: f"{col}_s_and_p_500" for col in s_and_p_500_data.columns if col != 'Datetime'})
    
    # Merge both dataframes on the "Datetime" column
    merged_data = pd.merge(final_data, dji_data, on='Datetime')
    merged_data = pd.merge(merged_data, nasdaq_data, on='Datetime')
    merged_data = pd.merge(merged_data, s_and_p_500_data, on='Datetime')

    final_data = merged_data.copy()
    
    st.write(final_data)

    # # Initial quick identification of market profile shape based on POC, VAH, and VAL
    # def quick_identify_profile_shape(vah, val, poc):
    #     if poc > vah:
    #         return "P-shape (Bullish Accumulation)"
    #     elif poc < val:
    #         return "b-shape (Bearish Accumulation)"
    #     elif vah > poc > val:
    #         return "D-shape (Balanced Market)"
    #     else:
    #         return "B-shape (Double Distribution)"

    # # Refine the initial guess with skewness and kurtosis
    # def refine_with_skew_kurtosis(volume_profile, shape_guess):
    #     volumes = volume_profile['Total Volume'].values
    #     skewness = skew(volumes)
    #     kurt = kurtosis(volumes)

    #     if shape_guess == "P-shape" and skewness < 0:
    #         return "b-shape (Bearish Accumulation)"
    #     if shape_guess == "b-shape" and skewness > 0:
    #         return "P-shape (Bullish Accumulation)"

    #     if shape_guess == "D-shape" and abs(skewness) > 0.5 and kurt > 0:
    #         return "B-shape (Double Distribution)"

    #     return shape_guess

    # # Calculate the volume profile
    # volume_profile = calculate_volume_profile(data, row_layout=24)
    # vah, val, poc = calculate_vah_val_poc(volume_profile)

    # # Initial shape identification
    # initial_shape = quick_identify_profile_shape(vah, val, poc)

    # # Refined shape identification
    # refined_shape = refine_with_skew_kurtosis(volume_profile, initial_shape)


    # Create a 'casted_date' column to only capture the date part of the Datetime
    final_data['casted_date'] = final_data['Date']

    # Sort by casted_date to ensure correct order
    final_data = final_data.sort_values(by='Datetime')

    # Create a 'casted_date' column to only capture the date part of the Datetime
    final_data['casted_date'] = final_data['Date']

    final_data['casted_date'] = final_data['casted_date'].dt.date

    # Get a sorted list of unique dates
    sorted_dates = sorted(final_data['casted_date'].unique())

    # Find the index of the selected date in the sorted list
    current_date_index = sorted_dates.index(input_date) if input_date in sorted_dates else None

    # Determine the previous date if it exists
    previous_date = sorted_dates[current_date_index - 1] if current_date_index and current_date_index > 0 else None

    # st.write(input_date)
    
    # Filter based on the input date (input_date) from the sidebar
    filtered_data = final_data[final_data['casted_date'] == input_date]

    # st.write(final_data.tail())
    # st.write(len(filtered_data))

    # Filter based on the input date (input_date) from the sidebar
    previous_filtered_data = final_data[final_data['casted_date'] == previous_date]
    # st.write(previous_date)
    # st.write(len(previous_filtered_data))
    # st.write(filtered_data.columns)

    # Section 7: Display the Data for Selected Date
    if not filtered_data.empty:
        st.title(f"Market Profile for {input_date}")
        st.write(f"Previous Day Type: {previous_filtered_data['Day Type'].values[0]}")
        st.write(f"Previous Adjusted Day Type: {previous_filtered_data['Adjusted Day Type'].values[0]}")
        st.write(f"Previous Close Type: {previous_filtered_data['Close Type'].values[0]}")
    #     st.write(f"Close Type VA: {filtered_data['Close Type VA'].values[0]}")
        st.write(f"Previous 2 Day VAH and VAL:{previous_filtered_data['2 Day VAH and VAL'].values[0]}")
        st.write(f"IB Range: {filtered_data['Initial Balance Range'].values[0]}")
        st.write(f"2 Day VAH and VAL: VAH - {filtered_data['VAH'].values[0]}, VAL - {signals_df['Previous Day VAL'].values[-1]}")

        st.write(filtered_data)
    #     st.write(filtered_data.columns)

        # Calculate the opening price and difference from the previous close
        opening_price = filtered_data.iloc[0]['Open']
        previous_close = previous_filtered_data.iloc[-1]['Close']
        open_point_diff = round(opening_price - previous_close, 2) if previous_close else None
        open_percent_diff = round((open_point_diff / previous_close) * 100, 2) if previous_close else None
        open_above_below = "above" if open_point_diff > 0 else "below" if open_point_diff < 0 else "no change"

        current_row = filtered_data.iloc[0]
        last_row = previous_filtered_data.iloc[-1]


        input_text = (
            f"Today’s profile on {input_date} for {ticker} with IB Range Type {current_row['IB Range']} Range. The market opened at {opening_price}, "
            f"Opening_Gap_Percentage is {open_percent_diff}% ( Opening_Gap_Points {abs(open_point_diff)} points) {open_above_below} the previous day's close. "
            f"The Initial Balance High is {current_row['Initial Balance High']} and Initial Balance Low is {current_row['Initial Balance Low']}, "
            f"giving an Initial Balance Range of {current_row['Initial Balance Range']}. "
            f"Yesterday's VAH was {last_row['VAH']} and Yesterday's VAL was {last_row['VAL']}. "
            f"Day before yesterday's VAH was {last_row['Previous Day VAH']} and Day before yesterday's VAL was {last_row['Previous Day VAL']}. "
            f"Previous day Type: {last_row['Day Type']}.\n"
            f"Previous Adjusted Day Type: {final_day_type}.\n"
            f"Previous Close Type: {last_row['Close Type']}.\n"
            f"Previous 2 Day VAH and VAL: {actions}.\n"

            # Adding indicators
            f"MA_20_Day is {last_row['MA_20']}. "
            f"RSI is {last_row['RSI']}. "
            f"MACD is {last_row['MACD']} with Signal Line at {last_row['Signal Line']}. "
            f"Bollinger Bands Upper at {last_row['Bollinger_Upper']} and Bollinger Bands Lower at {last_row['Bollinger_Lower']}. "
            f"VWAP is {last_row['VWAP']}. "
            f"Fibonacci Levels: 38.2% at {last_row['Fib_38.2']}, 50% at {last_row['Fib_50']}, 61.8% at {last_row['Fib_61.8']}. "
            f"ATR is {last_row['ATR']}. "
            f"Stochastic Oscillator %K is {last_row['%K']} and %D is {last_row['%D']}. "
            f"Parabolic SAR is at {last_row['PSAR']}. "

            f"Given these indicators, what is the expected market direction for today?"
        )

        st.write(input_text)

        # Segments dictionary
        segments = {
            "IB_High": {"numeric_value": "Initial Balance High"},
            "IB_Low": {"numeric_value": "Initial Balance Low"},
            "Range_Type": {"text": "IB Range Type"},
            "Previous_Day_Type": {"text": "Previous day Type:"},
            "Previous_Adjusted_Day_Type": {"text": "Previous Adjusted Day Type:"},
            "Previous_Close_Type": {"text": "Previous Close Type:"},
            "Previous_2_D_VAH_VAL": {"text": "Previous 2 Day VAH and VAL:"},
            "IB_Range": {"numeric_value": "Initial Balance Range"},
            "VAH_Yesterday": {"numeric_value": "Yesterday's VAH"},
            "VAL_Yesterday": {"numeric_value": "Yesterday's VAL"},
            "VAH_DayBefore": {"numeric_value": "Day before yesterday's VAH"},
            "VAL_DayBefore": {"numeric_value": "Day before yesterday's VAL"},
            "Moving_Avg_20": {"numeric_value": "MA_20_Day"},
            "RSI": {"numeric_value": "RSI"},
            "MACD": {"numeric_value": "MACD"},
            "Signal_Line": {"numeric_value": "Signal Line"},
            "Bollinger_Upper": {"numeric_value": "Bollinger Bands Upper"},
            "Bollinger_Lower": {"numeric_value": "Bollinger Bands Lower"},
            "VWAP": {"numeric_value": "VWAP"},
            "ATR": {"numeric_value": "ATR"},
            "Market_Open": {"numeric_value": "market opened at"},
            "Opening_Gap_Percentage": {"numeric_value": "Opening_Gap_Percentage"},
            "Opening_Gap_Points": {"numeric_value": "Opening_Gap_Points"},
            "Fibonacci_38.2%": {"numeric_value": "38.2% at"},
            "Fibonacci_50%": {"numeric_value": "50% at"},
            "Fibonacci_61.8%": {"numeric_value": "61.8% at"},
            "Stochastic_Oscillator_%K": {"numeric_value": "Stochastic Oscillator %K"},
            "Stochastic_Oscillator_%D": {"numeric_value": "%D"},
            "Parabolic_SAR": {"numeric_value": "Parabolic SAR"},
        }

        # Function to extract values
        def extract_values(input_text, segments):
            extracted_data = {}
            for key, details in segments.items():
    #             text_marker = details["text"]
                text_marker = details.get("text", details.get("numeric_value"))

                # Extract numeric value following the text_marker
                match = re.search(rf"{text_marker}[^0-9\.\-]*([-+]?[0-9]*\.?[0-9]+)", input_text)
                if match:
                    extracted_data[key] = float(match.group(1))
                else:
                    # Extract categories or other types if present
                    match = re.search(rf"{text_marker}\s*([A-Za-z\s]+)", input_text)
                    if match:
                        extracted_data[key] = match.group(1).strip()
                    else:
                        extracted_data[key] = None  # Mark as None if not found

            return extracted_data

        # Extract values
        extracted_data = extract_values(input_text, segments)

        st.write(extracted_data)

        # Display the input text for reference
        st.write("Paste the input details below:")

#         # Input boxes for the headers and cookies
#         xsrf_token = st.text_input("Enter X-Xsrf-Token:", st.session_state.xsrf_token)
#         laravel_token = st.text_input("Enter laravel_token:", st.session_state.laravel_token)
#         xsrf_cookie = st.text_input("Enter XSRF-TOKEN:", st.session_state.xsrf_cookie)
#         laravel_session = st.text_input("Enter laravel_session:", st.session_state.laravel_session)
        st.write(st.session_state.xsrf_token)
        # Button to update the headers and cookies
        if st.session_state.xsrf_token != "":
            # Store the input values in session state
            xsrf_token = st.session_state.xsrf_token
            laravel_token = st.session_state.laravel_token
            xsrf_cookie = st.session_state.xsrf_cookie
            laravel_session = st.session_state.laravel_session
            
            headers = {
                "Accept": "*/*",
                "Accept-Encoding": "gzip, deflate, br, zstd",
                "Accept-Language": "en-GB,en-US;q=0.9,en;q=0.8",
                "Cache-Control": "no-cache",
                "Pragma": "no-cache",
                "Referer": f"https://www.barchart.com/stocks/quotes/{ticker}/interactive-chart/new",
                "Sec-Fetch-Dest": "empty",
                "Sec-Fetch-Mode": "cors",
                "Sec-Fetch-Site": "same-origin",
                "User-Agent": "Mozilla/5.0 (Linux; Android 6.0; Nexus 5 Build/MRA58N) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0.0.0 Mobile Safari/537.36",
                "X-Xsrf-Token": xsrf_token
            }

            cookies = {
                        "market": "eyJpdiI6IndEcWdvUFdXTTA3eFZ1YUZYQUVEWGc9PSIsInZhbHVlIjoiZjZxVmw3eXVOTFNNTloycHQyT0Qwdy81aHJpTE1EbjJudVpLNGJqR2hVblB6cmtQK0RkL2JCWUkyTnlLSXVBMSIsIm1hYyI6ImM1NzM1Nzc5YWRiNDE2MjJkMmNjMjc1NWM1OTI5YmI5MTI0OTFhNWQ3ZjU5N2VkZWU0MGM5YzRmYTgxNDEwMjAiLCJ0YWciOiIifQ==",
                "ic_tagmanager": "AY",
                "bcFreeUserPageView": "0",
                "OTGPPConsent": "DBABLA~BAQCAAAACZA.QA",
                "_cc_id": "37bbd3b5e25518f363099731bc6b4c8e",
                "panoramaId": "f2e101d6354a4fa589dba4a814894945a7021e82bf1f4b9c260e5fab92281dc6",
                "panoramaIdType": "panoIndiv",
                "_gcl_au": "1.1.1896302454.1730073368",
                "_au_1d": "AU1D-0100-001713575418-U5TFIBZP-SOZJ",
                "_gid": "GA1.2.1151398600.1730073386",
                "cnx_userId": "29b1d83be06e472baaefcb22d3ae3611",
                "_li_dcdm_c": ".barchart.com",
                "_lc2_fpi": "0963eb871108--01hvwgn5ebegbbt56a5xrs0n3k",
                "panoramaId_expiry": "1730678237798",
                "webinarClosed": "216",
                "cto_bundle": "WhAAXF9XVHlwJTJCV2pkT3dvRklBM0ZuVGFNaXRYeTZGRXpINXBpVWxkWUExNHhFcjkxYWRtJTJCdkhrR2s4cjNyVG9odEtFRU53eGdEOFc0ZVpmOElUMWdjSjRsZlRqQzA0WTVxJTJCY25QeW5YMzl3ZGJwSW9DJTJCZDJYYUdHWEQyMyUyQm5tMkZpMVJpRm82JTJGcXNqYUJIZVVaOVpvSVZFJTJCNzZ3M1diZEt1eGJtSnozOWExY3A3NUxHQUQ3Ynp1Q1klMkJqR1YlMkJEWms5eWxvSHRaNUZrbkg1U1JWVkFvVTNuWEJ3JTNEJTNE",
                "_awl": "2.1730074105.5-9a5fb48ce841a53bb7cbf467fc09fe58-6763652d75732d6561737431-0",
                "cto_bundle": "rGPslV9XVHlwJTJCV2pkT3dvRklBM0ZuVGFNaXUxck1uZGE3TW96Q0Q2Yjh6YlY1dzV4ZyUyQlhzQndLMDhVb2lQTW5xRDVGTyUyRjBaOVlpRCUyQmpJSHRlM2hmWTVNNmdiMXZ0V1JRJTJGQlVLMmUyWnIwZDdpVkElMkZuVms1VzlYUDJKVk1xbWFuNlNSTzA1YTklMkJXVzRkNSUyRmFRd1RBVVBsOWdlaGRuNWsxVWdlSG9OSDROdlZrM0RmOWRxaGQlMkZJNnlBdzFXNTlTMGtxdU5KTDN4bzduWGJhaDBESDh6OVE3S0hRJTNEJTNE",
                "cto_bidid": "HPdoC19temNLenppZCUyRlV6MlZ6TkF1U2ZBazJRMXVmeURScXQxJTJCdXpabzV6U1l1S2hveHJ6V2czMENDSDBNWVVOZUt2Nlc5aUt0aW1QQmNteWJvRnVHbU8zWVBISiUyRlFYcUY5aW9qTWJmSmJ5T0NZSSUzRA",
                "__gads": "ID=d422c3d1f37af320:T=1730073271:RT=1730074598:S=ALNI_MYPfcjnFlb26OY3jDa8lh6eQhttjg",
                "__gpi": "UID=00000f3b329b8d6b:T=1730073271:RT=1730074598:S=ALNI_MbX0tq_KDBj7JRUe566svZ4KsL5Uw",
                "__eoi": "ID=dbbbc4dd293f08d5:T=1730073271:RT=1730074598:S=AA-AfjZK3mqLQsYEwNslTXbY1AnC",
                "OptanonAlertBoxClosed": "2024-10-28T00:18:33.274Z",
                "_gat_UA-2009749-51": "1",
                "_ga": "GA1.1.134986912.1730073384",
                "_ga_FVWZ0RM4DH": "GS1.1.1730073947.1.0.1730074720.40.0.0",
                "IC_ViewCounter_www.barchart.com": "4",
                "laravel_token": laravel_token,
                "XSRF-TOKEN": xsrf_cookie,
                "laravel_session": laravel_session
            }

            st.write("Headers and Cookies updated successfully.")
    #         st.write("Headers:", headers)
    #         st.write("Cookies:", cookies)

            # Base URL and parameters
            BASE_URL = f"https://www.barchart.com/proxies/timeseries/historical/queryminutes.ashx?symbol={ticker}&interval=30&maxrecords=640&volume=contract&order=asc&dividends=false&backadjust=false&daystoexpiration=1&contractroll=combined"


            response = requests.get(BASE_URL, headers=headers, cookies=cookies)

            st.write(response.status_code)

            if response.status_code == 200:
                try:
                    # Read the response text into a DataFrame with proper headers
                    data = StringIO(response.text)
                    df = pd.read_csv(data, header=None, names=["Datetime", "Unknown", "Open", "High", "Low", "Close", "Volume"])
                    df = df.drop(columns=["Unknown"])  # Drop any unnecessary columns
                    df['Datetime'] = pd.to_datetime(df['Datetime'])  # Convert to datetime format
                except pd.errors.ParserError:
                    print("Failed to parse response as CSV.")
            else:
                print(f"Failed to fetch data. Status Code: {response.status_code}")

            min_timestamp = df['Datetime'].min()

            # Format min_timestamp for the next URL
            formatted_min_timestamp = min_timestamp.strftime('%Y%m%d%H%M%S')

            # Generate the new URL with the min timestamp as the end parameter
            next_url = f"https://www.barchart.com/proxies/timeseries/historical/queryminutes.ashx?symbol={ticker}&interval=30&maxrecords=640&end={formatted_min_timestamp}&volume=contract&order=asc&dividends=false&backadjust=false&daystoexpiration=1&contractroll=combined"

            # Define the recursive data fetching function
            def fetch_data_until_start_date(start_date='2024-01-01'):
                all_data = pd.DataFrame()  # Initialize an empty DataFrame to store all data
                next_url = BASE_URL  # Start with the base URL
                # print(next_url)

                while True:
                    # Fetch data from the API
                    response = requests.get(next_url, headers=headers, cookies=cookies)

                    if response.status_code == 200:
                        try:
                            # Convert response text to DataFrame
                            data = StringIO(response.text)
                            df = pd.read_csv(data, header=None, names=["Datetime", "Unknown", "Open", "High", "Low", "Close", "Volume"])
                            df = df.drop(columns=["Unknown"])  # Drop any unnecessary columns
                            df['Datetime'] = pd.to_datetime(df['Datetime'])  # Convert to datetime format

                            # Append new data to the cumulative DataFrame
                            all_data = pd.concat([all_data, df], ignore_index=True)

                            # Check if the earliest date in the data meets the start date
                            min_timestamp = df['Datetime'].min()
                            if min_timestamp <= pd.to_datetime(start_date):
                                break  # Stop the loop if we have reached or passed the start date

                            # Format the min_timestamp for the next URL
                            formatted_min_timestamp = min_timestamp.strftime('%Y%m%d%H%M%S')

                            # Generate the next URL with the new end parameter
                            next_url = f"{BASE_URL}&end={formatted_min_timestamp}"

                        except pd.errors.ParserError:
                            print("Failed to parse response as CSV.")
                            break  # Exit if parsing fails
                    else:
                        print(f"Failed to fetch data. Status Code: {response.status_code}")
                        break  # Exit if fetching fails
                st.write(all_data.head())
                # Filter the data to only include dates after the start date
                all_data = all_data[all_data['Datetime'] >= pd.to_datetime(start_date)]


                return all_data.reset_index(drop=True)

            # Add a date input widget for selecting the start date
            start_date = st.date_input("Select the Start Date", value=datetime(2024, 1, 1).date())

            # Convert Streamlit date input to string format suitable for the function
            start_date_str = start_date.strftime('%Y-%m-%d')

            # Run the function to fetch data
            temp_final_data = fetch_data_until_start_date(start_date=start_date_str)

            # Assuming 'df' is the initial dataset and 'final_data' contains the recursively fetched data
            merged_data = pd.concat([df, temp_final_data], ignore_index=True)

            # Remove any duplicate rows if necessary, based on 'Datetime' and other columns
            merged_data = merged_data.drop_duplicates(subset=['Datetime'], keep='first')

            # Sort by 'Datetime' in ascending order to maintain chronological order
            merged_data = merged_data.sort_values(by='Datetime').reset_index(drop=True)
            # merged_data = merged_data[merged_data['Datetime'] < pd.to_datetime(input_date)]

            st.write("Merged Data")
            # Display the first few rows of the combined data
            st.write(merged_data.head())
            st.write(merged_data.tail())

            # # List of tickers to fetch data for
            # tickers = ['$SPX', 'DOW', '$NDXT']

            # # # Add a date input widget for selecting the start date
            # # start_date = st.date_input("Select the Start Date", value=datetime(2024, 1, 1).date())

            # # # Convert Streamlit date input to string format suitable for the function
            # # start_date_str = start_date.strftime('%Y-%m-%d')

            # # Create dataframes for each ticker
            # dataframes = {}
            # for ticker in tickers:
            #     dataframes[ticker] = fetch_stock_data(ticker, start_date=start_date_str, headers=headers, cookies=cookies)
            #     st.write(f"Data for {ticker}:")
            #     st.dataframe(dataframes[ticker])

            import time
            # # Fetch data for major indices
            # dji_data = fetch_stock_data('DOW')
            # time.sleep(5)
            # nasdaq_data = fetch_stock_data('$NDXT')
            # time.sleep(5)
            # s_and_p_500_data = fetch_stock_data('$SPX')

            # # List of tickers to fetch data for
            # tickers = ['$SPX', 'DOW', '$NDXT']

            # # # Add a date input widget for selecting the start date
            # # start_date = st.date_input("Select the Start Date", value=datetime(2024, 1, 1).date())

            # # # Convert Streamlit date input to string format suitable for the function
            # # start_date_str = start_date.strftime('%Y-%m-%d')

            # # Create dataframes for each ticker with a delay between fetches
            # dataframes = {}
            # for ticker in tickers:
            #     dataframes[ticker] = fetch_stock_data(ticker, start_date=start_date_str, headers=headers, cookies=cookies)
            #     st.write(f"Data for {ticker}:")
            #     st.dataframe(dataframes[ticker])
            #     time.sleep(5)

            # List of tickers to fetch data for
            tickers = ['$SPX', '$DOWI', '$NDXT']
            
            # tickers = ['AAPL']

            # # Add a date input widget for selecting the start date
            # start_date = st.date_input("Select the Start Date", value=datetime(2024, 1, 1).date())

            # # Convert Streamlit date input to string format suitable for the function
            # start_date_str = start_date.strftime('%Y-%m-%d')

            # Convert Streamlit date input to string format suitable for the function
            start_date_str = start_date.strftime('%Y-%m-%d')

            # Create dataframes for each ticker
            dataframes = {}
            for ind_ticker in tickers:
                time.sleep(5)
                st.write(ind_ticker)

                # Base URL and parameters
                BASE_URL = f"https://www.barchart.com/proxies/timeseries/historical/queryminutes.ashx?symbol={ind_ticker}&interval=30&maxrecords=640&volume=contract&order=asc&dividends=false&backadjust=false&daystoexpiration=1&contractroll=combined"
                response = requests.get(BASE_URL, headers=headers, cookies=cookies)

                st.write(response.status_code)

                if response.status_code == 200:
                    try:
                        # Read the response text into a DataFrame with proper headers
                        data = StringIO(response.text)
                        df = pd.read_csv(data, header=None, names=["Datetime", "Unknown", "Open", "High", "Low", "Close", "Volume"])
                        df = df.drop(columns=["Unknown"])  # Drop any unnecessary columns
                        df['Datetime'] = pd.to_datetime(df['Datetime'])  # Convert to datetime format
                    except pd.errors.ParserError:
                        print("Failed to parse response as CSV.")
                else:
                    print(f"Failed to fetch data. Status Code: {response.status_code}")

                min_timestamp = df['Datetime'].min()

                st.write(min_timestamp)

                # Format min_timestamp for the next URL
                formatted_min_timestamp = min_timestamp.strftime('%Y%m%d%H%M%S')

                # Generate the new URL with the min timestamp as the end parameter
                next_url = f"https://www.barchart.com/proxies/timeseries/historical/queryminutes.ashx?symbol={ind_ticker}&interval=30&maxrecords=640&end={formatted_min_timestamp}&volume=contract&order=asc&dividends=false&backadjust=false&daystoexpiration=1&contractroll=combined"

                # Run the function to fetch data
                temp_final_data = fetch_data_until_start_date(start_date=start_date_str)

                # Assuming 'df' is the initial dataset and 'final_data' contains the recursively fetched data
                complete_data = pd.concat([df, temp_final_data], ignore_index=True)

                # Remove any duplicate rows if necessary, based on 'Datetime' and other columns
                complete_data = complete_data.drop_duplicates(subset=['Datetime'], keep='first')

                # Sort by 'Datetime' in ascending order to maintain chronological order
                complete_data = complete_data.sort_values(by='Datetime').reset_index(drop=True)
                # complete_data = complete_data[complete_data['Datetime'] < pd.to_datetime(input_date)]

                dataframes[ind_ticker] = complete_data

                st.write(complete_data)

                # dataframes[ticker] = fetch_stock_data_barchart(ticker, start_date=start_date_str)
                # st.write(f"Data for {ticker}:")
                # # time.sleep(25)
                # st.dataframe(dataframes[ticker])

            spx_data = dataframes['$SPX']
            dji_data = dataframes['$DOWI']
            nasdaq_data = dataframes['$NDXT']

            st.write("First few rows of ticker")
            st.write(merged_data.head())

            st.write("First few rows of S&P 500 Data:")
            st.dataframe(spx_data.head())

            st.write("First few rows of Dow Jones Industrial Data:")
            st.dataframe(dji_data.head())

            st.write("First few rows of NASDAQ Data:")
            st.dataframe(nasdaq_data.head())

            # # Ensure 'Datetime' column is present and in datetime format
            # for df in [data, dji_data, nasdaq_data, s_and_p_500_data]:
            #     if 'Datetime' in df.columns:
            #         df['Datetime'] = pd.to_datetime(df['Datetime'])
            #     else:
            #         st.error(f"The dataframe for {df} is missing the 'Datetime' column.")
            #         # return

            # Calculate percentage changes in 'Close' prices
            merged_data['pct_change'] = merged_data['Close'].pct_change()
            dji_data['DJI_pct_change'] = dji_data['Close'].pct_change()
            nasdaq_data['NASDAQ_pct_change'] = nasdaq_data['Close'].pct_change()
            spx_data['S_AND_P_500_pct_change'] = spx_data['Close'].pct_change()

            # Align all dataframes to match by timestamp
            merged_data = pd.merge(merged_data[['Datetime', 'Open', 'High', 'Low','Close','Volume','pct_change']],
                                dji_data[['Datetime', 'DJI_pct_change']],
                                on='Datetime', how='inner')

            merged_data = pd.merge(merged_data,
                                nasdaq_data[['Datetime', 'NASDAQ_pct_change']],
                                on='Datetime', how='inner')

            merged_data = pd.merge(merged_data,
                                spx_data[['Datetime', 'S_AND_P_500_pct_change']],
                                on='Datetime', how='inner')

            # Drop NaN values resulting from the percentage change calculation
            merged_data.dropna(inplace=True)

            # Calculate correlations
            correlation_dji = merged_data['pct_change'].corr(merged_data['DJI_pct_change'])
            correlation_nasdaq = merged_data['pct_change'].corr(merged_data['NASDAQ_pct_change'])
            correlation_s_and_p_500 = merged_data['pct_change'].corr(merged_data['S_AND_P_500_pct_change'])

            # Display correlations using Streamlit
            st.write(f"Correlation between {tickers[0]} and DJI movements: {correlation_dji:.2f}")
            st.write(f"Correlation between {tickers[0]} and NASDAQ movements: {correlation_nasdaq:.2f}")
            st.write(f"Correlation between {tickers[0]} and SPX movements: {correlation_s_and_p_500:.2f}")

            st.write(dji_data.head())
            dji_data = calculate_indicators_and_scores(dji_data)
            nasdaq_data = calculate_indicators_and_scores(nasdaq_data)
            spx_data = calculate_indicators_and_scores(spx_data)

            dji_data = dji_data[['Datetime','Open','High','Low','Close','Volume','RSI_Score','%K_Score','Stoch_RSI_Score','CCI_Score','BBP_Score','MA_Score','VWAP_Score','BB_Score','Supertrend_Score','Linear_Regression_Score','Sentiment']]
            nasdaq_data = nasdaq_data[['Datetime','Open','High','Low','Close','Volume','RSI_Score','%K_Score','Stoch_RSI_Score','CCI_Score','BBP_Score','MA_Score','VWAP_Score','BB_Score','Supertrend_Score','Linear_Regression_Score','Sentiment']]
            spx_data = spx_data[['Datetime','Open','High','Low','Close','Volume','RSI_Score','%K_Score','Stoch_RSI_Score','CCI_Score','BBP_Score','MA_Score','VWAP_Score','BB_Score','Supertrend_Score','Linear_Regression_Score','Sentiment']]

            # Rename columns in dji_data to add '_dji' suffix except for the 'Datetime' column
            dji_data = dji_data.rename(columns={col: f"{col}_dji" for col in dji_data.columns if col != 'Datetime'})
            nasdaq_data = nasdaq_data.rename(columns={col: f"{col}_nasdaq" for col in nasdaq_data.columns if col != 'Datetime'})
            spx_data = spx_data.rename(columns={col: f"{col}_s_and_p_500" for col in spx_data.columns if col != 'Datetime'})
            
            index_data = dji_data.copy()
            # Merge both dataframes on the "Datetime" column
            index_data = pd.merge(index_data, nasdaq_data, on='Datetime')
            index_data = pd.merge(index_data, spx_data, on='Datetime')


            final_data = merged_data.copy()

            # st.write(final_data)

            # # Parameters for TPO calculation
            # value_area_percent = 70
            # tick_size = 0.01  # Tick size as per the image

            # # Calculate ATR for 30-minute and daily intervals
            # atr_period = 14

            # # True Range Calculation for 30-minute data
            # data['H-L'] = data['High'] - data['Low']
            # data['H-PC'] = np.abs(data['High'] - data['Close'].shift(1))
            # data['L-PC'] = np.abs(data['Low'] - data['Close'].shift(1))
            # data['TR'] = data[['H-L', 'H-PC', 'L-PC']].max(axis=1)

            # # ATR for 30-minute data
            # data['ATR_14_30_mins'] = data['TR'].ewm(alpha=1/atr_period, adjust=False).mean()



            # # # Fetch daily data for Apple to calculate daily ATR
            # # daily_data = yf.download(ticker, period="60d", interval="1d")

            # start_date = start_date_str
            # end_date = datetime.today().strftime('%Y-%m-%d')

            # # Download data from start_date until today
            # daily_data = yf.download(ticker, start=start_date, end=end_date, interval="1d")

            # # Reset index to ensure 'Date' is a column, not an index
            # daily_data = daily_data.reset_index()

            # # True Range Calculation for daily data
            # daily_data['H-L'] = daily_data['High'] - daily_data['Low']
            # daily_data['H-PC'] = np.abs(daily_data['High'] - daily_data['Close'].shift(1))
            # daily_data['L-PC'] = np.abs(daily_data['Low'] - daily_data['Close'].shift(1))
            # daily_data['TR'] = daily_data[['H-L', 'H-PC', 'L-PC']].max(axis=1)
            # daily_data['ATR_14_1_day'] = daily_data['TR'].ewm(alpha=1/atr_period, adjust=False).mean()
            # # Shift the ATR to get the previous day's ATR
            # daily_data['Prev_Day_ATR_14_1_Day'] = daily_data['ATR_14_1_day'].shift(1)

            # # Ensure consistent date format
            # data['Date'] = pd.to_datetime(data['Datetime']).dt.date
            # daily_data['Date'] = pd.to_datetime(daily_data['Date']).dt.date

            # # Merge ATR from daily data into 30-minute data
            # final_data = pd.merge(data, daily_data[['Date', 'ATR_14_1_day','Prev_Day_ATR_14_1_Day']], on='Date', how='left')

            # final_data = final_data.drop(['H-L', 'H-PC','L-PC','TR'], axis=1)

            # # Round all columns to 2 decimal places
            # final_data = final_data.round(2)

            # # TPO Profile Calculation
            # def calculate_tpo(data, tick_size, value_area_percent):
            #     price_levels = np.arange(data['Low'].min(), data['High'].max(), tick_size)
            #     tpo_counts = defaultdict(list)
            #     letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
            #     letter_idx = 0

            #     for _, row in data.iterrows():
            #         current_letter = letters[letter_idx % len(letters)]
            #         for price in price_levels:
            #             if row['Low'] <= price <= row['High']:
            #                 tpo_counts[price].append(current_letter)
            #         letter_idx += 1

            #     total_tpo = sum(len(counts) for counts in tpo_counts.values())
            #     value_area_target = total_tpo * value_area_percent / 100

            #     sorted_tpo = sorted(tpo_counts.items(), key=lambda x: len(x[1]), reverse=True)
            #     value_area_tpo = 0
            #     value_area_high = 0
            #     value_area_low = float('inf')

            #     for price, counts in sorted_tpo:
            #         if value_area_tpo + len(counts) <= value_area_target:
            #             value_area_tpo += len(counts)
            #             value_area_high = max(value_area_high, price)
            #             value_area_low = min(value_area_low, price)
            #         else:
            #             break

            #     poc = sorted_tpo[0][0]  # Price with highest TPO count
            #     vah = value_area_high
            #     val = value_area_low

            #     return tpo_counts, poc, vah, val

            # # Group by date and calculate TPO profile for each day
            # daily_tpo_profiles = []

            # for date, group in final_data.groupby('Date'):
            #     tpo_counts, poc, vah, val = calculate_tpo(group, tick_size, value_area_percent)

            #     # Calculate Initial Balance Range (IBR)
            #     initial_balance_high = group['High'].iloc[:2].max()  # First 2 half-hour periods (1 hour)
            #     initial_balance_low = group['Low'].iloc[:2].min()
            #     initial_balance_range = initial_balance_high - initial_balance_low

            #     # Calculate day's total range
            #     day_range = group['High'].max() - group['Low'].min()
            #     range_extension = day_range - initial_balance_range

            #     # Identify Single Prints
            #     single_prints = [round(price, 2) for price, counts in tpo_counts.items() if len(counts) == 1]

            #     # Identify Poor Highs and Poor Lows
            #     high_price = group['High'].max()
            #     low_price = group['Low'].min()
            #     poor_high = len(tpo_counts[high_price]) > 1
            #     poor_low = len(tpo_counts[low_price]) > 1

            #     # Classify the day
            #     if day_range <= initial_balance_range * 1.15:
            #         day_type = 'Normal Day'
            #     elif initial_balance_range < day_range <= initial_balance_range * 2:
            #         day_type = 'Normal Variation Day'
            #     elif day_range > initial_balance_range * 2:
            #         day_type = 'Trend Day'
            #     else:
            #         day_type = 'Neutral Day'

            #     if last_row['Close'] >= initial_balance_high:
            #         close_type = 'Closed above Initial High'
            #     elif last_row['Close'] <= initial_balance_low:
            #         close_type = 'Closed below Initial Low'
            #     else:
            #         close_type = 'Closed between Initial High and Low'


            #     # Store the results in a dictionary
            #     tpo_profile = {
            #         'Date': date,
            #         'POC': round(poc, 2),
            #         'VAH': round(vah, 2),
            #         'VAL': round(val, 2),
            #         'Initial Balance High': round(initial_balance_high, 2),
            #         'Initial Balance Low': round(initial_balance_low, 2),
            #         'Initial Balance Range': round(initial_balance_range, 2),
            #         'Day Range': round(day_range, 2),
            #         'Range Extension': round(range_extension, 2),
            #         'Day Type': day_type,
            #         'Single Prints': single_prints,
            #         'Poor High': poor_high,
            #         'Poor Low': poor_low,
            #         'Close Type' : close_type
            #     }
            #     daily_tpo_profiles.append(tpo_profile)

            # # Convert the list of dictionaries to a DataFrame
            # tpo_profiles_df = pd.DataFrame(daily_tpo_profiles)

            # # Merge TPO profile data into final_data based on the 'Date'
            # final_data = pd.merge(final_data, tpo_profiles_df, on='Date', how='left')

            # # Calculate Moving Average (MA)
            # final_data['MA_20'] = final_data['Close'].rolling(window=20).mean()

            # # Calculate Relative Strength Index (RSI)
            # delta = final_data['Close'].diff()
            # gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            # loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            # final_data['RSI'] = 100 - (100 / (1 + gain / loss))

            # # Calculate Moving Average Convergence Divergence (MACD)
            # short_ema = final_data['Close'].ewm(span=12, adjust=False).mean()
            # long_ema = final_data['Close'].ewm(span=26, adjust=False).mean()
            # final_data['MACD'] = short_ema - long_ema
            # final_data['Signal Line'] = final_data['MACD'].ewm(span=9, adjust=False).mean()

            # # Calculate Bollinger Bands
            # final_data['MA_20'] = final_data['Close'].rolling(window=20).mean()
            # final_data['Bollinger_Upper'] = final_data['MA_20'] + (final_data['Close'].rolling(window=20).std() * 2)
            # final_data['Bollinger_Lower'] = final_data['MA_20'] - (final_data['Close'].rolling(window=20).std() * 2)

            # # Calculate Volume Weighted Average Price (VWAP)
            # final_data['VWAP'] = (final_data['Volume'] * (final_data['High'] + final_data['Low'] + final_data['Close']) / 3).cumsum() / final_data['Volume'].cumsum()

            # # Calculate Fibonacci Retracement Levels (use high and low from a specific range if applicable)
            # highest = final_data['High'].max()
            # lowest = final_data['Low'].min()
            # final_data['Fib_38.2'] = highest - (highest - lowest) * 0.382
            # final_data['Fib_50'] = (highest + lowest) / 2
            # final_data['Fib_61.8'] = highest - (highest - lowest) * 0.618

            # # Calculate Average True Range (ATR)
            # high_low = final_data['High'] - final_data['Low']
            # high_close = np.abs(final_data['High'] - final_data['Close'].shift())
            # low_close = np.abs(final_data['Low'] - final_data['Close'].shift())
            # true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            # final_data['ATR'] = true_range.rolling(window=14).mean()

            # # Calculate Stochastic Oscillator
            # final_data['14-high'] = final_data['High'].rolling(window=14).max()
            # final_data['14-low'] = final_data['Low'].rolling(window=14).min()
            # final_data['%K'] = (final_data['Close'] - final_data['14-low']) * 100 / (final_data['14-high'] - final_data['14-low'])
            # final_data['%D'] = final_data['%K'].rolling(window=3).mean()

            # # Calculate Parabolic SAR (for simplicity, this example uses a fixed acceleration factor)
            # final_data['PSAR'] = final_data['Close'].shift() + (0.02 * (final_data['High'] - final_data['Low']))

            # # Determine the trend for each day based on the previous day's levels
            # trends = []
            # for i in range(1, len(tpo_profiles_df)):
            #     previous_day = tpo_profiles_df.iloc[i - 1]
            #     current_day = tpo_profiles_df.iloc[i]

            #     if current_day['Initial Balance High'] > previous_day['VAH']:
            #         trend = 'Bullish'
            #     elif current_day['Initial Balance Low'] < previous_day['VAL']:
            #         trend = 'Bearish'
            #     else:
            #         trend = 'Neutral'

            #     trend_entry = {
            #         'Date': current_day['Date'],
            #         'Trend': trend,
            #         'Previous Day VAH': round(previous_day['VAH'], 2),
            #         'Previous Day VAL': round(previous_day['VAL'], 2),
            #         'Previous Day POC': round(previous_day['POC'], 2),
            #     }

            #     trends.append(trend_entry)

            # # Convert the trends list to a DataFrame
            # trends_df = pd.DataFrame(trends)

            # # Merge trend data into final_data
            # final_data = pd.merge(final_data, trends_df, on='Date', how='left')

            # # Define the conditions for Initial Balance Range classification
            # conditions = [
            #     final_data['Initial Balance Range'] < final_data['Prev_Day_ATR_14_1_Day'] / 3,
            #     (final_data['Initial Balance Range'] >= final_data['Prev_Day_ATR_14_1_Day'] / 3) & 
            #     (final_data['Initial Balance Range'] <= final_data['Prev_Day_ATR_14_1_Day']),
            #     final_data['Initial Balance Range'] > final_data['Prev_Day_ATR_14_1_Day']
            # ]

            # # Define the corresponding values for each condition
            # choices = ['Small', 'Medium', 'Large']

            # # Create the IB Range column using np.select()
            # final_data['IB Range'] = np.select(conditions, choices, default='Unknown')

            # # Round all values in final_data to 2 decimals
            # final_data = final_data.round(2)

            # # Initialize an empty list to store each day's input text and trend
            # training_data = []

            # final_data = final_data.sort_values(by='Datetime')

            # # Get the unique dates and sort them
            # sorted_dates = sorted(set(final_data['Date']))
            # final_data['2 Day VAH and VAL'] = ''
            # # final_data['Previous Initial Market Profile Shape'] = ''
            # # final_data['Previous Refined Market Profile Shape'] = ''

            # # Iterate over the sorted dates by index, starting from the third day to have data for previous two days
            # for i in range(2, len(sorted_dates)):
            #     date = sorted_dates[i]
            #     previous_date = sorted_dates[i - 1]
            #     two_days_ago = sorted_dates[i - 2]

            #     # Extract data for the current date and previous dates
            #     current_data = final_data[final_data['Date'] == date]
            #     previous_data = final_data[final_data['Date'] == previous_date]
            #     two_days_ago_data = final_data[final_data['Date'] == two_days_ago]

            #     # Calculate the maximum high and minimum low for the previous day
            #     day_high = previous_data['High'].max()
            #     day_low = previous_data['Low'].min()

            #     # Initialize an empty list for actions based on previous day's close and VAH/VAL comparisons
            #     actions = []

            #     if not previous_data.empty:
            #         last_row = previous_data.iloc[-1]

            #         # Determine close position relative to VAH and VAL
            #         if last_row['Close'] >= last_row['VAH']:
            #             actions.append('Previous Day Close Above VAH')
            #             actions.append('Previous Day Close Bullish')
            #         elif last_row['Close'] <= last_row['VAL']:
            #             actions.append('Previous Day Close Below VAL')
            #             actions.append('Previous Day Close Bearish')
            #         else:
            #             actions.append('Previous Day Close Neutral')

            #         # Insider/Outsider neutral positioning based on VAH/VAL ranges
            #         if last_row['Previous Day VAH'] >= last_row['VAH'] and last_row['Previous Day VAL'] <= last_row['VAL']:
            #             actions.append('Insider Neutral')
            #         elif last_row['Previous Day VAH'] <= last_row['VAH'] and last_row['Previous Day VAL'] >= last_row['VAL']:
            #             actions.append('Outsider Neutral')

            #         # Determine day type based on Initial Balance range and close
            #         if last_row['IB Range'] == 'Large' and last_row['Close'] <= last_row['Initial Balance High']:
            #             final_day_type = 'Large Range Normal Day'
            #         elif last_row['IB Range'] == 'Medium' and day_high >= last_row['Initial Balance High'] and day_low <= last_row['Initial Balance Low']:
            #             final_day_type = 'Medium Range Neutral Day'
            #         elif last_row['IB Range'] == 'Medium' and last_row['Close'] >= last_row['Initial Balance High']:
            #             final_day_type = 'Medium Range +ve Normal Variation Day'
            #         elif last_row['IB Range'] == 'Medium' and last_row['Close'] <= last_row['Initial Balance Low']:
            #             final_day_type = 'Medium Range -ve Normal Variation Day'
            #         elif last_row['IB Range'] == 'Small' and last_row['Close'] >= last_row['Initial Balance High']:
            #             final_day_type = 'Small Range +ve Trend Variation Day'
            #         elif last_row['IB Range'] == 'Small' and last_row['Close'] <= last_row['Initial Balance Low']:
            #             final_day_type = 'Small Range -ve Trend Variation Day'
            #         elif last_row['IB Range'] == 'Small' and last_row['Close'] <= last_row['Initial Balance High'] and last_row['Close'] >= last_row['Initial Balance Low']:
            #             final_day_type = 'Small Range Non Trend Variation Day'
            #         else:
            #             final_day_type = ''

            #     # Calculate the opening price and difference from the previous close
            #     opening_price = current_data.iloc[0]['Open']
            #     previous_close = previous_data.iloc[-1]['Close']
            #     open_point_diff = round(opening_price - previous_close, 2) if previous_close else None
            #     open_percent_diff = round((open_point_diff / previous_close) * 100, 2) if previous_close else None
            #     open_above_below = "above" if open_point_diff > 0 else "below" if open_point_diff < 0 else "no change"

            #     current_row = current_data.iloc[0]

            # #     # Generate the LLM input text
            # #     input_text = (
            # #         f"Today’s profile on {date} for {ticker} indicates an {current_row['IB Range']} Range. The market opened at {opening_price}, "
            # #         f"which is {open_percent_diff}% ({abs(open_point_diff)} points) {open_above_below} the previous day's close. "
            # #         f"The Initial Balance High is {current_row['Initial Balance High']} and Low is {current_row['Initial Balance Low']}, "
            # #         f"giving an Initial Balance Range of {current_row['Initial Balance Range']}. "
            # #         f"Yesterday's VAH was {last_row['VAH']} and VAL was {last_row['VAL']}. "
            # #         f"Day before yesterday's VAH was {last_row['Previous Day VAH']} and VAL was {last_row['Previous Day VAL']}. "
            # #         f"Previous day Type : {last_row['Day Type']}\n"
            # #         f"Previous Adjusted Day Type : {final_day_type}\n"
            # #         f"Previous Close Type : {last_row['Close Type']}\n"
            # #         f"Previous 2 Day VAH and VAL : {actions}. "
            # #         f"Given these indicators, what is the expected market direction for tomorrow?"
            # #     )

            #     # Generate the LLM input text with added indicators
            #     input_text = (
            #         f"Today’s profile on {date} for {ticker} with IB Range Type {current_row['IB Range']} Range. The market opened at {opening_price}, "
            #         f"Opening_Gap_Percentage is {open_percent_diff}% ( Opening_Gap_Points {abs(open_point_diff)} points) {open_above_below} the previous day's close. "
            #         f"The Initial Balance High is {current_row['Initial Balance High']} and Initial Balance Low is {current_row['Initial Balance Low']}, "
            #         f"giving an Initial Balance Range of {current_row['Initial Balance Range']}. "
            #         f"Yesterday's VAH was {last_row['VAH']} and Yesterday's VAL was {last_row['VAL']}. "
            #         f"Day before yesterday's VAH was {last_row['Previous Day VAH']} and Day before yesterday's VAL was {last_row['Previous Day VAL']}. "
            #         f"Previous day Type: {last_row['Day Type']}.\n"
            #         f"Previous Adjusted Day Type: {final_day_type}.\n"
            #         f"Previous Close Type: {last_row['Close Type']}.\n"
            #         f"Previous 2 Day VAH and VAL: {actions}.\n"

            #         # Adding indicators
            #         f"MA_20_Day is {last_row['MA_20']}. "
            #         f"RSI is {last_row['RSI']}. "
            #         f"MACD is {last_row['MACD']} with Signal Line at {last_row['Signal Line']}. "
            #         f"Bollinger Bands Upper at {last_row['Bollinger_Upper']} and Bollinger Bands Lower at {last_row['Bollinger_Lower']}. "
            #         f"VWAP is {last_row['VWAP']}. "
            #         f"Fibonacci Levels: 38.2% at {last_row['Fib_38.2']}, 50% at {last_row['Fib_50']}, 61.8% at {last_row['Fib_61.8']}. "
            #         f"ATR is {last_row['ATR']}. "
            #         f"Stochastic Oscillator %K is {last_row['%K']} and %D is {last_row['%D']}. "
            #         f"Parabolic SAR is at {last_row['PSAR']}. "

            #         f"Given these indicators, what is the expected market direction for today?"
            #     )

            #     print(input_text)

            #     current_day_close = current_data.iloc[-1]['Close']
            #     previous_day_high = previous_data['High'].max()
            #     previous_day_low = previous_data['Low'].max()

            #     result = ''
            #     if current_day_close >= previous_close:
            #         result += 'The stock closed above yesterday close \n'
            #     else:
            #         result += 'The stock closed below yesterday close \n'
            #     if current_day_close >= previous_day_high:
            #         result += 'The stock closed above Previous Day High \n'
            #     else:
            #         result += 'The stock closed below Previous Day High \n'

            #     if current_day_close >= previous_day_low:
            #         result += 'The stock closed above Previous Day Low \n'
            #     else:
            #         result += 'The stock closed below Previous Day Low \n'

            #     if current_day_close >= current_row['Initial Balance High']:
            #         result += 'The stock closed above Initial Balance High \n'
            #     else:
            #         result += 'The stock closed below Initial Balance High \n'

            #     if current_day_close >= current_row['Initial Balance Low']:
            #         result += 'The stock closed above Initial Balance Low \n'
            #     else:
            #         result += 'The stock closed below Initial Balance Low \n'

            #     # Get the trend (output) for the current date
            #     trend = current_data.iloc[-1]['Trend'] if 'Trend' in current_data.columns else None

            #     # Append the input-output pair to the training data list
            #     training_data.append({
            #         'Date': date,
            #         'Input Text': input_text,
            #         'Trend': trend,
            #         'Result': result
            #     })

            # # Convert the training data list to a DataFrame
            # training_data_df = pd.DataFrame(training_data)

            st.write(final_data)

            def get_price_distribution_for_date_updated(data):
                last_300_candles = data.tail(300)
                high_300 = last_300_candles['High'].max()
                low_300 = last_300_candles['Low'].min()
                return {
                    'data': data,
                    'high_300': high_300,
                    'low_300': low_300
                }

            distribution = get_price_distribution_for_date_updated(final_data)

            HighValue = distribution['high_300']
            LowValue = distribution['low_300']

            MinimumTick = 0.01
            RowsRequired = 80

            MinTickRange = (HighValue - LowValue) / MinimumTick
            RowTicks = MinTickRange / RowsRequired

            if 1 <= RowTicks <= 100:
                increment = 5
            elif 100 <= RowTicks <= 1000:
                increment = 50
            elif 1000 <= RowTicks <= 10000:
                increment = 500
            elif 10000 <= RowTicks <= 100000:
                increment = 5000
            else:
                increment = 50000

            ticks_per_row = round(RowTicks / increment) * increment

            tick_size = ticks_per_row * MinimumTick

            # Function to calculate ticks and determine adjusted high and low
            def calculate_ticks(high, low, tick_size):
                high_floor = math.floor(high / tick_size) * tick_size
                high_ceil = math.ceil(high / tick_size) * tick_size
                low_floor = math.floor(low / tick_size) * tick_size
                low_ceil = math.ceil(low / tick_size) * tick_size
                adjusted_high = high_ceil
                adjusted_low = low_floor
                return {
                    "adjusted_high": adjusted_high,
                    "adjusted_low": adjusted_low,
                }

            # Adjust high and low values in the data
            distribution['data']['adjusted_high'] = distribution['data'].apply(lambda row: calculate_ticks(row['High'], row['Low'], tick_size)['adjusted_high'], axis=1)
            distribution['data']['adjusted_low'] = distribution['data'].apply(lambda row: calculate_ticks(row['High'], row['Low'], tick_size)['adjusted_low'], axis=1)

            # Calculate TPO based on adjusted high and low
            def calculate_tpo(data, tick_size):
                data['Date'] = data['Datetime'].dt.date
                grouped = data.groupby('Date')
                
                tpo_counts = defaultdict(list)
                letter_counts = defaultdict(lambda: {'count': 0, 'letters': set(),'range_low':[],'range_high':[]})
                
                for date, group in grouped:
                    letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
                    letter_idx = 0
                    
                    for index, row in group.iterrows():
                        current_letter = letters[letter_idx % len(letters)]
                        letter_idx += 1
                        
                        # Generate price levels within the adjusted range and round to avoid floating-point precision issues
                        price_levels = np.arange(row['adjusted_low'], row['adjusted_high'] + tick_size, tick_size)
                        price_levels = np.round(price_levels, 2)  # Round to two decimal places
                        
                        # Filter to include only prices within the specified range
                        price_levels = price_levels[(price_levels >= row['adjusted_low'] - 0.00001) & 
                                                    (price_levels <= row['adjusted_high'] + 0.00001)]
                        
                        for i in range(len(price_levels)):
                            if i == len(price_levels) - 1:
                                continue
                            else:
                                price = price_levels[i]
                                tpo_counts[(date, price)].append(current_letter)
                                letter_counts[(date, price)]['count'] += 1
                                letter_counts[(date, price)]['letters'].add(current_letter)
                                letter_counts[(date, price)]['range_low'].append(price)
                                letter_counts[(date, price)]['range_high'].append(price_levels[i+1])
                            
                return tpo_counts, letter_counts

            tpo_counts, letter_counts = calculate_tpo(distribution['data'], tick_size)

            # Create DataFrame with TPO counts and letters
            ranges = []

            for date, group in distribution['data'].groupby('Date'):
                sorted_prices = sorted([key for key in letter_counts.keys() if key[0] == date], key=lambda x: x[1])
                
                for i in range(len(sorted_prices)):
                    price = sorted_prices[i][1]
                    filter_tpo = letter_counts[(date, price)]
                    next_price = filter_tpo['range_high'][0] if filter_tpo['range_high'] else price

                    ranges.append({
                        'Date': date,
                        'Range_Low': price,
                        'Range_High': next_price,
                        'Letter_Count': filter_tpo['count'],
                        'Letters': ''.join(sorted(filter_tpo['letters']))
                    })

            range_df = pd.DataFrame(ranges)

            # Identify Single Footprint Rows
            range_df['Single_Footprint'] = range_df['Letter_Count'] == 1

            # Calculate Initial Balance (IB) High, Low, and Range using the first two rows of each day
            ib_data = distribution['data'].groupby('Date').head(2)

            # Create IB High, Low, and Range based on first two rows
            ib_high_low = ib_data.groupby('Date').agg(
                IB_High=('High', 'max'),
                IB_Low=('Low', 'min')
            )

            # Assign the IB values to the ib_data DataFrame
            ib_data = distribution['data'].groupby('Date').first()
            ib_data['IB_High'] = ib_high_low['IB_High']
            ib_data['IB_Low'] = ib_high_low['IB_Low']
            ib_data['IB_Range'] = ib_data['IB_High'] - ib_data['IB_Low']

            # Merge IB data with TPO ranges
            final_df = pd.merge(range_df, ib_data[['IB_High', 'IB_Low', 'IB_Range']], left_on='Date', right_index=True, how='left')

            # Calculate the POC (Point of Control)
            poc_list = []
            for date, group in range_df.groupby('Date'):
                max_tpo_count = group['Letter_Count'].max()
                max_tpo_rows = group[group['Letter_Count'] == max_tpo_count]
                
                if len(max_tpo_rows) > 1:
                    middle_index = len(group) // 2
                    max_tpo_rows['Distance_From_Middle'] = abs(max_tpo_rows.index - middle_index)
                    closest_row = max_tpo_rows.sort_values(by=['Distance_From_Middle', 'Range_Low']).iloc[0]
                else:
                    closest_row = max_tpo_rows.iloc[0]
                
                poc = round((closest_row['Range_Low'] + closest_row['Range_High']) / 2, 2)
                poc_list.append({'Date': date, 'POC': poc})

            poc_df = pd.DataFrame(poc_list)

            # Merge POC data with range data
            final_df = pd.merge(final_df, poc_df, on='Date', how='left')

            # Decide which boundary to use for VAH and VAL
            use_low = False

            # Calculate VAH and VAL
            vah_val_list = []

            for date, group in final_df.groupby('Date'):
                group = group.sort_values(by='Range_Low').reset_index(drop=True)
                
                total_tpos = group['Letter_Count'].sum()
                value_area_threshold = total_tpos * 0.7
                
                poc_row = group.loc[(group['Range_Low'] <= group['POC']) & (group['Range_High'] >= group['POC'])]
                if poc_row.empty:
                    continue

                poc_index = poc_row.index[0]
                included_tpos = group.loc[poc_index, 'Letter_Count']
                vah_index, val_index = poc_index, poc_index

                while included_tpos < value_area_threshold:
                    up_index = vah_index + 1 if vah_index + 1 < len(group) else None
                    down_index = val_index - 1 if val_index - 1 >= 0 else None
                    
                    if up_index is not None and down_index is not None:
                        if group.loc[up_index, 'Letter_Count'] > group.loc[down_index, 'Letter_Count']:
                            included_tpos += group.loc[up_index, 'Letter_Count']
                            vah_index = up_index
                        elif group.loc[up_index, 'Letter_Count'] < group.loc[down_index, 'Letter_Count']:
                            included_tpos += group.loc[down_index, 'Letter_Count']
                            val_index = down_index
                        else:
                            included_tpos += group.loc[up_index, 'Letter_Count']
                            included_tpos += group.loc[down_index, 'Letter_Count']
                            vah_index = up_index
                            val_index = down_index
                    elif up_index is not None:
                        included_tpos += group.loc[up_index, 'Letter_Count']
                        vah_index = up_index
                    elif down_index is not None:
                        included_tpos += group.loc[down_index, 'Letter_Count']
                        val_index = down_index
                    else:
                        break

                if use_low:
                    vah = group.loc[vah_index, 'Range_Low']
                    val = group.loc[val_index, 'Range_Low']
                else:
                    vah = group.loc[vah_index, 'Range_High']
                    val = group.loc[val_index, 'Range_High']

                vah_val_list.append({'Date': date, 'VAH': vah, 'VAL': val})

            vah_val_df = pd.DataFrame(vah_val_list)

            vah_val_df.rename(columns={'VAH': 'Predicted_VAH', 'VAL': 'Predicted_VAL'}, inplace=True)

            # Merge with the original DataFrame to get all columns
            final_df = pd.merge(final_df, vah_val_df, on='Date', how='left')
            # final_df = pd.merge(final_df, distribution['data'][['Datetime', 'Open', 'High', 'Low', 'Close', 'Volume', 'adjusted_high', 'adjusted_low']], left_on='Date', right_on=distribution['data']['Datetime'].dt.date, how='left')

            # # Display final DataFrame
            # print(final_df)

            # Sort the dataframe by Date and Range_Low
            filtered_final_df = final_df[final_df['Single_Footprint'] == True]

            filtered_final_df['Date'] = pd.to_datetime(filtered_final_df['Date'])
            filtered_final_df = filtered_final_df.sort_values(by=['Date', 'Range_Low'])

            # Function to group continuous ranges
            def group_continuous_ranges(df):
                grouped_rows = []
                
                # Iterate through each group by date
                for date, group in df.groupby('Date'):
                    group = group.sort_values(by='Range_Low')
                    ranges = []
                    start = group.iloc[0]['Range_Low']
                    end = group.iloc[0]['Range_High']

                    for i in range(1, len(group)):
                        current_low = group.iloc[i]['Range_Low']
                        current_high = group.iloc[i]['Range_High']

                        if current_low == end:
                            end = current_high
                        else:
                            ranges.append(f"[{start} - {end}]")
                            start = current_low
                            end = current_high

                    ranges.append(f"[{start} - {end}]")
                    grouped_rows.append([date, ', '.join(ranges)])

                return pd.DataFrame(grouped_rows, columns=['Date', 'Single_Footprint_Ranges'])

            # Apply the grouping function
            grouped_df = group_continuous_ranges(filtered_final_df)

            # st.write("grouped_df")
            # st.write(grouped_df)

            # # Display the grouped dataframe
            # print(grouped_df)

            # Create an empty list to store rows for the new dataframe
            poc_data_list = []

            # st.write(final_df)

            distribution['data']['Date'] = pd.to_datetime(distribution['data']['Date']).dt.date
            grouped_df['Date'] = pd.to_datetime(grouped_df['Date']).dt.date
            final_df['Date'] = pd.to_datetime(final_df['Date']).dt.date

            # Iterate through each date's group in distribution['data']
            for date, group in distribution['data'].groupby('Date'):
                poor_low = False
                poor_high = False
                
                # Convert the current date to datetime format to match the final_df format
                # date = pd.to_datetime(date)
                # st.write(date)
                # Filter the final_df by the current date
                filtered_date_df = final_df[final_df['Date'] == date]
                filtered_grouped_single_fp_df = grouped_df[grouped_df['Date'] == date]

                # st.write(filtered_date_df)
                # st.write(filtered_grouped_single_fp_df)
                
                # Check if the condition is True for any row in the filtered dataframe
                if (filtered_date_df[filtered_date_df['Range_Low'] == filtered_date_df['Range_Low'].min()]['Letter_Count'] > 1).any():
                    poor_low = True
                if (filtered_date_df[filtered_date_df['Range_High'] == filtered_date_df['Range_High'].max()]['Letter_Count'] > 1).any():
                    poor_high = True
                
                # print(filtered_date_df)
            #     print(filtered_grouped_single_fp_df)
                
                # Extract unique values of IB_High and IB_Low for the current date
                IB_High_unique_values = filtered_date_df['IB_High'].unique() if 'IB_High' in filtered_date_df.columns else []
                IB_Low_unique_values = filtered_date_df['IB_Low'].unique() if 'IB_Low' in filtered_date_df.columns else []
                IB_Range_unique_values = filtered_date_df['IB_Range'].unique() if 'IB_Range' in filtered_date_df.columns else []
                POC_values = filtered_date_df['POC'].unique() if 'POC' in filtered_date_df.columns else []
                VAH_values = filtered_date_df['Predicted_VAH'].unique() if 'Predicted_VAH' in filtered_date_df.columns else []
                VAL_values = filtered_date_df['Predicted_VAL'].unique() if 'Predicted_VAL' in filtered_date_df.columns else []
                
                Single_Foot_Print_values = filtered_grouped_single_fp_df['Single_Footprint_Ranges'].unique() if 'Single_Footprint_Ranges' in filtered_grouped_single_fp_df.columns else []
                
                

                # Iterate over each row in the group
                for index, row in group.iterrows():
                    # Append a dictionary for each row with the required fields to the list
                    poc_data_list.append({
                        'Datetime': row['Datetime'],
                        'Date':date,
                        'Open': row['Open'],
                        'High': row['High'],
                        'Low': row['Low'],
                        'Close': row['Close'],
                        'Volume': row.get('Volume', None),  # Using .get() to handle the case if 'Volume' is not present in the row
                        'IB_Low': IB_Low_unique_values[0] if len(IB_Low_unique_values) > 0 else None,  # Assuming to use the first unique value
                        'IB_High': IB_High_unique_values[0] if len(IB_High_unique_values) > 0 else None,  # Assuming to use the first unique value
                        'IB_Range': IB_Range_unique_values[0] if len(IB_Range_unique_values) > 0 else None,  # Assuming to use the first unique value
                        'POC': POC_values[0] if len(POC_values) > 0 else None,  
                        'VAH': VAH_values[0] if len(VAH_values) > 0 else None,  
                        'VAL': VAL_values[0] if len(VAL_values) > 0 else None,  
                        'Single_FootPrint':Single_Foot_Print_values[0] if len(Single_Foot_Print_values) > 0 else None,
                        'Poor_Low':poor_low,
                        'Poor_High':poor_high,
                    })

            # Create a DataFrame from the list of dictionaries
            final_poc_data = pd.DataFrame(poc_data_list)

            final_poc_data['Date'] = pd.to_datetime(final_poc_data['Date']).dt.date

            # Print the new dataframe to verify its contents
            st.write(final_poc_data)

            day_ranges = []
            day_types = []
            close_types = []


            for date, group in final_poc_data.groupby('Date'):
                day_type = ''
                close_type = ''

                # st.write(group)
                
                # Calculate day's total range
                day_range = round(group['High'].max() - group['Low'].min(),2)
                
                initial_balance_range = round(group['IB_Range'].iloc[1],2)
                
                # Classify the day
                if day_range <= initial_balance_range * 1.15:
                    day_type = 'Normal Day'
                elif initial_balance_range < day_range <= initial_balance_range * 2:
                    day_type = 'Normal Variation Day'
                elif day_range > initial_balance_range * 2:
                    day_type = 'Trend Day'
                else:
                    day_type = 'Neutral Day'
                
                last_row = group.iloc[-1]
                if last_row['Close'] >= last_row['IB_High']:
                    close_type = 'Closed above Initial High'
                elif last_row['Close'] <= last_row['IB_Low']:
                    close_type = 'Closed below Initial Low'
                else:
                    close_type = 'Closed between Initial High and Low'
                    
                day_ranges.append(day_range)
                day_types.append(day_type)
                close_types.append(close_type)
                    
            # Assign the calculated values back to the dataframe
            final_poc_data['Day_Range'] = final_poc_data['Date'].map(dict(zip(final_poc_data['Date'].unique(), day_ranges)))
            final_poc_data['Day_Type'] = final_poc_data['Date'].map(dict(zip(final_poc_data['Date'].unique(), day_types)))
            final_poc_data['Close_Type'] = final_poc_data['Date'].map(dict(zip(final_poc_data['Date'].unique(), close_types)))

            final_poc_data = final_poc_data.sort_values('Datetime',ascending=True)

            st.write(ticker)
            # Download daily data for the ATR calculation
            daily_data = yf.download(ticker, start=start_date_str, interval="1d").reset_index()

            # Convert 'Date' column in both dataframes to datetime format
            final_poc_data['Date'] = pd.to_datetime(final_poc_data['Date'])
            daily_data['Date'] = pd.to_datetime(daily_data['Date'])

            # Calculate ATR for 30-minute and daily intervals using TA-Lib
            atr_period = 14

            st.write(daily_data)

            # ATR for daily data using TA-Lib
            daily_data['ATR_14_1_day'] = talib.ATR(daily_data['High'], daily_data['Low'], daily_data['Close'], timeperiod=atr_period)
            daily_data['Prev_Day_ATR_14_1_Day'] = daily_data['ATR_14_1_day'].shift(1)

            # Merge ATR from daily data into 30-minute data
            final_poc_data = pd.merge(final_poc_data, daily_data[['Date', 'ATR_14_1_day', 'Prev_Day_ATR_14_1_Day']], on='Date', how='left')

            # Calculate ATR for 30-minute data using TA-Lib
            final_poc_data['ATR_14_30_mins'] = talib.ATR(final_poc_data['High'], final_poc_data['Low'], final_poc_data['Close'], timeperiod=atr_period)

            final_poc_data['ATR'] = final_poc_data['ATR_14_30_mins']

            # Drop unnecessary columns from earlier calculations
            final_poc_data = final_poc_data.drop(['H-L', 'H-PC', 'L-PC', 'TR'], axis=1, errors='ignore')

            # Calculate Moving Average (MA) using TA-Lib
            final_poc_data['MA_20'] = talib.SMA(final_poc_data['Close'], timeperiod=20)

            # Calculate RSI using TA-Lib with a 14-day period
            final_poc_data['RSI'] = talib.RSI(final_poc_data['Close'], timeperiod=14)

            # Calculate Moving Average Convergence Divergence (MACD) using TA-Lib
            final_poc_data['MACD'], final_poc_data['Signal Line'], _ = talib.MACD(final_poc_data['Close'], fastperiod=12, slowperiod=26, signalperiod=9)

            # Calculate Bollinger Bands using TA-Lib
            final_poc_data['MA_20_BB'], final_poc_data['Bollinger_Upper'], final_poc_data['Bollinger_Lower'] = talib.BBANDS(
                final_poc_data['Close'], timeperiod=20, nbdevup=2, nbdevdn=2, matype=0
            )

            # # Calculate Volume Weighted Average Price (VWAP)
            # # VWAP calculation not available in TA-Lib. Calculating manually.
            # typical_price = (final_poc_data['High'] + final_poc_data['Low'] + final_poc_data['Close']) / 3
            # final_poc_data['VWAP'] = (final_poc_data['Volume'] * typical_price).cumsum() / final_poc_data['Volume'].cumsum()

            # # Set up parameters
            # band_multipliers = [1, 2, 3]  # Equivalent to Bands Multiplier #1, #2, and #3
            # source = (final_poc_data['High'] + final_poc_data['Low'] + final_poc_data['Close']) / 3  # Equivalent to 'hlc3'

            # # Calculate the cumulative VWAP
            # typical_price = source
            # vwap = (final_poc_data['Volume'] * typical_price).cumsum() / final_poc_data['Volume'].cumsum()
            # final_poc_data['VWAP'] = vwap

            final_poc_data['Date'] = pd.to_datetime(final_poc_data['Date'])

            # Define a function to calculate VWAP for each group
            def calculate_vwap(df):
                typical_price = (df['High'] + df['Low'] + df['Close']) / 3
                vwap = (typical_price * df['Volume']).cumsum() / df['Volume'].cumsum()
                return vwap

            grouped = final_poc_data.groupby(final_poc_data['Date'].dt.date)

            # Apply the VWAP calculation for each group
            final_poc_data['VWAP'] = grouped.apply(lambda x: calculate_vwap(x)).reset_index(level=0, drop=True)

            # # Calculate the rolling standard deviation of the typical price for each group
            # final_poc_data['Typical_Price'] = (final_poc_data['High'] + final_poc_data['Low'] + final_poc_data['Close']) / 3
            # final_poc_data['Rolling_STD'] = grouped['Typical_Price'].apply(lambda x: x.rolling(window=20).std()).reset_index(level=0, drop=True)

            # # Calculate upper and lower bands based on standard deviation multipliers
            # band_multipliers = [1, 2, 3]  # Equivalent to Bands Multiplier #1, #2, and #3
            # for i, multiplier in enumerate(band_multipliers, start=1):
            #     final_poc_data[f'VWAP_Upper_Band_{i}'] = final_poc_data['VWAP'] + final_poc_data['Rolling_STD'] * multiplier
            #     final_poc_data[f'VWAP_Lower_Band_{i}'] = final_poc_data['VWAP'] - final_poc_data['Rolling_STD'] * multiplier

            # # Drop unnecessary columns
            # final_poc_data.drop(['Typical_Price', 'Rolling_STD'], axis=1, inplace=True)

            # Extract the date part to group the data by day
            final_poc_data['Only_Date'] = final_poc_data['Date'].dt.date

            # Calculate daily high and low values from the 30-minute data
            daily_high_low = final_poc_data.groupby('Only_Date').agg({'High': 'max', 'Low': 'min'}).reset_index()

            # Add the previous day's high and low to the 30-minute data
            daily_high_low['Previous_High'] = daily_high_low['High'].shift(1)
            daily_high_low['Previous_Low'] = daily_high_low['Low'].shift(1)

            # Merge the previous high and low values into the original 30-minute data
            final_poc_data = final_poc_data.merge(daily_high_low[['Only_Date', 'Previous_High', 'Previous_Low']], 
                                                left_on='Only_Date', right_on='Only_Date', how='left')

            # Calculate Fibonacci retracement levels for each 30-minute interval based on the previous day's high and low
            final_poc_data['Fib_38.2'] = final_poc_data['Previous_High'] - (final_poc_data['Previous_High'] - final_poc_data['Previous_Low']) * 0.382
            final_poc_data['Fib_50'] = (final_poc_data['Previous_High'] + final_poc_data['Previous_Low']) / 2
            final_poc_data['Fib_61.8'] = final_poc_data['Previous_High'] - (final_poc_data['Previous_High'] - final_poc_data['Previous_Low']) * 0.618

            # Drop unnecessary columns
            final_poc_data.drop(['Previous_High', 'Previous_Low'], axis=1, inplace=True)

            # Calculate Average True Range (ATR) using TA-Lib (already done above)
            # final_poc_data['ATR'] is equivalent to 'ATR_14_30_mins'

            # Calculate Stochastic Oscillator using TA-Lib
            final_poc_data['%K'], final_poc_data['%D'] = talib.STOCH(
                final_poc_data['High'], final_poc_data['Low'], final_poc_data['Close'],
                fastk_period=14, slowk_period=3, slowk_matype=0, slowd_period=3, slowd_matype=0
            )

            # Calculate Parabolic SAR using TA-Lib
            final_poc_data['PSAR'] = talib.SAR(final_poc_data['High'], final_poc_data['Low'], acceleration=0.02, maximum=0.2)

            # Define the conditions for Initial Balance Range classification
            conditions = [
                final_poc_data['IB_Range'] < final_poc_data['Prev_Day_ATR_14_1_Day'] / 3,
                (final_poc_data['IB_Range'] >= final_poc_data['Prev_Day_ATR_14_1_Day'] / 3) & 
                (final_poc_data['IB_Range'] <= final_poc_data['Prev_Day_ATR_14_1_Day']),
                final_poc_data['IB_Range'] > final_poc_data['Prev_Day_ATR_14_1_Day']
            ]

            # Define the corresponding values for each condition
            choices = ['Small', 'Medium', 'Large']

            # Create the IB Range column using np.select()
            final_poc_data['IB Range Category'] = np.select(conditions, choices, default='Unknown')

            trends = []

            # Get the unique dates and sort them
            sorted_dates = sorted(set(final_poc_data['Date']))

            # Iterate over the sorted dates by index, starting from the third day to have data for previous two days
            for i in range(2, len(sorted_dates)):
                date = sorted_dates[i]
                previous_date = sorted_dates[i - 1]
                two_days_ago = sorted_dates[i - 2]

                # Extract data for the current date and previous dates
                current_data = final_poc_data[final_poc_data['Date'] == date]
                previous_data = final_poc_data[final_poc_data['Date'] == previous_date]
                two_days_ago_data = final_poc_data[final_poc_data['Date'] == two_days_ago]
                
                previous_data_last_row = previous_data.iloc[-1]
                previous_data_vah = previous_data_last_row['VAH']
                previous_data_val = previous_data_last_row['VAL']
                previous_data_poc = previous_data_last_row['POC']
                
                current_data_first_row = current_data.iloc[0]
                current_data_ib_high = current_data_first_row['IB_High']
                current_data_ib_low = current_data_first_row['IB_Low']
                
                if current_data_ib_high > previous_data_vah:
                    trend = 'Bullish'
                elif current_data_ib_low < previous_data_val:
                    trend = 'Bearish'
                else:
                    trend = 'Neutral'
                
                trend_entry = {
                    'Date': date,
                    'Trend': trend,
                    'Previous Day VAH': round(previous_data_vah, 2),
                    'Previous Day VAL': round(previous_data_val, 2),
                    'Previous Day POC': round(previous_data_poc, 2),
                }
                
                trends.append(trend_entry)
                

            # Convert the trends list to a DataFrame
            trends_df = pd.DataFrame(trends)

            # Merge trend data into final_data
            final_poc_data = pd.merge(final_poc_data, trends_df, on='Date', how='left')

            # Initialize an empty list to store each day's input text and trend
            training_data = []

            # st.write(input_date)
            # st.write(final_poc_data.tail())

            final_poc_data['Date'] = final_poc_data['Date'].dt.date

            # filtered_poc_today_df = final_poc_data[final_poc_data['Date'] == input_date]

            # st.write(filtered_poc_today_df)

            st.write("final_poc_data :")
            st.write(final_poc_data)

            st.write("index_data")
            st.write(index_data)

            # Assuming 'Datetime' is the common column in both DataFrames
            final_poc_data = pd.merge(final_poc_data, index_data, on='Datetime', how='inner')

            st.write("final_poc_data :")
            st.write(final_poc_data)

            

            
            # final_poc_data = final_poc_data[final_poc_data['Datetime'] < pd.to_datetime(input_date)]

            # Get the unique dates and sort them
            sorted_dates = sorted(set(final_poc_data['Date']))
            final_poc_data['2 Day VAH and VAL'] = ''

            # Iterate over the sorted dates by index, starting from the third day to have data for previous two days
            for i in range(2, len(sorted_dates)):
                date = sorted_dates[i]
                previous_date = sorted_dates[i - 1]
                two_days_ago = sorted_dates[i - 2]

                # Extract data for the current date and previous dates
                current_data = final_poc_data[final_poc_data['Date'] == date]
                previous_data = final_poc_data[final_poc_data['Date'] == previous_date]
                two_days_ago_data = final_poc_data[final_poc_data['Date'] == two_days_ago]
                
                # Calculate the maximum high and minimum low for the previous day
                day_high = previous_data['High'].max()
                day_low = previous_data['Low'].min()
                
                # Initialize an empty list for actions based on previous day's close and VAH/VAL comparisons
                actions = []

                if not previous_data.empty:
                    last_row = previous_data.iloc[-1]

                    # st.write("previous_data")
                    # st.write(previous_data)

                    # Determine close position relative to VAH and VAL
                    if last_row['Close'] >= last_row['VAH']:
                        actions.append('Previous Day Close Above VAH')
                        actions.append('Previous Day Close Bullish')
                    elif last_row['Close'] <= last_row['VAL']:
                        actions.append('Previous Day Close Below VAL')
                        actions.append('Previous Day Close Bearish')
                    else:
                        actions.append('Previous Day Close Neutral')

                    # Insider/Outsider neutral positioning based on VAH/VAL ranges
                    if last_row['Previous Day VAH'] >= last_row['VAH'] and last_row['Previous Day VAL'] <= last_row['VAL']:
                        actions.append('Insider Neutral')
                    elif last_row['Previous Day VAH'] <= last_row['VAH'] and last_row['Previous Day VAL'] >= last_row['VAL']:
                        actions.append('Outsider Neutral')

                    # Determine day type based on Initial Balance range and close
                    if last_row['IB Range Category'] == 'Large' and last_row['Close'] <= last_row['IB_High']:
                        final_day_type = 'Large Range Normal Day'
                    elif last_row['IB Range Category'] == 'Medium' and day_high >= last_row['IB_High'] and day_low <= last_row['IB_Low']:
                        final_day_type = 'Medium Range Neutral Day'
                    elif last_row['IB Range Category'] == 'Medium' and last_row['Close'] >= last_row['IB_High']:
                        final_day_type = 'Medium Range +ve Normal Variation Day'
                    elif last_row['IB Range Category'] == 'Medium' and last_row['Close'] <= last_row['IB_Low']:
                        final_day_type = 'Medium Range -ve Normal Variation Day'
                    elif last_row['IB Range Category'] == 'Small' and last_row['Close'] >= last_row['IB_High']:
                        final_day_type = 'Small Range +ve Trend Variation Day'
                    elif last_row['IB Range Category'] == 'Small' and last_row['Close'] <= last_row['IB_Low']:
                        final_day_type = 'Small Range -ve Trend Variation Day'
                    elif last_row['IB Range Category'] == 'Small' and last_row['Close'] <= last_row['IB_High'] and last_row['Close'] >= last_row['IB_Low']:
                        final_day_type = 'Small Range Non Trend Variation Day'
                    else:
                        final_day_type = ''

                # Calculate the opening price and difference from the previous close
                opening_price = current_data.iloc[0]['Open']
                previous_close = previous_data.iloc[-1]['Close']
                open_point_diff = round(opening_price - previous_close, 2) if previous_close else None
                open_percent_diff = round((open_point_diff / previous_close) * 100, 2) if previous_close else None
                open_above_below = "above" if open_point_diff > 0 else "below" if open_point_diff < 0 else "no change"

                current_row = current_data.iloc[0]

                current_data_trend = current_data.head(2)
                # Create the formatted string for MA_20 trend
                ma_20_trend = " -> ".join(round(previous_data['MA_20'],2).astype(str))
                ma_20_trend_current = " -> ".join(round(current_data_trend['MA_20'],2).astype(str))
                rsi_trend = " -> ".join(round(previous_data['RSI'],2).astype(str))
                rsi_trend_current = " -> ".join(round(current_data_trend['RSI'],2).astype(str))
                macd_trend = " -> ".join(round(previous_data['MACD'],2).astype(str))
                macd_trend_current = " -> ".join(round(current_data_trend['MACD'],2).astype(str))
                macd_signal_trend = " -> ".join(round(previous_data['Signal Line'],2).astype(str))
                macd_signal_trend_current = " -> ".join(round(current_data_trend['Signal Line'],2).astype(str))
                bb_upper_trend = " -> ".join(round(previous_data['Bollinger_Upper'],2).astype(str))
                bb_upper_trend_current = " -> ".join(round(current_data_trend['Bollinger_Upper'],2).astype(str))
                bb_lower_trend = " -> ".join(round(previous_data['Bollinger_Lower'],2).astype(str))
                bb_lower_trend_current = " -> ".join(round(current_data_trend['Bollinger_Lower'],2).astype(str))
                vwap_trend = " -> ".join(round(previous_data['VWAP'],2).astype(str))
                vwap_trend_current = " -> ".join(round(current_data_trend['VWAP'],2).astype(str))
                atr_trend = " -> ".join(round(previous_data['ATR'],2).astype(str))
                atr_trend_current = " -> ".join(round(current_data_trend['ATR'],2).astype(str))
                # st.write(ma_20_trend)

                highest_correaltion = 0
                # Determine the highest correlation and update the suffix dynamically
                if correlation_dji >= correlation_nasdaq and correlation_dji >= correlation_s_and_p_500:
                    suffix = '_dji'
                    highest_correaltion = correlation_dji
                elif correlation_nasdaq >= correlation_dji and correlation_nasdaq >= correlation_s_and_p_500:
                    suffix = '_nasdaq'
                    highest_correaltion = correlation_nasdaq
                else:
                    suffix = '_s_and_p_500'
                    highest_correaltion = correlation_s_and_p_500

                # Update the dynamic strings for trends
                index_ma_20_trend = " -> ".join(round(previous_data[f'MA_Score{suffix}'], 2).astype(str))
                index_ma_20_trend_current = " -> ".join(round(current_data_trend[f'MA_Score{suffix}'], 2).astype(str))
                index_rsi_trend = " -> ".join(round(previous_data[f'RSI_Score{suffix}'], 2).astype(str))
                index_rsi_trend_current = " -> ".join(round(current_data_trend[f'RSI_Score{suffix}'], 2).astype(str))
                index_perc_k_trend = " -> ".join(round(previous_data[f'%K_Score{suffix}'], 2).astype(str))
                index_perc_k_trend_current = " -> ".join(round(current_data_trend[f'%K_Score{suffix}'], 2).astype(str))
                index_stoch_rsi_trend = " -> ".join(round(previous_data[f'Stoch_RSI_Score{suffix}'], 2).astype(str))
                index_stoch_rsi_trend_current = " -> ".join(round(current_data_trend[f'Stoch_RSI_Score{suffix}'], 2).astype(str))
                index_cci_trend = " -> ".join(round(previous_data[f'CCI_Score{suffix}'], 2).astype(str))
                index_cci_trend_current = " -> ".join(round(current_data_trend[f'CCI_Score{suffix}'], 2).astype(str))
                index_bbp_trend = " -> ".join(round(previous_data[f'BBP_Score{suffix}'], 2).astype(str))
                index_bbp_trend_current = " -> ".join(round(current_data_trend[f'BBP_Score{suffix}'], 2).astype(str))
                index_vwap_trend = " -> ".join(round(previous_data[f'VWAP_Score{suffix}'], 2).astype(str))
                index_vwap_trend_current = " -> ".join(round(current_data_trend[f'VWAP_Score{suffix}'], 2).astype(str))
                index_bb_trend = " -> ".join(round(previous_data[f'BB_Score{suffix}'], 2).astype(str))
                index_bb_trend_current = " -> ".join(round(current_data_trend[f'BB_Score{suffix}'], 2).astype(str))
                index_st_trend = " -> ".join(round(previous_data[f'Supertrend_Score{suffix}'], 2).astype(str))
                index_st_trend_current = " -> ".join(round(current_data_trend[f'Supertrend_Score{suffix}'], 2).astype(str))
                index_reg_trend = " -> ".join(round(previous_data[f'Linear_Regression_Score{suffix}'], 2).astype(str))
                index_reg_trend_current = " -> ".join(round(current_data_trend[f'Linear_Regression_Score{suffix}'], 2).astype(str))
                index_sentiment_trend = " -> ".join(round(previous_data[f'Sentiment{suffix}'], 2).astype(str))
                index_sentiment_trend_current = " -> ".join(round(current_data_trend[f'Sentiment{suffix}'], 2).astype(str))

                # dji_ma_20_trend = " -> ".join(round(previous_data['MA_Score_dji'],2).astype(str))
                # dji_ma_20_trend_current = " -> ".join(round(current_data_trend['MA_Score_dji'],2).astype(str))
                # dji_rsi_trend = " -> ".join(round(previous_data['RSI_Score_dji'],2).astype(str))
                # dji_rsi_trend_current = " -> ".join(round(current_data_trend['RSI_Score_dji'],2).astype(str))
                # dji_perc_k_trend = " -> ".join(round(previous_data['%K_Score_dji'],2).astype(str))
                # dji_perc_k_trend_current = " -> ".join(round(current_data_trend['%K_Score_dji'],2).astype(str))
                # dji_stoch_rsi_trend = " -> ".join(round(previous_data['Stoch_RSI_Score_dji'],2).astype(str))
                # dji_stoch_rsi_trend_current = " -> ".join(round(current_data_trend['Stoch_RSI_Score_dji'],2).astype(str))
                # dji_cci_trend = " -> ".join(round(previous_data['CCI_Score_dji'],2).astype(str))
                # dji_cci_trend_current = " -> ".join(round(current_data_trend['CCI_Score_dji'],2).astype(str))
                # dji_bbp_trend = " -> ".join(round(previous_data['BBP_Score_dji'],2).astype(str))
                # dji_bbp_trend_current = " -> ".join(round(current_data_trend['BBP_Score_dji'],2).astype(str))
                # dji_vwap_trend = " -> ".join(round(previous_data['VWAP_Score_dji'],2).astype(str))
                # dji_vwap_trend_current = " -> ".join(round(current_data_trend['VWAP_Score_dji'],2).astype(str))
                # dji_bb_trend = " -> ".join(round(previous_data['BB_Score_dji'],2).astype(str))
                # dji_bb_trend_current = " -> ".join(round(current_data_trend['BB_Score_dji'],2).astype(str))
                # dji_st_trend = " -> ".join(round(previous_data['Supertrend_Score_dji'],2).astype(str))
                # dji_st_trend_current = " -> ".join(round(current_data_trend['Supertrend_Score_dji'],2).astype(str))
                # dji_reg_trend = " -> ".join(round(previous_data['Linear_Regression_Score_dji'],2).astype(str))
                # dji_reg_trend_current = " -> ".join(round(current_data_trend['Linear_Regression_Score_dji'],2).astype(str))
                # dji_sentiment_trend = " -> ".join(round(previous_data['Sentiment_dji'],2).astype(str))
                # dji_sentiment_trend_current = " -> ".join(round(current_data_trend['Sentiment_dji'],2).astype(str))
                
                # Generate the LLM input text with added indicators
                input_text = (
                    f"Today’s profile on {date} for {ticker} with IB Range Type {current_row['IB Range Category']} Range. The market opened at {opening_price}, "
                    f"Opening_Gap_Percentage is {open_percent_diff}% ( Opening_Gap_Points {abs(open_point_diff)} points) {open_above_below} the previous day's close. "
                    f"The Initial Balance High is {current_row['IB_High']} and Initial Balance Low is {current_row['IB_Low']}, "
                    f"giving an Initial Balance Range of {round(current_row['IB_Range'],2)}. "
                    f"Yesterday Single FootPrint Ranges was {last_row['Single_FootPrint']} "
                    f"Yesterday Poor High : {last_row['Poor_High']} "
                    f"Yesterday Poor Low : {last_row['Poor_Low']} "
                    f"Yesterday's Close was {last_row['Close']} "
                    f"Yesterday's VAH was {last_row['VAH']}, Yesterday's VAL was {last_row['VAL']} and Yesterday's POC was {last_row['POC']}"
                    f"Day before yesterday's VAH was {last_row['Previous Day VAH']} , Day before yesterday's VAL was {last_row['Previous Day VAL']} and Day before yesterday's POC was {last_row['Previous Day POC']}. "
                    f"Previous Day High was {day_high} and Previous Day Low was {day_low} \n"
                    f"Previous day Type: {last_row['Day_Type']}.\n"
                    f"Previous Adjusted Day Type: {final_day_type}.\n"
                    f"Previous Close Type: {last_row['Close_Type']}.\n"
                    f"Previous 2 Day VAH and VAL: {actions}.\n"

                    # Adding indicators
                    f"MovingAverage_20_Day trend for Previous Day (09:30 - 16:00) to Current Day (09:30 - 10:30): {ma_20_trend} -> {ma_20_trend_current}\n"
                    # f"MovingAverage_20_Day trend for Current Day (09:30 - 10:30): {ma_20_trend_current}\n"
                    f"MA_20_Day is {round(current_data_trend.iloc[1]['MA_20'],2)}.\n"
                    f"RSI_14 trend for Previous Day (09:30 - 16:00) to Current Day (09:30 - 10:30): {rsi_trend} -> {rsi_trend_current} \n"
                    # f"RSI_14 trend for Current Day (09:30 - 10:30): {rsi_trend_current} \n"
                    f"RSI is {round(current_data_trend.iloc[1]['RSI'],2)}. \n"
                    f"MACD_LINE trend for Previous Day (09:30 - 16:00) to Current Day (09:30 - 10:30): {macd_trend} -> {macd_trend_current} \n"
                    # f"MACD_LINE trend for Current Day (09:30 - 10:30): {macd_trend_current} \n"
                    f"MACD_Signal trend for Previous Day (09:30 - 16:00) to Current Day (09:30 - 10:30): {macd_signal_trend} -> {macd_signal_trend_current} \n"
                    # f"MACD_Signal trend for Current Day (09:30 - 10:30): {macd_signal_trend_current} \n"
                    f"MACD is {round(current_data_trend.iloc[1]['MACD'],2)} with Signal Line at {round(current_data_trend.iloc[1]['Signal Line'],2)}. \n"
                    f"BB_Upper trend for Previous Day (09:30 - 16:00) to Current Day (09:30 - 10:30): {bb_upper_trend} -> {bb_upper_trend_current} \n"
                    # f"BB_Upper trend for Current Day (09:30 - 10:30): {bb_upper_trend_current} \n"
                    f"BB_Lower trend for Previous Day (09:30 - 16:00) to Current Day (09:30 - 10:30): {bb_lower_trend} -> {bb_lower_trend_current} \n"
                    # f"BB_Lower trend for Current Day (09:30 - 10:30): {bb_lower_trend_current} \n"
                    f"Bollinger Bands Upper at {round(current_data_trend.iloc[1]['Bollinger_Upper'],2)} and Bollinger Bands Lower at {round(current_data_trend.iloc[1]['Bollinger_Lower'],2)}. "
                    f"VWAP_14 trend for Previous Day (09:30 - 16:00) to Current Day (09:30 - 10:30): {vwap_trend} -> {vwap_trend_current} \n"
                    # f"VWAP_14 trend for Current Day (09:30 - 10:30): {vwap_trend_current} \n"
                    f"VWAP is {round(current_data_trend.iloc[1]['VWAP'],2)}. \n"
                    f"Fibonacci Levels: 38.2% at {round(last_row['Fib_38.2'],2)}, 50% at {round(last_row['Fib_50'],2)}, 61.8% at {round(last_row['Fib_61.8'],2)}. "
                    f"ATR_14 trend for Previous Day (09:30 - 16:00) to Current Day (09:30 - 10:30): {vwap_trend} -> {vwap_trend_current} \n"
                    # f"ATR_14 trend for Current Day (09:30 - 10:30): {vwap_trend_current} \n"
                    f"ATR is {round(current_data_trend.iloc[1]['ATR'],2)}. "
                    f"Stochastic Oscillator %K is {round(last_row['%K'],2)} and %D is {round(last_row['%D'],2)}. "
                    f"Parabolic SAR is at {round(last_row['PSAR'],2)}. "

                    f"Below is the Scores trend with Major Index and {ticker} with correlation of  {highest_correaltion:.2f}"
                    # st.write(f"Correlation between {ticker} and NASDAQ movements: {correlation_nasdaq:.2f}")
                    # st.write(f"Correlation between {ticker} and S&P 500 movements: {correlation_s_and_p_500:.2f}")
                    # Adding indicators
                    f"Index MA20_Score trend for Previous Day (09:30 - 16:00) to Current Day (09:30 - 10:30): {index_ma_20_trend} -> {index_ma_20_trend_current}\n"
                    f"Index RSI_Score trend for Previous Day (09:30 - 16:00) to Current Day (09:30 - 10:30): {index_rsi_trend} -> {index_rsi_trend_current}\n"
                    f"Index %K_Score trend for Previous Day (09:30 - 16:00) to Current Day (09:30 - 10:30): {index_perc_k_trend} -> {index_perc_k_trend_current}\n"
                    f"Index Stoch_RSI trend for Previous Day (09:30 - 16:00) to Current Day (09:30 - 10:30): {index_stoch_rsi_trend} -> {index_stoch_rsi_trend_current}\n"
                    f"Index CCI_Score trend for Previous Day (09:30 - 16:00) to Current Day (09:30 - 10:30): {index_cci_trend} -> {index_cci_trend_current}\n"
                    f"Index BBP_Score trend for Previous Day (09:30 - 16:00) to Current Day (09:30 - 10:30): {index_bbp_trend} -> {index_bbp_trend_current}\n"
                    f"Index VWAP_Score trend for Previous Day (09:30 - 16:00) to Current Day (09:30 - 10:30): {index_vwap_trend} -> {index_vwap_trend_current}\n"
                    f"Index BB_Score trend for Previous Day (09:30 - 16:00) to Current Day (09:30 - 10:30): {index_bb_trend} -> {index_bb_trend_current}\n"
                    f"Index Supertrend_Score trend for Previous Day (09:30 - 16:00) to Current Day (09:30 - 10:30): {index_st_trend} -> {index_st_trend_current}\n"
                    f"Index Linear_Regression_Score trend for Previous Day (09:30 - 16:00) to Current Day (09:30 - 10:30): {index_reg_trend} -> {index_reg_trend_current}\n"
                    f"Index Sentiment_Score trend for Previous Day (09:30 - 16:00) to Current Day (09:30 - 10:30): {index_sentiment_trend} -> {index_sentiment_trend_current}\n"

                    f"Given these indicators, what is the expected market direction for today?"
                )

                # print(input_text)
                
                current_day_close = current_data.iloc[-1]['Close']
                previous_day_high = previous_data['High'].max()
                previous_day_low = previous_data['Low'].min()

                result = f'The stock closed at {round(current_day_close,2)} \n'
                if current_day_close >= previous_close:
                    result += 'The stock closed above yesterday close \n'
                else:
                    result += 'The stock closed below yesterday close \n'
                if current_day_close >= previous_day_high:
                    result += 'The stock closed above Previous Day High \n'
                else:
                    result += 'The stock closed below Previous Day High \n'

                if current_day_close >= previous_day_low:
                    result += 'The stock closed above Previous Day Low \n'
                else:
                    result += 'The stock closed below Previous Day Low \n'

                if current_day_close >= current_row['IB_High']:
                    result += 'The stock closed above Initial Balance High \n'
                else:
                    result += 'The stock closed below Initial Balance High \n'

                if current_day_close >= current_row['IB_Low']:
                    result += 'The stock closed above Initial Balance Low \n'
                else:
                    result += 'The stock closed below Initial Balance Low \n'

                # Get the trend (output) for the current date
                trend = current_data.iloc[-1]['Trend'] if 'Trend' in current_data.columns else None

                # Append the input-output pair to the training data list
                training_data.append({
                    'Date': date,
                    'Input Text': input_text,
                    'Trend': trend,
                    'Result': result
                })

            # Convert the training data list to a DataFrame
            training_data_df = pd.DataFrame(training_data)

            # st.write(training_data_df)

            current_training_df = training_data_df[training_data_df['Date'] == input_date]
            st.write(current_training_df)

            if not current_training_df.empty:
                input_text = current_training_df.iloc[0]['Input Text']
                
                # st.write(current_row)
                # input_text = (
                #     f"Today’s profile on {input_date} for {ticker} with IB Range Type {current_row['IB_Range']} Range. The market opened at {opening_price}, "
                #     f"Opening_Gap_Percentage is {open_percent_diff}% ( Opening_Gap_Points {abs(open_point_diff)} points) {open_above_below} the previous day's close. "
                #     f"The Initial Balance High is {current_row['Initial Balance High']} and Initial Balance Low is {current_row['Initial Balance Low']}, "
                #     f"giving an Initial Balance Range of {current_row['Initial Balance Range']}. "
                #     f"Yesterday's VAH was {last_row['VAH']} and Yesterday's VAL was {last_row['VAL']}. "
                #     f"Day before yesterday's VAH was {last_row['Previous Day VAH']} and Day before yesterday's VAL was {last_row['Previous Day VAL']}. "
                #     f"Previous day Type: {last_row['Day Type']}.\n"
                #     f"Previous Adjusted Day Type: {final_day_type}.\n"
                #     f"Previous Close Type: {last_row['Close Type']}.\n"
                #     f"Previous 2 Day VAH and VAL: {actions}.\n"

                #     # Adding indicators
                #     f"MA_20_Day is {last_row['MA_20']}. "
                #     f"RSI is {last_row['RSI']}. "
                #     f"MACD is {last_row['MACD']} with Signal Line at {last_row['Signal Line']}. "
                #     f"Bollinger Bands Upper at {last_row['Bollinger_Upper']} and Bollinger Bands Lower at {last_row['Bollinger_Lower']}. "
                #     f"VWAP is {last_row['VWAP']}. "
                #     f"Fibonacci Levels: 38.2% at {last_row['Fib_38.2']}, 50% at {last_row['Fib_50']}, 61.8% at {last_row['Fib_61.8']}. "
                #     f"ATR is {last_row['ATR']}. "
                #     f"Stochastic Oscillator %K is {last_row['%K']} and %D is {last_row['%D']}. "
                #     f"Parabolic SAR is at {last_row['PSAR']}. "

                #     f"Given these indicators, what is the expected market direction for today?"
                # )

                st.write(input_text)

            st.write(training_data_df)
            st.write(input_date)
            st.write(pd.to_datetime(input_date))

            # training_data_df = training_data_df[training_data_df['Date'] < pd.to_datetime(input_date)]
            training_data_df = training_data_df[training_data_df['Date'] < input_date]


            # Display the final training DataFrame
            st.write(training_data_df.head())

            # Initialize model for semantic similarity
            model = SentenceTransformer('all-MiniLM-L6-v2')

            # Define segments with both text descriptions and numeric values for dynamic similarity computation
    #         segments = extracted_data

            segments = {
                key: {"numeric_value": value} if isinstance(value, (int, float)) else {"category": value}
                for key, value in extracted_data.items()
            }

            st.write(segments)

    #         query_segment_embeddings = {
    #             key: model.encode(str(value.get("text", value.get("numeric_value"))), convert_to_tensor=True)
    #             for key, value in segments.items()
    #             if isinstance(value, dict) and ("text" in value or "numeric_value" in value)
    #         }
    #         query_segment_embeddings = {key: model.encode(value["text"], convert_to_tensor=True) for key, value in segments.items() if 'text' in value}

    #         st.write(query_segment_embeddings.keys())

            # Generate embeddings for entries with a `category` (text) or `numeric_value` (numeric) key
            query_segment_embeddings = {
                key: model.encode(str(value.get("category", value.get("numeric_value"))), convert_to_tensor=True)
                for key, value in segments.items()
                if isinstance(value, dict) and ("category" in value or "numeric_value" in value)
            }

            st.write(query_segment_embeddings.keys())


            # Custom function to calculate Euclidean similarity for numeric comparisons
            def calculate_normalized_similarity(query_value, comparison_value):
                if query_value is None or comparison_value is None:
                    return 1  # Max similarity if no numeric comparison is possible
                euclidean_dist = np.sqrt((query_value - comparison_value) ** 2)
                return np.exp(-euclidean_dist / 10)  # Convert distance to similarity

            # Improved extraction function for numeric values
            def extract_numeric_value(text, segment):


                # Define patterns dynamically for known segments
                patterns = {
                    "IB_High": r"Initial Balance High is ([\d.,]+)",
                    "IB_Low": r"Initial Balance Low is ([\d.,]+)",
                    "Range_Type": r"IB Range Type ([\d.,]+)",
                    "VAH_Yesterday": r"Yesterday's VAH ([\d.,]+)",
                    "VAL_Yesterday": r"Yesterday's VAL ([\d.,]+)",  # Ensure it's VAL by matching after VAH
                    "VAH_DayBefore": r"Day before yesterday's VAH ([\d.,]+)",
                    "VAL_DayBefore": r"Day before yesterday's VAL ([\d.,]+)",  # Ensure it's VAL by matching after Day before VAH

                    "Previous_Day_Type": r"Previous day Type: ([\d.,]+)",
                    "Previous_Adjusted_Day_Type": r"Previous Adjusted Day Type: ([\d.,]+)",
                    "Previous_Close_Type": r"Previous Close Type: ([\d.,]+)",
                    "Previous_2_D_VAH_VAL": r"Previous 2 Day VAH and VAL: ([\d.,]+)",

                    "IB_Range": r"Initial Balance Range ([\d.,]+)",


                    "Moving_Avg_20": r"MA_20_Day ([\d.,]+)",
                    "RSI": r"RSI ([\d.,]+)",
                    "MACD": r"MACD ([\d.,-]+)",  # Allows negative values
                    "Signal_Line": r"Signal Line ([\d.,]+)",
                    "Bollinger_Upper": r"Bollinger Bands Upper ([\d.,]+)",
                    "Bollinger_Lower": r"Bollinger Bands Lower ([\d.,]+)",
                    "VWAP": r"VWAP ([\d.,]+)",
                    "Fibonacci_38.2%": r"38\.2% at ([\d.,]+)",
                    "Fibonacci_50%": r"50% at ([\d.,]+)",
                    "Fibonacci_61.8%": r"61\.8% at ([\d.,]+)",
                    "ATR": r"ATR ([\d.,]+)",
                    "Stochastic_Oscillator_%K": r"Stochastic Oscillator %K is ([\d.,]+)",
                    "Stochastic_Oscillator_%D": r"%D is ([\d.,]+)",
                    "Parabolic_SAR": r"Parabolic SAR is at ([\d.,]+)",
                    "Market_Open": r"market opened at ([\d.,]+)",
                    "Opening_Gap_Percentage": r"Opening_Gap_Percentage ([\d.,]+)%",  # Matches percentage
                    "Opening_Gap_Points": r"Opening_Gap_Points \(([\d.,]+) points\)"
                }

                pattern = patterns.get(segment)
                if pattern:
                    match = re.search(pattern, text)
                    if match:
                        # Clean the extracted value: remove commas and ensure no extra period at the end
                        cleaned_value = match.group(1).replace(',', '').rstrip('.')
    #                     st.write(cleaned_value)
                        try:
                            return float(cleaned_value)
                        except ValueError:
                            print(f"Warning: Could not convert '{cleaned_value}' to float.")
                            return None
                return None

            # Set weights for different categories
            weights = {
                'technical': 0.4,    # weight for technical indicators (e.g., Moving_Avg_20, RSI, MACD, etc.)
                'ib_ranges': 0.2,    # weight for Initial Balance (IB) ranges (e.g., IB_High, IB_Low, IB_Range)
                'gaps': 0.2,         # weight for gap indicators (e.g., Opening_Gap_Percentage, Opening_Gap_Points)
                'range_type': 0.2    # weight for categorical range types (e.g., Range_Type)
            }

    #         # Calculate similarities for each segment
    #         similarity_results = []
    #         for index, row in training_data_df.iterrows():
    #         #     row_similarity = {'Date': row['Date'], 'Trend': row['Trend']}
    #             row_similarity = {'Date': row['Date'],'Input Text':row['Input Text'],'Trend':row['Trend'], 'Result': row['Result']}

    #             # Calculate embedding if 'embedding' column does not exist
    #             if 'embedding' not in row:
    #                 row['embedding'] = model.encode(row['Input Text'], convert_to_tensor=True)

    #             # Accumulators for weighted score
    #             technical_score = 0
    #             ib_ranges_score = 0
    #             gaps_score = 0
    #             range_type_score = 0

    #             for segment, details in segments.items():
    # #                 st.write(segment)
    # #                 st.write(details)
    #                 if 'category' in details:
    #                     # Categorical similarity
    #                     segment_embedding = query_segment_embeddings[segment]
    #                     semantic_similarity = util.pytorch_cos_sim(segment_embedding, row['embedding']).item()
    #                     row_similarity[f"{segment}_semantic_similarity"] = semantic_similarity

    #                     # Add to range type score if it's Range_Type
    #                     if segment == "Range_Type":
    #                         range_type_score += semantic_similarity

    #                 elif 'numeric_value' in details:
    #                     # Numeric similarity
    #                     query_value = details['numeric_value']
    #                     extracted_value = extract_numeric_value(row['Input Text'], segment)
    #                     euclidean_similarity = calculate_normalized_similarity(query_value, extracted_value)
    #                     row_similarity[f"{segment}_euclidean_similarity"] = euclidean_similarity

    #                     # Categorize and add scores to respective categories
    #                     if segment in ["Moving_Avg_20", "RSI", "MACD", "Signal_Line", "Bollinger_Upper", "Bollinger_Lower", "VWAP", "ATR"]:
    #                         technical_score += euclidean_similarity
    #                     elif segment in ["IB_High", "IB_Low", "IB_Range"]:
    #                         ib_ranges_score += euclidean_similarity
    #                     elif segment in ["Opening_Gap_Percentage", "Opening_Gap_Points"]:
    #                         gaps_score += euclidean_similarity
    #                 else:
    #                     # Additional semantic similarity for other segments
    #                     segment_embedding = query_segment_embeddings[segment]
    #                     semantic_similarity = util.pytorch_cos_sim(segment_embedding, row['embedding']).item()
    #                     row_similarity[f"{segment}_semantic_similarity"] = semantic_similarity


    #             # Calculate weighted similarity score
    #             total_similarity_score = (
    #                 technical_score * weights['technical'] +
    #                 ib_ranges_score * weights['ib_ranges'] +
    #                 gaps_score * weights['gaps'] +
    #                 range_type_score * weights['range_type']
    #             )
    #             row_similarity['total_similarity_score'] = total_similarity_score

    #             # Append row results to similarity results
    #             similarity_results.append(row_similarity)

    #         # Convert results to DataFrame for inspection
    #         similarity_df = pd.DataFrame(similarity_results)

            similarity_results = []

            for index, row in training_data_df.iterrows():
                row_similarity = {
                    'Date': row['Date'],
                    'Input Text': row['Input Text'],
                    'Trend': row['Trend'],
                    'Result': row['Result']
                }

                # Calculate embedding if 'embedding' column does not exist
                if 'embedding' not in row or pd.isna(row['embedding']):
                    embedding_tensor = model.encode(row['Input Text'], convert_to_tensor=True)
                    row['embedding'] = embedding_tensor.cpu().numpy()  # Move tensor to CPU and convert to NumPy

                # Convert stored embedding back to tensor for computation
                row_embedding_tensor = torch.tensor(row['embedding']).to('cuda' if torch.cuda.is_available() else 'cpu')

                # Accumulators for weighted score
                technical_score = 0
                ib_ranges_score = 0
                gaps_score = 0
                range_type_score = 0

                for segment, details in segments.items():
                    if 'category' in details:
                        # Categorical similarity
                        segment_embedding = query_segment_embeddings[segment]
                        segment_embedding = segment_embedding.to(row_embedding_tensor.device)  # Ensure same device
                        semantic_similarity = util.pytorch_cos_sim(segment_embedding, row_embedding_tensor).item()
                        row_similarity[f"{segment}_semantic_similarity"] = semantic_similarity

                        # Add to range type score if it's Range_Type
                        if segment == "Range_Type":
                            range_type_score += semantic_similarity

                    elif 'numeric_value' in details:
                        # Numeric similarity
                        query_value = details['numeric_value']
                        extracted_value = extract_numeric_value(row['Input Text'], segment)
                        euclidean_similarity = calculate_normalized_similarity(query_value, extracted_value)
                        row_similarity[f"{segment}_euclidean_similarity"] = euclidean_similarity

                        # Categorize and add scores to respective categories
                        if segment in ["Moving_Avg_20", "RSI", "MACD", "Signal_Line", "Bollinger_Upper", "Bollinger_Lower", "VWAP", "ATR"]:
                            technical_score += euclidean_similarity
                        elif segment in ["IB_High", "IB_Low", "IB_Range"]:
                            ib_ranges_score += euclidean_similarity
                        elif segment in ["Opening_Gap_Percentage", "Opening_Gap_Points"]:
                            gaps_score += euclidean_similarity
                    else:
                        # Additional semantic similarity for other segments
                        segment_embedding = query_segment_embeddings[segment]
                        segment_embedding = segment_embedding.to(row_embedding_tensor.device)  # Ensure same device
                        semantic_similarity = util.pytorch_cos_sim(segment_embedding, row_embedding_tensor).item()
                        row_similarity[f"{segment}_semantic_similarity"] = semantic_similarity

                # Calculate weighted similarity score
                total_similarity_score = (
                    technical_score * weights['technical'] +
                    ib_ranges_score * weights['ib_ranges'] +
                    gaps_score * weights['gaps'] +
                    range_type_score * weights['range_type']
                )
                row_similarity['total_similarity_score'] = total_similarity_score

                # Append row results to similarity results
                similarity_results.append(row_similarity)

            # Convert results to DataFrame for inspection
            similarity_df = pd.DataFrame(similarity_results)

            # Sort by total similarity score and select the top 15 rows
            top_15_similar = similarity_df.sort_values(by='total_similarity_score', ascending=False).head(15)
            st.write(top_15_similar)


            # Prepare reference information from the top 15 similar entries
            reference_info = ""
            for _, sim in top_15_similar.iterrows():
                # filtered_data = training_data_df[training_data_df['Date'] == pd.Timestamp(sim['Date'])]
                filtered_data = training_data_df[training_data_df['Date'] == sim['Date']]

                if not filtered_data.empty:
                    entry_text = filtered_data.iloc[0]['Input Text']
            #         trend = filtered_data.iloc[0]['Trend']
                    result = filtered_data.iloc[0]['Result']
            #         reference_info += f"Date: {sim['Date']}\nInput Text: {entry_text}\nTrend: {trend}\n\n"
                    reference_info += f"Date: {sim['Date']}\nInput Text: {entry_text}\Result: {result}\n\n"


            # # Build the full prompt for LLM
            # prompt = f"""
            # You are an advanced financial data analyst AI model.  Your task is to predict stock movement trends for today by analyzing the provided historical top 15 most similar day patterns. Use the given context and criteria for accurate prediction.
            # Task:

            # 1. Analyze the provided top 15 most similar day patterns from the past and their movements.
            # 2. Predict the following outcomes for today:
            #     Will the stock close above or below yesterday's close?
            #     Will the stock close above or below the previous day's high?
            #     Will the stock close above or below the previous day's low?
            #     Will the stock close above or below the initial balance high?
            #     Will the stock close above or below the initial balance low?

            # Input Format:
            # 1. Top 15 Similar Patterns: A dataset containing:
            #     Dates of the similar days.
            #     Key metrics for each day: Open, High, Low, Close, Initial Balance Metrics, Value Areas, Day Type Information, Technical Indicators, Market Opening Information and other relevant information
            #     Performance following these patterns (e.g., next-day close relative to various benchmarks).
            # 2. Today's Data(from Input Text): Includes Open, High, Low, Close, Initial Balance Metrics, Value Areas, Day Type Information, Technical Indicators, Market Opening Information etc

            # Output Format:
            # Provide a structured prediction in the following table format:

            # Criteria	Prediction (Above/Below)	Confidence (%)	Supporting Similar Patterns (List Days)
            # Yesterday's Close			
            # Previous Day's High			
            # Previous Day's Low			
            # Initial Balance High			
            # Initial Balance Low			

            # Example Prediction:
            # Using the historical patterns provided, justify your prediction for each criterion. For example:

            # Yesterday's Close: Prediction = "Above" with 85% confidence. Supporting patterns: Jan 5, Feb 10, and Mar 20, where similar upward trends were observed following analogous market conditions.

            # Additional Analysis:
            # Highlight common trends in the top 15 patterns.
            # Indicate any deviations in today's metrics from those observed in historical patterns.
            # Suggest external factors (e.g., news, earnings reports) that could influence predictions differently from historical behavior. Here you need to fetch the recent 2 days news information for {ticker}

            # Constraints:
            # Ensure that predictions are data-driven and align with insights from the top 15 patterns.
            # Include confidence scores and supporting evidence for transparency.

            # Top 15 similar trends in the past:
            # {reference_info}

            # Input Text:
            # {input_text}
            

            # Based on the top 15 similar trends and the input text, predict the market result for today with confidence and supporting. 
            # Provide the result prediction with below :
            # The stock will close above/below yesterday close 
            # The stock will close above/below Previous Day High 
            # The stock will close above/below Previous Day Low 
            # The stock will close above/below Initial Balance High 
            # The stock will close above/below Initial Balance Low 
            # """
            

            prompt = f"""
            You are an advanced financial data analyst AI model. Your task is to predict stock movement trends for today by analyzing the provided historical top 15 most similar day patterns. Use the context and criteria provided below for an accurate prediction.

            ### Task:
            1. Analyze the provided **top 15 most similar day patterns** and their movements. Prioritize **Initial Balance Metrics (IB High, IB Low)** and **Value Area Metrics (VAH, VAL)** in your analysis. These metrics should weigh more heavily in your predictions compared to technical indicators.
            2. Predict the following outcomes for today:
            - Will the stock close above or below yesterday's close?
            - Will the stock close above or below the previous day's high?
            - Will the stock close above or below the previous day's low?
            - Will the stock close above or below the initial balance high?
            - Will the stock close above or below the initial balance low?

            ### Prediction Guidelines:
            - **Priority Hierarchy**:
            - **First Priority**: Initial Balance Metrics (IB High, IB Low).
            - **Second Priority**: Value Areas (VAH, VAL).
            - **Third Priority**: Technical Indicators (RSI, MACD, Bollinger Bands, etc.).
            - Use IB and VAH as primary drivers for predictions. Technical indicators should provide supporting evidence but not override the influence of IB and VAH.

            ### Trend Analysis Instructions:

            - **Use MA_20 to identify the broader trend.
            - **Use RSI to evaluate momentum strength or reversals, with overbought (>70) or oversold (<30) conditions signaling potential changes.
            - **Use MACD crossovers to confirm trend direction, particularly in conjunction with IB levels.
            - **Use Bollinger Bands to assess volatility-driven breakouts in alignment with IB metrics.
            - **Use VWAP for intraday trend confirmation; bullish if above, bearish if below.
            - **Use ATR to gauge volatility shifts that support trend continuation or reversal.

            ### Logical Validation:
            - Ensure predictions align logically:
            1. If the stock is predicted to close **above Yesterday's Close**, it must not be below the **Previous Day's Low**.
            2. If the stock is predicted to close **below the Previous Day's High**, it must also be below the **Initial Balance High**.
            3. If the stock is predicted to close **above the Initial Balance Low**, it must also be above the **Previous Day's Low**.
            - Adjust predictions to resolve logical inconsistencies and explain the reasoning behind adjustments in the output.

            ### Input Format:
            1. **Top 15 Similar Patterns**:
            - Dataset containing:
                - Dates of the similar days.
                - Key metrics for each day: Open, High, Low, Close, Initial Balance Metrics, Value Areas, Day Type Information, Technical Indicators, Market Opening Information, and other relevant details.
                - Performance following these patterns (e.g., next-day close relative to various benchmarks).
            2. **Today's Data (from Input Text)**:
            - Includes the following metrics: Open, High, Low, Close, Initial Balance Metrics, Value Areas, Day Type Information, Technical Indicators, Market Opening Information, etc.

            ### Output Format:
            What is the initital Balance High and Initital Balance Low for {input_date}
            Provide a structured prediction in the following table format, including respective values where applicable. Make sure you provide the correct Values after that are provided in the Input Data for Date:{input_date}:

            | **Criteria**            | **Prediction (Above/Below)** | **Confidence (%)** | **Value** | **Supporting Similar Patterns (List Dates)**    |
            |--------------------------|-----------------------------|--------------------|-------------------------------------------------------------|
            | Yesterday's Close        |                             |                    |            |                                                |
            | Previous Day's High      |                             |                    |            |                                                |
            | Previous Day's Low       |                             |                    |            |                                                |
            | Initial Balance High     |                             |                    |            |                                                |
            | Initial Balance Low      |                             |                    |            |                                                |

            ### Example Prediction:
            Using the historical patterns provided, justify your prediction for each criterion. For example:
            - **Yesterday's Close**: Prediction = "Above" with 85% confidence, value = $305. Supporting patterns: Jan 5, Feb 10, and Mar 20, where similar upward trends were observed following analogous market conditions.

            ### Additional Analysis:
            1. Highlight common trends across the top 15 patterns.
            2. Indicate any deviations in today's metrics compared to historical patterns.
            3. Suggest external factors (e.g., recent news, earnings reports) that could influence predictions differently from historical behavior. Fetch the **recent 2 days of news** for the stock ticker `{ticker}` to include in your analysis.

            ### Constraints:
            - Ensure that predictions are data-driven and align with insights from the top 15 patterns.
            - Provide confidence scores and supporting evidence for each prediction to maintain transparency.
            - Validate predictions against logical constraints outlined above and provide corrections where needed.

            ### Input Data:
            **Top 15 Similar Trends in the Past:**
            {reference_info}

            **Today's Profile:**
            {input_text}
            """

            prompt = f"""
            What is the yesterday close , Previous Day High , Previous Day Low, Initital Balance High, Initital Balance Low for {input_date}

            Provide a structured prediction in the following table format, including respective values where applicable. Make sure you provide the correct Values after that are provided in the Input Data for Date:{input_date}:

            | **Criteria**             | **Prediction (Above/Below)** | **Confidence (%)** | **Value**  | **Supporting Similar Patterns (List Dates)**   |
            |--------------------------|------------------------------|--------------------|------------|------------------------------------------------|
            | Yesterday's Close        |                              |                    |            |                                                |
            | Previous Day's High      |                              |                    |            |                                                |
            | Previous Day's Low       |                              |                    |            |                                                |
            | Initial Balance High     |                              |                    |            |                                                |
            | Initial Balance Low      |                              |                    |            |                                                |

            Also based on the Predictions above write the range of the closing price : ** Range of Closing Price **
            
            ### Prediction Guidelines
            - Ensure consistency across criteria (Above/Below) to avoid logical contradictions in directional predictions.
            - The predicted closing range must align with directional movements and reflect realistic market behavior.
            - Confidence levels should be validated and based on strong evidence to maintain reliability and trust.
            **Today's Profile:**
            {input_text}

            ### Input Data:
            **Top 15 Similar Trends in the Past:**
            {reference_info}

            """

            st.write(prompt)

    #         # Get trend prediction
    #         if st.button("Get Prediction"):

            # Set up OpenAI API Key
            openai.api_key = "XXX"
            # Function to get trend prediction from OpenAI's language model
            def get_trend_prediction(prompt):
                response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=[
            #             {"role": "system", "content": "You are a financial analyst model trained to predict trends based on historical data."},
                        {"role": "system", "content": "You are a financial analyst model trained to predict results based on historical data."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=500,
                    temperature=0
                )
                return response['choices'][0]['message']['content'].strip()

        # Get trend prediction for the input text
    #         predicted_trend = get_trend_prediction(prompt)
    #         st.session_state.predicted_trend = predicted_trend
    #         st.write(f"Predicted Trend for Input Text: {predicted_trend}")



            predicted_trend = get_trend_prediction(prompt)

            # Store in chat history
            st.session_state.history.append({"input": input_text, "prediction": predicted_trend})

            # Display prediction
            st.write(f"Predicted Trend for Input Text: {predicted_trend}")

            # Create input text box for user input
            st.write("### Chat with the Bot")
            user_input = st.text_input("Type your message here and press Enter:")

            # Check for user input and generate prediction
            if user_input:
                # Create prompt with reference information and user input
                prompt = f"""

                The chatbot has returned the below reply and i have a few questions and give the proper reason after thinking logically and justify if correct or give correct output. \n
                Reply: 
                Predicted Trend for Input Text: {predicted_trend} \n

                questions: 
                The following are profiles and results from previous market days. Use these as references to determine the result for the provided input text.

                Reference Information:
                {reference_info}

                Input Text:
                {user_input}

                Based on the reference information and the input text, predict the market result for today. 
                Provide the result prediction with the following details:
                - The stock will close above/below yesterday's close
                - The stock will close above/below the Previous Day High
                - The stock will close above/below the Previous Day Low
                - The stock will close above/below Initial Balance High
                - The stock will close above/below Initial Balance Low
                """

                # Get trend prediction
                predicted_trend = get_trend_prediction(prompt)

                # Append user message and bot response to the chat history
                st.session_state.history.append({"role": "user", "content": user_input})
                st.session_state.history.append({"role": "assistant", "content": predicted_trend})

    #     # Display chat history
    #     if st.session_state.history:
    #         st.write("### Chat History")
    #         for i, entry in enumerate(st.session_state.history, 1):
    #             st.write(f"**User Input {i}:** {entry['input']}")
    #             st.write(f"**Predicted Trend {i}:** {entry['prediction']}")

        
        # Probability of repeatability based on the types of days
        day_type_summary = final_data['Day Type'].value_counts().reset_index()
        day_type_summary.columns = ['Day Type', 'Number of Days']
        total_days = len(final_data)
        day_type_summary['Probability of Repeatability (%)'] = (day_type_summary['Number of Days'] / total_days) * 100

        # Display the probability summary
        st.title(f"Probability Summary for {ticker}")
        st.write(day_type_summary)

        # Generate the Comparison Matrix
        comparison_summary = pd.DataFrame({
            "Day Type": ["Normal Day", "Normal Variation Day", "Trend Day", "Neutral Day"],
            "Number of Days (Selected Stock)": [
                day_type_summary.loc[day_type_summary['Day Type'] == 'Normal Day', 'Number of Days'].values[0] if 'Normal Day' in day_type_summary['Day Type'].values else 0,
                day_type_summary.loc[day_type_summary['Day Type'] == 'Normal Variation Day', 'Number of Days'].values[0] if 'Normal Variation Day' in day_type_summary['Day Type'].values else 0,
                day_type_summary.loc[day_type_summary['Day Type'] == 'Trend Day', 'Number of Days'].values[0] if 'Trend Day' in day_type_summary['Day Type'].values else 0,
                day_type_summary.loc[day_type_summary['Day Type'] == 'Neutral Day', 'Number of Days'].values[0] if 'Neutral Day' in day_type_summary['Day Type'].values else 0
            ],
            "Probability of Repeatability (Selected Stock)": [
                day_type_summary.loc[day_type_summary['Day Type'] == 'Normal Day', 'Probability of Repeatability (%)'].values[0] if 'Normal Day' in day_type_summary['Day Type'].values else 0,
                day_type_summary.loc[day_type_summary['Day Type'] == 'Normal Variation Day', 'Probability of Repeatability (%)'].values[0] if 'Normal Variation Day' in day_type_summary['Day Type'].values else 0,
                day_type_summary.loc[day_type_summary['Day Type'] == 'Trend Day', 'Probability of Repeatability (%)'].values[0] if 'Trend Day' in day_type_summary['Day Type'].values else 0,
                day_type_summary.loc[day_type_summary['Day Type'] == 'Neutral Day', 'Probability of Repeatability (%)'].values[0] if 'Neutral Day' in day_type_summary['Day Type'].values else 0
            ]
        })

        st.title(f"Comparison Matrix for {ticker}")
        st.write(comparison_summary)

        import plotly.express as px

        # Group by 'Day Type' and count occurrences
        day_type_summary = final_data.groupby('Day Type').size().reset_index(name='Counts')

        # Group by 'IB Range' and count occurrences
        ib_range_summary = final_data.groupby('IB Range').size().reset_index(name='Counts')

        # Group by 'Trend' and count occurrences
        trend_summary = final_data.groupby('Trend').size().reset_index(name='Counts')

        # Group by 'Initial Balance Range' and count occurrences
        prev_day_type_summary = final_data.groupby('Initial Balance Range').size().reset_index(name='Counts')

        # Visualizing the count of different 'Day Types'
        fig_day_type = px.bar(day_type_summary, x='Day Type', y='Counts', title='Distribution of Day Types')
        st.plotly_chart(fig_day_type)

        # Visualizing the count of different 'IB Ranges'
        fig_ib_range = px.bar(ib_range_summary, x='IB Range', y='Counts', title='Distribution of IB Ranges')
        st.plotly_chart(fig_ib_range)

        # Visualizing the count of different 'Trends'
        fig_trend = px.bar(trend_summary, x='Trend', y='Counts', title='Distribution of Market Trends')
        st.plotly_chart(fig_trend)

        # Visualizing the count of 'Initial Balance Ranges'
        fig_prev_day_type = px.bar(prev_day_type_summary, x='Initial Balance Range', y='Counts', title='Initial Balance Range')
        st.plotly_chart(fig_prev_day_type)

        # Visualizing the comparison between '2 Day VAH and VAL' 
        fig_vah_val = px.scatter(final_data, x='VAH', y='VAL', color='IB Range', title='VAH vs VAL with IB Range')
        st.plotly_chart(fig_vah_val)

        # Visualizing the relationship between Initial Balance Range and Day Range
        fig_ib_day_range = px.scatter(final_data, x='Initial Balance Range', y='Day Range', color='Day Type', title='Initial Balance Range vs Day Range')
        st.plotly_chart(fig_ib_day_range)
    else:
        st.warning(f"No data found for the selected date: {start_date}")
    # Section 5: Trade Performance Monitoring
    st.title("Trade Performance Monitoring")
    uploaded_file = st.file_uploader("Upload Trade Data (CSV)", type="csv")

    if uploaded_file is not None:
        trades_df = pd.read_csv(uploaded_file)
        st.write(trades_df)
        st.line_chart(trades_df[['Profit/Loss']])

    # Section 6: LLM Chat for Strategy Insights
    st.title("AI Chat for Strategy Insights")

    if st.button("Ask AI about strategy performance"):
        llm_response = ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are an assistant for a day trader analyzing strategies."},
                {"role": "user", "content": f"What is your assessment of the {selected_strategy} strategy's performance?"}
            ]
        )
        st.write(llm_response.choices[0].message['content'])

    st.success(f"Monitoring strategy '{selected_strategy}' in real-time.")