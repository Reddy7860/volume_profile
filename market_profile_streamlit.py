import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.stats import skew, kurtosis
from collections import defaultdict
import requests
from io import StringIO
from datetime import datetime
import re
from sentence_transformers import SentenceTransformer, util
import openai

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

# Main interface with tab selection
tab = st.selectbox("Select Tab", ["Chat Interface", "Headers and Cookies"])

if tab == "Headers and Cookies":
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

    ticker = st.text_input("Enter Stock Ticker", value="AAPL")
    default_date = datetime.today().date()
    input_date = st.date_input("Start Date", value=default_date)

    # Fetch stock data in real-time
    def fetch_stock_data(ticker, start, interval='1m'):
        stock_data = yf.download(ticker, start=start, interval=interval)
        return stock_data

    data = fetch_stock_data(ticker, input_date)

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

    # st.write(data)

    # Section 2: True Range and ATR Calculations
    def calculate_atr(data, atr_period=14):
        data['H-L'] = data['High'] - data['Low']
        data['H-PC'] = np.abs(data['High'] - data['Adj Close'].shift(1))
        data['L-PC'] = np.abs(data['Low'] - data['Adj Close'].shift(1))
        data['TR'] = data[['H-L', 'H-PC', 'L-PC']].max(axis=1)
        data['ATR_14'] = data['TR'].ewm(alpha=1/atr_period, adjust=False).mean()
        return data

    data = calculate_atr(data)

    # Fetch daily data to calculate ATR for daily intervals
    daily_data = yf.download(ticker, period="60d", interval="1d").reset_index()
    daily_data['H-L'] = daily_data['High'] - daily_data['Low']
    daily_data['H-PC'] = np.abs(daily_data['High'] - daily_data['Adj Close'].shift(1))
    daily_data['L-PC'] = np.abs(daily_data['Low'] - daily_data['Adj Close'].shift(1))
    daily_data['TR'] = daily_data[['H-L', 'H-PC', 'L-PC']].max(axis=1)
    daily_data['ATR_14_1_day'] = daily_data['TR'].ewm(alpha=1/14, adjust=False).mean()
    daily_data['Prev_Day_ATR_14_1_Day'] = daily_data['ATR_14_1_day'].shift(1)
    daily_data['Date'] = pd.to_datetime(daily_data['Date']).dt.date

    # Merge ATR into 30-minute data
    data['Date'] = pd.to_datetime(data['Datetime']).dt.date
    final_data = pd.merge(data, daily_data[['Date', 'ATR_14_1_day', 'Prev_Day_ATR_14_1_Day']], on='Date', how='left')

    # Calculate Moving Average (MA)
    final_data['MA_20'] = final_data['Close'].rolling(window=20).mean()

    # Calculate Relative Strength Index (RSI)
    delta = final_data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    final_data['RSI'] = 100 - (100 / (1 + gain / loss))

    # Calculate Moving Average Convergence Divergence (MACD)
    short_ema = final_data['Close'].ewm(span=12, adjust=False).mean()
    long_ema = final_data['Close'].ewm(span=26, adjust=False).mean()
    final_data['MACD'] = short_ema - long_ema
    final_data['Signal Line'] = final_data['MACD'].ewm(span=9, adjust=False).mean()

    # Calculate Bollinger Bands
    final_data['MA_20'] = final_data['Close'].rolling(window=20).mean()
    final_data['Bollinger_Upper'] = final_data['MA_20'] + (final_data['Close'].rolling(window=20).std() * 2)
    final_data['Bollinger_Lower'] = final_data['MA_20'] - (final_data['Close'].rolling(window=20).std() * 2)

    # Calculate Volume Weighted Average Price (VWAP)
    final_data['VWAP'] = (final_data['Volume'] * (final_data['High'] + final_data['Low'] + final_data['Close']) / 3).cumsum() / final_data['Volume'].cumsum()

    # Calculate Fibonacci Retracement Levels (use high and low from a specific range if applicable)
    highest = final_data['High'].max()
    lowest = final_data['Low'].min()
    final_data['Fib_38.2'] = highest - (highest - lowest) * 0.382
    final_data['Fib_50'] = (highest + lowest) / 2
    final_data['Fib_61.8'] = highest - (highest - lowest) * 0.618

    # Calculate Average True Range (ATR)
    high_low = final_data['High'] - final_data['Low']
    high_close = np.abs(final_data['High'] - final_data['Close'].shift())
    low_close = np.abs(final_data['Low'] - final_data['Close'].shift())
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    final_data['ATR'] = true_range.rolling(window=14).mean()

    # Calculate Stochastic Oscillator
    final_data['14-high'] = final_data['High'].rolling(window=14).max()
    final_data['14-low'] = final_data['Low'].rolling(window=14).min()
    final_data['%K'] = (final_data['Close'] - final_data['14-low']) * 100 / (final_data['14-high'] - final_data['14-low'])
    final_data['%D'] = final_data['%K'].rolling(window=3).mean()

    # Calculate Parabolic SAR (for simplicity, this example uses a fixed acceleration factor)
    final_data['PSAR'] = final_data['Close'].shift() + (0.02 * (final_data['High'] - final_data['Low']))

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

    # Use a for loop with range() to iterate over the sorted dates by index
    for i in range(2, len(sorted_dates)):
        date = sorted_dates[i]
        previous_date = sorted_dates[i - 1]

        print(f"Current Date: {date}")
        print(f"Previous Date: {previous_date}")

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


        # Print or perform further operations with actions
        print(actions)

        final_data.loc[final_data['Date'] == previous_date, '2 Day VAH and VAL'] = str(actions)
        final_data.loc[final_data['Date'] == previous_date, 'Adjusted Day Type'] = str(final_day_type)


    # Create a 'casted_date' column to only capture the date part of the Datetime
    final_data['casted_date'] = final_data['Date']

    # Sort by casted_date to ensure correct order
    final_data = final_data.sort_values(by='Datetime')

    # Create a 'casted_date' column to only capture the date part of the Datetime
    final_data['casted_date'] = final_data['Date']

    # Get a sorted list of unique dates
    sorted_dates = sorted(final_data['casted_date'].unique())

    # Find the index of the selected date in the sorted list
    current_date_index = sorted_dates.index(input_date) if input_date in sorted_dates else None

    # Determine the previous date if it exists
    previous_date = sorted_dates[current_date_index - 1] if current_date_index and current_date_index > 0 else None


    # Filter based on the input date (input_date) from the sidebar
    filtered_data = final_data[final_data['casted_date'] == input_date]

    # Filter based on the input date (input_date) from the sidebar
    previous_filtered_data = final_data[final_data['casted_date'] == previous_date]
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
                "Referer": "https://www.barchart.com/stocks/quotes/TSLA/interactive-chart/new",
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
            BASE_URL = "https://www.barchart.com/proxies/timeseries/historical/queryminutes.ashx?symbol=TSLA&interval=30&maxrecords=640&volume=contract&order=asc&dividends=false&backadjust=false&daystoexpiration=1&contractroll=combined"


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
            next_url = f"https://www.barchart.com/proxies/timeseries/historical/queryminutes.ashx?symbol=TSLA&interval=30&maxrecords=640&end={formatted_min_timestamp}&volume=contract&order=asc&dividends=false&backadjust=false&daystoexpiration=1&contractroll=combined"

            # Define the recursive data fetching function
            def fetch_data_until_start_date(start_date='2024-01-01'):
                all_data = pd.DataFrame()  # Initialize an empty DataFrame to store all data
                next_url = BASE_URL  # Start with the base URL

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
            merged_data = merged_data[merged_data['Datetime'] < pd.to_datetime(input_date)]


            # Display the first few rows of the combined data
            st.write(merged_data.head())
            st.write(merged_data.tail())

            data = merged_data.copy()

            # Parameters for TPO calculation
            value_area_percent = 70
            tick_size = 0.01  # Tick size as per the image

            # Calculate ATR for 30-minute and daily intervals
            atr_period = 14

            # True Range Calculation for 30-minute data
            data['H-L'] = data['High'] - data['Low']
            data['H-PC'] = np.abs(data['High'] - data['Close'].shift(1))
            data['L-PC'] = np.abs(data['Low'] - data['Close'].shift(1))
            data['TR'] = data[['H-L', 'H-PC', 'L-PC']].max(axis=1)

            # ATR for 30-minute data
            data['ATR_14_30_mins'] = data['TR'].ewm(alpha=1/atr_period, adjust=False).mean()



            # # Fetch daily data for Apple to calculate daily ATR
            # daily_data = yf.download(ticker, period="60d", interval="1d")

            start_date = start_date_str
            end_date = datetime.today().strftime('%Y-%m-%d')

            # Download data from start_date until today
            daily_data = yf.download(ticker, start=start_date, end=end_date, interval="1d")

            # Reset index to ensure 'Date' is a column, not an index
            daily_data = daily_data.reset_index()

            # True Range Calculation for daily data
            daily_data['H-L'] = daily_data['High'] - daily_data['Low']
            daily_data['H-PC'] = np.abs(daily_data['High'] - daily_data['Close'].shift(1))
            daily_data['L-PC'] = np.abs(daily_data['Low'] - daily_data['Close'].shift(1))
            daily_data['TR'] = daily_data[['H-L', 'H-PC', 'L-PC']].max(axis=1)
            daily_data['ATR_14_1_day'] = daily_data['TR'].ewm(alpha=1/atr_period, adjust=False).mean()
            # Shift the ATR to get the previous day's ATR
            daily_data['Prev_Day_ATR_14_1_Day'] = daily_data['ATR_14_1_day'].shift(1)

            # Ensure consistent date format
            data['Date'] = pd.to_datetime(data['Datetime']).dt.date
            daily_data['Date'] = pd.to_datetime(daily_data['Date']).dt.date

            # Merge ATR from daily data into 30-minute data
            final_data = pd.merge(data, daily_data[['Date', 'ATR_14_1_day','Prev_Day_ATR_14_1_Day']], on='Date', how='left')

            final_data = final_data.drop(['H-L', 'H-PC','L-PC','TR'], axis=1)

            # Round all columns to 2 decimal places
            final_data = final_data.round(2)

            # TPO Profile Calculation
            def calculate_tpo(data, tick_size, value_area_percent):
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
                value_area_high = 0
                value_area_low = float('inf')

                for price, counts in sorted_tpo:
                    if value_area_tpo + len(counts) <= value_area_target:
                        value_area_tpo += len(counts)
                        value_area_high = max(value_area_high, price)
                        value_area_low = min(value_area_low, price)
                    else:
                        break

                poc = sorted_tpo[0][0]  # Price with highest TPO count
                vah = value_area_high
                val = value_area_low

                return tpo_counts, poc, vah, val

            # Group by date and calculate TPO profile for each day
            daily_tpo_profiles = []

            for date, group in final_data.groupby('Date'):
                tpo_counts, poc, vah, val = calculate_tpo(group, tick_size, value_area_percent)

                # Calculate Initial Balance Range (IBR)
                initial_balance_high = group['High'].iloc[:2].max()  # First 2 half-hour periods (1 hour)
                initial_balance_low = group['Low'].iloc[:2].min()
                initial_balance_range = initial_balance_high - initial_balance_low

                # Calculate day's total range
                day_range = group['High'].max() - group['Low'].min()
                range_extension = day_range - initial_balance_range

                # Identify Single Prints
                single_prints = [round(price, 2) for price, counts in tpo_counts.items() if len(counts) == 1]

                # Identify Poor Highs and Poor Lows
                high_price = group['High'].max()
                low_price = group['Low'].min()
                poor_high = len(tpo_counts[high_price]) > 1
                poor_low = len(tpo_counts[low_price]) > 1

                # Classify the day
                if day_range <= initial_balance_range * 1.15:
                    day_type = 'Normal Day'
                elif initial_balance_range < day_range <= initial_balance_range * 2:
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


                # Store the results in a dictionary
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
                    'Single Prints': single_prints,
                    'Poor High': poor_high,
                    'Poor Low': poor_low,
                    'Close Type' : close_type
                }
                daily_tpo_profiles.append(tpo_profile)

            # Convert the list of dictionaries to a DataFrame
            tpo_profiles_df = pd.DataFrame(daily_tpo_profiles)

            # Merge TPO profile data into final_data based on the 'Date'
            final_data = pd.merge(final_data, tpo_profiles_df, on='Date', how='left')

            # Calculate Moving Average (MA)
            final_data['MA_20'] = final_data['Close'].rolling(window=20).mean()

            # Calculate Relative Strength Index (RSI)
            delta = final_data['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            final_data['RSI'] = 100 - (100 / (1 + gain / loss))

            # Calculate Moving Average Convergence Divergence (MACD)
            short_ema = final_data['Close'].ewm(span=12, adjust=False).mean()
            long_ema = final_data['Close'].ewm(span=26, adjust=False).mean()
            final_data['MACD'] = short_ema - long_ema
            final_data['Signal Line'] = final_data['MACD'].ewm(span=9, adjust=False).mean()

            # Calculate Bollinger Bands
            final_data['MA_20'] = final_data['Close'].rolling(window=20).mean()
            final_data['Bollinger_Upper'] = final_data['MA_20'] + (final_data['Close'].rolling(window=20).std() * 2)
            final_data['Bollinger_Lower'] = final_data['MA_20'] - (final_data['Close'].rolling(window=20).std() * 2)

            # Calculate Volume Weighted Average Price (VWAP)
            final_data['VWAP'] = (final_data['Volume'] * (final_data['High'] + final_data['Low'] + final_data['Close']) / 3).cumsum() / final_data['Volume'].cumsum()

            # Calculate Fibonacci Retracement Levels (use high and low from a specific range if applicable)
            highest = final_data['High'].max()
            lowest = final_data['Low'].min()
            final_data['Fib_38.2'] = highest - (highest - lowest) * 0.382
            final_data['Fib_50'] = (highest + lowest) / 2
            final_data['Fib_61.8'] = highest - (highest - lowest) * 0.618

            # Calculate Average True Range (ATR)
            high_low = final_data['High'] - final_data['Low']
            high_close = np.abs(final_data['High'] - final_data['Close'].shift())
            low_close = np.abs(final_data['Low'] - final_data['Close'].shift())
            true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            final_data['ATR'] = true_range.rolling(window=14).mean()

            # Calculate Stochastic Oscillator
            final_data['14-high'] = final_data['High'].rolling(window=14).max()
            final_data['14-low'] = final_data['Low'].rolling(window=14).min()
            final_data['%K'] = (final_data['Close'] - final_data['14-low']) * 100 / (final_data['14-high'] - final_data['14-low'])
            final_data['%D'] = final_data['%K'].rolling(window=3).mean()

            # Calculate Parabolic SAR (for simplicity, this example uses a fixed acceleration factor)
            final_data['PSAR'] = final_data['Close'].shift() + (0.02 * (final_data['High'] - final_data['Low']))

            # Determine the trend for each day based on the previous day's levels
            trends = []
            for i in range(1, len(tpo_profiles_df)):
                previous_day = tpo_profiles_df.iloc[i - 1]
                current_day = tpo_profiles_df.iloc[i]

                if current_day['Initial Balance High'] > previous_day['VAH']:
                    trend = 'Bullish'
                elif current_day['Initial Balance Low'] < previous_day['VAL']:
                    trend = 'Bearish'
                else:
                    trend = 'Neutral'

                trend_entry = {
                    'Date': current_day['Date'],
                    'Trend': trend,
                    'Previous Day VAH': round(previous_day['VAH'], 2),
                    'Previous Day VAL': round(previous_day['VAL'], 2),
                    'Previous Day POC': round(previous_day['POC'], 2),
                }

                trends.append(trend_entry)

            # Convert the trends list to a DataFrame
            trends_df = pd.DataFrame(trends)

            # Merge trend data into final_data
            final_data = pd.merge(final_data, trends_df, on='Date', how='left')

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

            # Initialize an empty list to store each day's input text and trend
            training_data = []

            final_data = final_data.sort_values(by='Datetime')

            # Get the unique dates and sort them
            sorted_dates = sorted(set(final_data['Date']))
            final_data['2 Day VAH and VAL'] = ''

            # Iterate over the sorted dates by index, starting from the third day to have data for previous two days
            for i in range(2, len(sorted_dates)):
                date = sorted_dates[i]
                previous_date = sorted_dates[i - 1]
                two_days_ago = sorted_dates[i - 2]

                # Extract data for the current date and previous dates
                current_data = final_data[final_data['Date'] == date]
                previous_data = final_data[final_data['Date'] == previous_date]
                two_days_ago_data = final_data[final_data['Date'] == two_days_ago]

                # Calculate the maximum high and minimum low for the previous day
                day_high = previous_data['High'].max()
                day_low = previous_data['Low'].min()

                # Initialize an empty list for actions based on previous day's close and VAH/VAL comparisons
                actions = []

                if not previous_data.empty:
                    last_row = previous_data.iloc[-1]

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

                # Calculate the opening price and difference from the previous close
                opening_price = current_data.iloc[0]['Open']
                previous_close = previous_data.iloc[-1]['Close']
                open_point_diff = round(opening_price - previous_close, 2) if previous_close else None
                open_percent_diff = round((open_point_diff / previous_close) * 100, 2) if previous_close else None
                open_above_below = "above" if open_point_diff > 0 else "below" if open_point_diff < 0 else "no change"

                current_row = current_data.iloc[0]

            #     # Generate the LLM input text
            #     input_text = (
            #         f"Today’s profile on {date} for {ticker} indicates an {current_row['IB Range']} Range. The market opened at {opening_price}, "
            #         f"which is {open_percent_diff}% ({abs(open_point_diff)} points) {open_above_below} the previous day's close. "
            #         f"The Initial Balance High is {current_row['Initial Balance High']} and Low is {current_row['Initial Balance Low']}, "
            #         f"giving an Initial Balance Range of {current_row['Initial Balance Range']}. "
            #         f"Yesterday's VAH was {last_row['VAH']} and VAL was {last_row['VAL']}. "
            #         f"Day before yesterday's VAH was {last_row['Previous Day VAH']} and VAL was {last_row['Previous Day VAL']}. "
            #         f"Previous day Type : {last_row['Day Type']}\n"
            #         f"Previous Adjusted Day Type : {final_day_type}\n"
            #         f"Previous Close Type : {last_row['Close Type']}\n"
            #         f"Previous 2 Day VAH and VAL : {actions}. "
            #         f"Given these indicators, what is the expected market direction for tomorrow?"
            #     )

                # Generate the LLM input text with added indicators
                input_text = (
                    f"Today’s profile on {date} for {ticker} with IB Range Type {current_row['IB Range']} Range. The market opened at {opening_price}, "
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

                print(input_text)

                current_day_close = current_data.iloc[-1]['Close']
                previous_day_high = previous_data['High'].max()
                previous_day_low = previous_data['Low'].max()

                result = ''
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

                if current_day_close >= current_row['Initial Balance High']:
                    result += 'The stock closed above Initial Balance High \n'
                else:
                    result += 'The stock closed below Initial Balance High \n'

                if current_day_close >= current_row['Initial Balance Low']:
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

            # Calculate similarities for each segment
            similarity_results = []
            for index, row in training_data_df.iterrows():
            #     row_similarity = {'Date': row['Date'], 'Trend': row['Trend']}
                row_similarity = {'Date': row['Date'],'Input Text':row['Input Text'],'Trend':row['Trend'], 'Result': row['Result']}

                # Calculate embedding if 'embedding' column does not exist
                if 'embedding' not in row:
                    row['embedding'] = model.encode(row['Input Text'], convert_to_tensor=True)

                # Accumulators for weighted score
                technical_score = 0
                ib_ranges_score = 0
                gaps_score = 0
                range_type_score = 0

                for segment, details in segments.items():
    #                 st.write(segment)
    #                 st.write(details)
                    if 'category' in details:
                        # Categorical similarity
                        segment_embedding = query_segment_embeddings[segment]
                        semantic_similarity = util.pytorch_cos_sim(segment_embedding, row['embedding']).item()
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
                        semantic_similarity = util.pytorch_cos_sim(segment_embedding, row['embedding']).item()
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
                filtered_data = training_data_df[training_data_df['Date'] == pd.Timestamp(sim['Date'])]

                if not filtered_data.empty:
                    entry_text = filtered_data.iloc[0]['Input Text']
            #         trend = filtered_data.iloc[0]['Trend']
                    result = filtered_data.iloc[0]['Result']
            #         reference_info += f"Date: {sim['Date']}\nInput Text: {entry_text}\nTrend: {trend}\n\n"
                    reference_info += f"Date: {sim['Date']}\nInput Text: {entry_text}\Result: {result}\n\n"

            # Build the full prompt for LLM
            prompt = f"""
            The following are profiles and trends and results from previous market days. Use these as references to determine the result for the provided input text.

            Reference Information:
            {reference_info}

            Input Text:
            {input_text}

            Based on the reference information and the input text, predict the market result for today. 
            Provide the result prediction with below :
            The stock will close above/below yesterday close 
            The stock will close above/below Previous Day High 
            The stock will close above/below Previous Day Low 
            The stock will close above/below Initial Balance High 
            The stock will close above/below Initial Balance Low 
            """

    #         # Get trend prediction
    #         if st.button("Get Prediction"):

            # Set up OpenAI API Key
            openai.api_key = 'XXX'

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



# sk-zoZKWYjBeiDxyd8qGnRid_wBzVOnp6KIwvuJBH8HWsT3BlbkFJTctyWrKzuS30R-3vk7K0dN-o7ewkCsaLOmDB6zZhwA
