import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.stats import skew, kurtosis
from collections import defaultdict

# Sidebar for inputs
st.sidebar.title("Trading Dashboard")
capital_per_trade = st.sidebar.number_input("Capital Per Trade", value=2000, min_value=100)
selected_strategy = st.sidebar.selectbox("Select Strategy", ['Momentum', 'Reversal', 'Breakout'])

# Section 1: Stock and Volume Profile Inputs
st.title("Real-time Volume Profile with Market Shape Detection")

ticker = st.text_input("Enter Stock Ticker", value="AAPL")
start_date = st.date_input("Start Date", value=pd.to_datetime("2024-10-25"))

# Fetch stock data in real-time
def fetch_stock_data(ticker, start, interval='1m'):
    stock_data = yf.download(ticker, start=start, interval=interval)
    return stock_data

data = fetch_stock_data(ticker, start_date)

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
current_date_index = sorted_dates.index(start_date) if start_date in sorted_dates else None

# Determine the previous date if it exists
previous_date = sorted_dates[current_date_index - 1] if current_date_index and current_date_index > 0 else None
        

# Filter based on the input date (start_date) from the sidebar
filtered_data = final_data[final_data['casted_date'] == start_date]

# Filter based on the input date (start_date) from the sidebar
previous_filtered_data = final_data[final_data['casted_date'] == previous_date]
# st.write(filtered_data.columns)

# Section 7: Display the Data for Selected Date
if not filtered_data.empty:
    st.title(f"Market Profile for {start_date}")
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
    
    # Generate the LLM input text with added indicators
    input_text = (
        f"Today’s profile on {start_date} for {ticker} indicates an {current_row['IB Range']} Range. The market opened at {opening_price}, "
        f"which is {open_percent_diff}% ({abs(open_point_diff)} points) {open_above_below} the previous day's close. "
        f"The Initial Balance High is {current_row['Initial Balance High']} and Low is {current_row['Initial Balance Low']}, "
        f"giving an Initial Balance Range of {current_row['Initial Balance Range']}. "
        f"Yesterday's VAH was {last_row['VAH']} and VAL was {last_row['VAL']}. "
        f"Day before yesterday's VAH was {last_row['Previous Day VAH']} and VAL was {last_row['Previous Day VAL']}. "
        f"Previous day Type: {last_row['Day Type']}.\n"
        f"Previous Adjusted Day Type: {final_day_type}.\n"
        f"Previous Close Type: {last_row['Close Type']}.\n"
        f"Previous 2 Day VAH and VAL: {current_row['2 Day VAH and VAL']}.\n"

        # Adding indicators
        f"Moving Average (20-day) is {last_row['MA_20']}. "
        f"Relative Strength Index (RSI) is {last_row['RSI']}. "
        f"MACD is {last_row['MACD']} with Signal Line at {last_row['Signal Line']}. "
        f"Bollinger Bands Upper at {last_row['Bollinger_Upper']} and Lower at {last_row['Bollinger_Lower']}. "
        f"Volume Weighted Average Price (VWAP) is {last_row['VWAP']}. "
        f"Fibonacci Levels: 38.2% at {last_row['Fib_38.2']}, 50% at {last_row['Fib_50']}, 61.8% at {last_row['Fib_61.8']}. "
        f"Average True Range (ATR) is {last_row['ATR']}. "
        f"Stochastic Oscillator %K is {last_row['%K']} and %D is {last_row['%D']}. "
        f"Parabolic SAR is at {last_row['PSAR']}. "

        f"Given these indicators, what is the expected market direction for today?"
    )

    st.write(input_text)
    
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
