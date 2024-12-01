import streamlit as st
import pandas as pd
import json
from datetime import datetime, timedelta
import numpy as np
import plotly.graph_objects as go
from scipy.signal import find_peaks

# Title of the app
st.title("File Upload and Timeframe Selection")

# File uploader
uploaded_file = st.file_uploader("Upload a text file")

# Dropdown for timeframe selection
timeframe_options = ["5 seconds", "5 minutes", "1 day"]
timeframe = st.selectbox("Select Timeframe", timeframe_options, index=1)  # Default to "5 minutes"

# Date and time picker for base timestamp
base_date = st.date_input("Select Base Date", datetime(2024, 6, 14).date())
base_time = st.time_input("Select Base Time", datetime.strptime("15:55:00", "%H:%M:%S").time())

# Combine date and time into a single datetime object
base_timestamp = datetime.combine(base_date, base_time)

# Date picker for subsetting the data
subset_date = st.date_input("Select Subset Date", datetime(2024, 6, 1).date())

# Define the trading holidays
trading_holidays = [
    datetime(2024, 1, 1),   # Monday, January 1
    datetime(2024, 1, 15),  # Monday, January 15
    datetime(2024, 2, 19),  # Monday, February 19
    datetime(2024, 3, 29),  # Friday, March 29
    datetime(2024, 5, 27),  # Monday, May 27
    datetime(2024, 6, 19),  # Wednesday, June 19
    datetime(2024, 7, 4)    # Thursday, July 4
]

# Submit button
if st.button("Submit"):
    if uploaded_file is not None:
        # Read and process the uploaded file
        file_content = uploaded_file.read().decode("utf-8")
        
        # Strip everything after the ~m~98~m~ marker
        marker = '~m~98~m~'
        file_content = file_content.split(marker)[0]

        try:
            # Load the JSON data from the file content
            main_data = json.loads(file_content)
            
            data_section = main_data['p'][1]['st1']['ns']['d']
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

            # Initialize the current timestamp with the selected base timestamp
            current_timestamp = base_timestamp

            # Initialize a dictionary to store timestamps for each index
            index_to_timestamp = {max(df1['index']): current_timestamp}

            # Market hours
            market_open = timedelta(hours=9, minutes=30)
            market_close = timedelta(hours=15, minutes=55)
            day_increment = timedelta(days=1)
            weekend_days = [5, 6]  # Saturday and Sunday

            # Calculate the timestamps backward in 5-minute intervals, excluding weekends and outside market hours
            for index in range(max(df1['index']) - 1, -1, -1):
                # Subtract 5 minutes
                current_timestamp -= timedelta(minutes=5)

                # Check if current timestamp is before market open
                while (current_timestamp.time() < (datetime.min + market_open).time() or
                       current_timestamp.time() > (datetime.min + market_close).time() or
                       current_timestamp.weekday() in weekend_days or
                       current_timestamp.date() in [holiday.date() for holiday in trading_holidays]):
                    # Move to previous trading day if before market open
                    if current_timestamp.time() < (datetime.min + market_open).time():
                        current_timestamp = datetime.combine(current_timestamp.date() - day_increment, (datetime.min + market_close).time())
                    else:
                        # Otherwise, just subtract 5 minutes
                        current_timestamp -= timedelta(minutes=5)

                    # Skip weekends and trading holidays
                    while current_timestamp.weekday() in weekend_days or current_timestamp.date() in [holiday.date() for holiday in trading_holidays]:
                        current_timestamp -= day_increment
                        current_timestamp = datetime.combine(current_timestamp.date(), (datetime.min + market_close).time())

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
