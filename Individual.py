#Import all the important Libraries for all 7 tabs
import streamlit as st
import yfinance as yf
import datetime
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


# Defining Functions for Tab 1

def get_sp500_details():
    try:
        link = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
        stock_table = pd.read_html(link, header=0)[0]
        options = stock_table.set_index('Symbol').T.to_dict('list')
        return options, list(stock_table.Symbol)
    except:
        st.error("Yfinance is currently not available to fetch data. Please try again later")
        return {}, []

def get_stock_data(ticker, timeframe):
    stock = yf.Ticker(ticker)
    today = datetime.date.today()
    if timeframe == '1M':
        start_date = today - datetime.timedelta(days=30)
    elif timeframe == '3M':
        start_date = today - datetime.timedelta(days=90)
    elif timeframe == '6M':
        start_date = today - datetime.timedelta(days=180)
    elif timeframe == 'YTD':
        start_date = datetime.date(today.year, 1, 1)
    elif timeframe == '1Y':
        start_date = today - datetime.timedelta(days=365)
    elif timeframe == '3Y':
        start_date = today - datetime.timedelta(days=3*365)
    elif timeframe == '5Y':
        start_date = today - datetime.timedelta(days=5*365)
    elif timeframe == 'MAX':
        start_date = '1900-01-01'
    else:
        start_date = '1900-01-01'
    data = stock.history(start=start_date, end=today, interval='1d')
    return data, stock

#If Tab1 is chosen, the display environment 

def tab1():
    st.title("Summary")
    sp500_details, tickers = get_sp500_details()
    selected_stock = st.selectbox('Select Stock 📈 ', tickers, key='tab1_selectbox1')
    timeframe = st.radio("Select timeframe 📅 ", ['1M', '3M', '6M', 'YTD', '1Y', '3Y', '5Y', 'MAX'], key='radio_option_1')
    update = st.button("Fetch Stock Summary")

    company_details = sp500_details.get(selected_stock, ["N/A"] * 9)
    st.subheader(f"Company Profile of {company_details[0]}")
    st.write(f"**{company_details[0]} ({selected_stock})**")
    st.write("Security:", company_details[0])
    st.write("Headquarters:", company_details[3] if len(company_details) > 3 else "N/A")
    st.write("Date Founded:", company_details[6] if len(company_details) > 3 else "N/A")
    st.write("GICS Sector:", company_details[1] if len(company_details) > 3 else "N/A")
    st.write("GICS Sub Industry:", company_details[2] if len(company_details) > 3 else "N/A")
    st.write("Central Index Key:", company_details[5] if len(company_details) > 3 else "N/A")


    if update:
        data, stock = get_stock_data(selected_stock, timeframe)
        st.write(f"**Summary for {selected_stock}**")
        st.write("Previous Close:", data['Close'].iloc[-1] if not data.empty else "N/A")
        st.write("Day's Range:", f"{data['Low'].iloc[-1]} - {data['High'].iloc[-1]}" if not data.empty else "N/A")
        st.write("Open:", data['Open'].iloc[-1] if not data.empty else "N/A")
        st.write("52 Week Range:", f"{data['Low'].min()} - {data['High'].max()}" if not data.empty else "N/A")
        st.write("Volume:", data['Volume'].iloc[-1] if not data.empty else "N/A")
        if not data.empty:
                st.area_chart(data['Close'])

# Defining Functions for Tab 2

def fetch_sp500_stocks():
    table = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
    df = table[0]
    return df.Symbol.tolist()


def get_major_holders(ticker):
    stocks = yf.Ticker(ticker)
    return stocks.major_holders


def get_institutional_holders(ticker):
    stocks = yf.Ticker(ticker)
    return stocks.institutional_holders

#If Tab2 is chosen, the display environment 

def tab2():
    st.title("S&P 500 Stock: Holders")
    selected_stock = st.selectbox("Stock 📈 ", fetch_sp500_stocks(), key='tab2_selectbox2')

    st.header(f"Major Holders for {selected_stock} Stock")
    list_major_holders = get_major_holders(selected_stock)
    st.table(list_major_holders)

    st.header(f"Top Institutional Holders for {selected_stock} Stock")
    list_institution_holders = get_institutional_holders(selected_stock)
    st.table(list_institution_holders)

# Defining Functions for Tab 3

def sp_500stocks():
    try:
        url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
        table = pd.read_html(url, header=0)[0]
        details = table.set_index('Symbol').T.to_dict('list')
        return details, list(table.Symbol)
    except:
        st.error("There was an error fetching the S&P 500 details.")
        return {}, []

def fetch_stock_info(ticker, timeframe):
    stock = yf.Ticker(ticker)
    today = datetime.date.today()
    if timeframe == '1M':
        start_date = today - datetime.timedelta(days=30)
    elif timeframe == '3M':
        start_date = today - datetime.timedelta(days=90)
    elif timeframe == '6M':
        start_date = today - datetime.timedelta(days=180)
    elif timeframe == 'YTD':
        start_date = datetime.date(today.year, 1, 1)
    elif timeframe == '1Y':
        start_date = today - datetime.timedelta(days=365)
    elif timeframe == '3Y':
        start_date = today - datetime.timedelta(days=3*365)
    elif timeframe == '5Y':
        start_date = today - datetime.timedelta(days=5*365)
    else:
        start_date = '1900-01-01'
    data = stock.history(start=start_date, end=today)
    data['Pct Change'] = data['Close'].pct_change() * 100
    
# Calculate the 50-day moving average to show a clearer analysis
    data['MA50'] = data['Close'].rolling(window=50).mean()
    return data

#If Tab3 is chosen, the display environment 

def tab3():
    st.title("Chart")
    
    sp500_details, tickers = sp_500stocks()
    selected_stock = st.selectbox('Select Your Stock 📈 ', tickers, key='tab3_selectbox3')
    timeframe = st.radio("Select Your Timeframe 📅 ", ['1M', '3M', '6M', 'YTD', '1Y', '3Y', '5Y', 'MAX'])
    chart_type = st.radio("Select Your Type 📊", ['Line', 'Candle'])

    data = fetch_stock_info(selected_stock, timeframe)

    if not data.empty:
        hovertext = []
        for index, row in data.iterrows():
            hovertext.append((
                f"Date: {index}<br>"
                f"Open: {row['Open']}<br>"
                f"High: {row['High']}<br>"
                f"Low: {row['Low']}<br>"
                f"Close: {row['Close']}<br>"
                f"Volume: {row['Volume']}<br>"
                f"Pct Change: {row['Pct Change']:.2f}%"
            ))

        chart = go.Figure()

        if chart_type == "Line":
            chart = make_subplots(rows=2, cols=1, shared_xaxes=True, subplot_titles=('Stock Price', '50-day MA'))
            chart.add_trace(go.Scatter(x=data.index, y=data['Close'], mode='lines', name='Close', text=hovertext, hoverinfo="text"), row=1, col=1)
            chart.add_trace(go.Scatter(x=data.index, y=data['MA50'], mode='lines', name='50-day MA', line=dict(color='orange')), row=2, col=1)
        else:
            chart.add_trace(go.Candlestick(x=data.index,
                                         open=data['Open'],
                                         high=data['High'],
                                         low=data['Low'],
                                         close=data['Close'],
                                         name='OHLC'))
            chart.add_trace(go.Scatter(x=data.index, y=data['MA50'], mode='lines', name='50-day MA', line=dict(color='orange')))
    
        st.plotly_chart(chart)

#Defining Functions for Tab 4

def get_sp500_details():
    try:
        url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
        table = pd.read_html(url, header=0)[0]
        details = table.set_index('Symbol').T.to_dict('list')
        return details, list(table.Symbol)
    except:
        st.error("There was an error fetching the S&P 500 details.")
        return {}, []

# Function to fetch financial data and type in Tab 4

def get_financial_data(ticker, financial_type, period):
    stock = yf.Ticker(ticker)
    if financial_type == 'Income Statement':
        if period == 'Annual':
            return stock.financials
        else:
            quarterly_data = stock.quarterly_financials
            quarterly_data['TTM'] = quarterly_data.sum(axis=1)
            return quarterly_data
    elif financial_type == 'Balance Sheet':
        if period == 'Annual':
            return stock.balance_sheet
        else:
            quarterly_data = stock.quarterly_balance_sheet
            quarterly_data['TTM'] = quarterly_data.sum(axis=1)
            return quarterly_data
    elif financial_type == 'Cash Flow':
        if period == 'Annual':
            return stock.cashflow
        else:
            quarterly_data = stock.quarterly_cashflow
            quarterly_data['TTM'] = quarterly_data.sum(axis=1)
            return quarterly_data

#If Tab4 is chosen, the display environment 

def tab4():
    st.title("Financials")

#Options for user to select from
    
    sp500_details, tickers = get_sp500_details()
    selected_stock = st.selectbox('Select Stock 📈', tickers, key='tab4_selectbox4')
    financial_type = st.radio("Select Type 📊", ['Income Statement', 'Balance Sheet', 'Cash Flow'])
    period = st.radio("Select Period 📅", ['Annual', 'Quarterly'])

# Fetch and display your finnacial data sheet as selected

    financial_data = get_financial_data(selected_stock, financial_type, period)
    st.write(f"**{financial_type} for {selected_stock} ({period})**")
    st.dataframe(financial_data)  # We transpose for display in Streamlit

# Defining Function for Tab 5


def fetch_sp500_stocks():
    table = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
    df = table[0]
    return df.Symbol.tolist()


def get_stock_news(ticker):
    stocks = yf.Ticker(ticker)
    return stocks.news

#If Tab5 is chosen, the display environment 

def tab5():
    st.title("News")

#Options for user to select from
    
    selected_stock = st.selectbox("Stock 📈 ", fetch_sp500_stocks(), key='tab5_selectbox5')

# News
    st.header(f"Catch up on the latest updates on {selected_stock}")
    stock_news = get_stock_news(selected_stock)
    if stock_news:
        for news in stock_news:
            st.write(f"**{news.get('title', 'No Title Available')}**")
            st.write(news.get('description', 'Click link below to read more'))
            st.write(f"_[Click here to open news article]({news.get('link', '#')})_")
            st.write("-----")
    else:
        st.write("No recent news available.")


#Defining Function for Tab 6

def sp500_stock():
    url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    table = pd.read_html(url, header=0)[0]
    tickers = list(table.Symbol)
    return tickers


# Fetch Current Stock Price
def get_current_price(ticker):
    stock = yf.Ticker(ticker)
    data = stock.history(period="1d")
    return data['Close'].iloc[0]



# Monte Carlo Simulation Function

def monte_carlo_simulation(ticker, nsim, days):
    # Fetching historical data from yahoofinance
    stock = yf.Ticker(ticker)
    data = stock.history(period="1y")
    
    # Calculate daily returns to run the simulation
    returns = data['Close'].pct_change().dropna()

    # Mean and standard deviation of daily returns using functions
    mean = returns.mean()
    std = returns.std()

    # Simulate the returns with the days
    simulations = np.zeros((nsim, days))
    for i in range(nsim):
        random_returns = np.random.normal(mean, std, days)
        simulations[i] = data['Close'].iloc[-1] * (random_returns + 1).cumprod()

    return simulations, data['Close'].iloc[-1]


#If Tab6 is chosen, the display environment 

def tab6():
    st.title("Monte Carlo Simulation for Stock Prices")

# Options
    
    tickers = sp500_stock()
    selected_ticker = st.selectbox("Select Stock 📈", tickers, key='tab6_selectbox6')

# Display Current Stock Price
    current_price = get_current_price(selected_ticker)
    st.markdown(f"Current Price of **{selected_ticker}**: **${current_price:.2f}**")

#Choosing your options 

    nsim = st.selectbox("Number of Simulations", [200, 500, 1000])
    days = st.selectbox("Time Horizon (Days)", [30, 60, 90])

# Run Simulation
    if st.button("Run Simulation"):
        simulations, last_close = monte_carlo_simulation(selected_ticker, nsim, days)

    # Plotting the simulation with the selected stocks and days
        fig, ax = plt.subplots(figsize=(10, 5))
        for i in range(nsim):
            ax.plot(simulations[i], linewidth=0.5)
        ax.set_title(f"Monte Carlo Simulation for {selected_ticker} over {days} Days")
        ax.set_xlabel("Days")
        ax.set_ylabel("Stock Price")
        st.pyplot(fig)

    # 95% VaR calculated to be displayed
        var_95 = np.percentile(simulations, 5, axis=0)
        st.write(f"Value at Risk (VaR) at 95% confidence for {days} days: ${last_close - var_95[-1]:.2f}")


# Defining Function for Tab 7

def sp500_stock():
    url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    table = pd.read_html(url, header=0)[0]
    tickers = list(table.Symbol)
    return tickers

def fetch_data(ticker):
    stock = yf.Ticker(ticker)
    data = stock.history(period="max")
    return data

def forecast_dividend_yield(data):
    data = data.dropna()
    data['Year'] = data.index.year
    X = data['Year'].values.reshape(-1,1)
    y = data['Dividends'].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    regressor = LinearRegression()
    regressor.fit(X_train, y_train)
    y_pred = regressor.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    
    next_year = np.array([data['Year'].iloc[-1] + 1]).reshape(-1,1)
    forecasted_dividend = regressor.predict(next_year)[0]
    
    return forecasted_dividend, mse

#If Tab7 is chosen, the display environment 

def tab7():
    st.title('Dividend Yield Forecast')
    
    tickers = sp500_stock()
    selected_ticker = st.selectbox("Select Stock 📈 ",tickers, key='tab7_selectbox7')  # Add your preferred stock tickers
    data = fetch_data(selected_ticker)
    forecasted_dividend, mse = forecast_dividend_yield(data)

    st.write("Historical Data:")
    st.dataframe(data.tail())  # Display the last few rows of the data

# Plotting a line graph upon selected options

    st.write("Historical Dividend Chart:")
    plt.figure(figsize=(10, 4))
    plt.plot(data['Dividends'])
    plt.title('Dividends Over Time')
    plt.xlabel('Year')
    plt.ylabel('Dividends')
    st.pyplot(plt)

    st.write(f"Forecasted Dividend Yield for {selected_ticker} for next year: ${forecasted_dividend:.2f}")
    st.write(f"Mean Squared Error of the Model: {mse:.2f}")

# Main Function to combine all the Tabs

def main():
    tab = st.tabs(["Summary: Tab 1", "Holders: Tab 2","Chart Analysis: Tab 3", "Financials: Tab 4","News: Tab 5","Monte Carlo: Tab 6","Dividends: Tab 7"])
    with tab[0]:
        tab1()
    with tab[1]:
        tab2()
    with tab[2]:
        tab3()
    with tab[3]:
        tab4()
    with tab[4]:
        tab5()
    with tab[5]:
        tab6()
    with tab[6]:
        tab7()


if __name__ == "__main__":
    main()
