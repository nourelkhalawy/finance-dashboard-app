import streamlit as st
from datetime import date
import yfinance as yf
from fbprophet import Prophet
from fbprophet.plot import plot_plotly
from plotly import graph_objs as go
import statsmodels.api as sm
import matplotlib.pyplot as plt
import numpy as np

START = "2015-01-01"
TODAY = date.today().strftime("%Y-%m-%d")

st.title('Stock Forecast App')

stocks = ('GOOG', 'AAPL', 'MSFT', 'GME')
selected_stock = st.selectbox('Select dataset for prediction', stocks)

n_years = st.slider('Years of prediction:', 1, 4)
period = n_years * 365


@st.cache
def load_data(ticker):
    data = yf.download(ticker, START, TODAY)
    data.reset_index(inplace=True)
    return data

	
data_load_state = st.text('Loading data...')
data = load_data(selected_stock)
data_load_state.text('Loading data... done!')

st.subheader('Raw Data')
st.write(data.tail())

# Plot raw data
def plot_raw_data():
	fig = go.Figure()
	fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name="stock_open"))
	fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name="stock_close"))
	fig.layout.update(title_text='Time Series data with Rangeslider', xaxis_rangeslider_visible=True)
	st.plotly_chart(fig)
	
plot_raw_data()

# Predict forecast with Prophet.
df_train = data[['Date','Close']]
df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})

m = Prophet()
m.fit(df_train)
future = m.make_future_dataframe(periods=period)
forecast = m.predict(future)

st.subheader('Autocorretion of Data')
def plot_series(time1, series1, label1, xlabel, ylabel,time2 = [], series2 = [], label2 ="",format="-", start=0, end=None, title="", line=False):
  """
  defining the plot function for plotting all the true and forcasted prices 
  together in the same graph to mak the code clean and concise
  """
  fig, ax = plt.subplots(figsize=(10, 6))
  ax.plot(time1[start:end], series1[start:end], label = label1)
  ax.plot(time2[start:end], series2[start:end], label = label2)
  plt.xlabel(xlabel)
  plt.ylabel(ylabel)
  plt.grid(True)
  if line ==True:
    plt.plot(np.zeros(len(time1)), color='black')
  legend = ax.legend(shadow=True, fontsize='x-large')
  title = ax.set_title(title)
  plt.show()
autocorr = sm.tsa.acf(df_train['y'],nlags=len(df_train))
fig3 = plot_series(np.arange(len(df_train['ds'])), autocorr, label1 = None,
            title='Company Autocorrelation between Daily Stock Prices',
            xlabel='Days Count',
            ylabel='Autocorrelation Value',
            line=True)
st.set_option('deprecation.showPyplotGlobalUse', False)
st.pyplot(fig3)
# Show and plot forecast
st.subheader('Forecast Data')
st.write(forecast.tail())
    
st.write(f'Forecast plot for {n_years} years')
fig1 = plot_plotly(m, forecast)
st.plotly_chart(fig1)

# st.write("Forecast components")
# fig2 = m.plot_components(forecast)
# st.write(fig2)

# I am fitting the data to get the correlation coefficents that will be plotted
