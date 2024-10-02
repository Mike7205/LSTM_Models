# To jest skrypt do strony z Benchmarkami
import streamlit as st
import pandas as pd
import numpy as np
import datetime as dt
from datetime import datetime, timedelta, date
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
import appdirs as ad
import pickle
import appdirs as ad
import yfinance as yf
from sklearn.linear_model import LinearRegression
from streamlit import set_page_config
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.arima.model import ARIMA

st.set_page_config(layout="wide")

# start definicji strony
st.title('LSTM Prediction Models')

st.html(
    """
<style>
[data-testid="stSidebarContent"] {color: black; background-color: #91BFCF} #90EE90 #ADD8E6 #9CC2CF
</style>
""")

today = date.today()
bench_dict = {'EURPLN=X':'EUR/PLN','PLN=X':'PLN/USD'}

# Pobieranie danych
def curr_f(bench):
    global df_c1
    #bench = 'EUR/PLN'
    for label, name in bench_dict.items():
        if name == bench:
            df_c = pd.DataFrame(yf.download(f'{label}', start='2000-09-01', end = today,interval='1d'))
            df_c1 = df_c.reset_index()
           
    return df_c1   

def Arima_f(bench, size_a):
    data = np.asarray(df_c1['Close'][-300:]).reshape(-1, 1)
    p = 10
    d = 0
    q = 5
    n = size_a

    model = ARIMA(data, order=(p, d, q))
    model_fit = model.fit(method_kwargs={'maxiter': 3000})
    model_fit = model.fit(method_kwargs={'xtol': 1e-6})
    fore_arima = model_fit.forecast(steps=n)  
    
    arima_dates = [datetime.today() + timedelta(days=i) for i in range(0, size_a)]
    arima_pred_df = pd.DataFrame({'Date': arima_dates, 'Predicted Close': fore_arima})
    arima_pred_df['Date'] = arima_pred_df['Date'].dt.strftime('%Y-%m-%d')
    arima_df = pd.DataFrame(df_c1[['Date','High','Close']][-500:])
    arima_df['Date'] = arima_df['Date'].dt.strftime('%Y-%m-%d')
    arima_chart_df = pd.concat([arima_df, arima_pred_df], ignore_index=True)
    x_ar = (list(arima_chart_df.index)[-1] + 1)
    arima_chart_dff = arima_chart_df.iloc[x_ar - 30:x_ar]
    arima_chart_dff.to_pickle('arima_chart_dff.pkl') 
    
def Arima_chart():    
    arima_chart_dff = pd.read_pickle('arima_chart_dff.pkl')
    fig_ar = px.line(arima_chart_dff, x='Date', y=['High', 'Close', 'Predicted Close'], color_discrete_map={
                      'High': 'yellow', 'Close': 'black', 'Predicted Close': 'red'}, width=1000, height=500)
    fig_ar.add_vline(x = today,line_width=3, line_dash="dash", line_color="green")
    fig_ar.update_layout(xaxis=None, yaxis=None)
    st.plotly_chart(fig_ar, use_container_width=True)
       
# Own LSTM EUR/PLN D+5 prediction model
st.subheader('Own LSTM EUR/PLN D+5 prediction model')  
val_D5E = pd.read_pickle('D5_eur_tabel.pkl')
val_D5EP = val_D5E[['Date','Day + 5 Prediction']][-100:]
val_D5EU = pd.read_pickle('D1_EUR_a.pkl')
val_D5EUR = val_D5EU[['Date','EUR/PLN']][-100:]
day_es = val_D5EUR.shape[0]

st.write(f'Predictions for the last {day_es} days')  
fig_D5E = px.line(val_D5EP, x='Date', y=['Day + 5 Prediction'],color_discrete_map={'Day + 5 Prediction':'red'}, width=1000, height=500)
fig_D5E.add_trace(go.Scatter(x=val_D5EUR['Date'], y=val_D5EUR['EUR/PLN'], mode='lines', name='EUR/PLN', line=dict(color='blue')))
fig_D5E.update_layout(plot_bgcolor='white',showlegend=True,xaxis=dict(showgrid=True, gridwidth=0.5, gridcolor='Lightgrey'),
                      yaxis=dict(showgrid=True, gridwidth=0.5, gridcolor='Lightgrey'))
fig_D5E.add_vline(x = today,line_width=1, line_dash="dash", line_color="black")
st.plotly_chart(fig_D5E)

# Own EUR/PLN LSTM prediction model (D+1)    
st.subheader('EUR/PLN exchange rate (D+1) predictions')
val = pd.read_pickle('D1_EUR_a.pkl')
val_1 = val[['Date','EUR/PLN','Day + 1 Prediction']][-100:]      #.iloc[:-1]
day_s = val_1.shape[0]
st.write(f'Predictions for the last {day_s} days')
fig_val = px.line(val_1, x='Date', y=['EUR/PLN','Day + 1 Prediction'],color_discrete_map={
                 'EUR/PLN':'blue','Day + 1 Prediction':'red'}, width=1000, height=500 ) 
fig_val.update_layout(plot_bgcolor='white',showlegend=True,xaxis=dict(showgrid=True, gridwidth=0.5, gridcolor='Lightgrey'),
                      yaxis=dict(showgrid=True, gridwidth=0.5, gridcolor='Lightgrey'))   
st.plotly_chart(fig_val, use_container_width=True)

# Own USD/PLN LSTM prediction model (D+1)     
st.subheader('USD/PLN exchange rate (D+1) predictions')
val_s = pd.read_pickle('D1_USD_a.pkl')
val_s1 = val_s[['Date','USD/PLN','Day + 1 Prediction']][-100:]      #.iloc[:-1]
day_s1 = val_s1.shape[0]
st.write(f'Predictions for the last {day_s1} days')
fig_vals = px.line(val_s1, x='Date', y=['USD/PLN','Day + 1 Prediction'],color_discrete_map={
                 'USD/PLN':'#800080','Day + 1 Prediction':'green'}, width=1000, height=500 ) # #89CFF0
fig_vals.update_layout(plot_bgcolor='white',showlegend=True,xaxis=dict(showgrid=True, gridwidth=0.5, gridcolor='Lightgrey'),
                      yaxis=dict(showgrid=True, gridwidth=0.5, gridcolor='Lightgrey'))   
st.plotly_chart(fig_vals, use_container_width=True)

# Definicja zakłądki bocznej
st.sidebar.title('Benchamrk models list')
bench = st.sidebar.radio('Benchmark for:', list(bench_dict.values()))
curr_f(bench)
size_a = st.sidebar.slider('Forecast length', 1, 10, 1, key = "<co1>")
Arima_f(bench, size_a)
Arima_chart()