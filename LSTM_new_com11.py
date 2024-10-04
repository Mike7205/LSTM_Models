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
col1, col2 = st.columns([0.5, 0.5])
with col1: 
    st.title('LSTM Prediction Models + AI tips and hints')
with col2:
    st.image("T5_v1.jpeg", caption="The T5 model (Text-To-Text Transfer Transformer) was developed by the research team at Google Research. It is one of the advanced language models designed to handle various language tasks by transforming them into text-based tasks", width=100)

from transformers import T5Tokenizer, T5ForConditionalGeneration
import requests
from bs4 import BeautifulSoup

# Załaduj tokenizer i model T5
tokenizer = T5Tokenizer.from_pretrained('t5-small')
model = T5ForConditionalGeneration.from_pretrained('t5-small')

st.subheader('Market forecasts and news with T5 Model', divider ='blue')

# Wprowadzenie tematu przez użytkownika
query = st.text_input("Just ask a question:", "Brent Oil Forecast")

if st.button("Top 3 answers by T5:"):
    # Wyszukiwanie w Google
    search_url = f"https://www.google.com/search?q={query}"
    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(search_url, headers=headers)
    soup = BeautifulSoup(response.text, 'html.parser')

    # Pobierz wyniki wyszukiwania
    results = soup.find_all('div', class_='BNeawe s3v9rd AP7Wnd')
    summaries = []

    for result in results:
        input_text = f"summarize: {result.text}"
        input_ids = tokenizer.encode(input_text, return_tensors='pt')
        #outputs = model.generate(input_ids)
        outputs = model.generate(input_ids, max_length=150, num_beams=4, early_stopping=True)
        summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
                
        if summary not in summaries:
            summaries.append(summary)
        
        if len(summaries) == 3:
            break

    for summary in summaries:
        st.markdown(
                f"""
                <div style="background-color: #1f77b4; color: white; padding: 10px; border-radius: 5px;">
                    {summary}
                </div>
                """,
                unsafe_allow_html=True
            )
        
#st.html(
#    """
#<style>
#[data-testid="stSidebarContent"] {color: black; background-color: #91BFCF} #90EE90 #ADD8E6 #9CC2CF
#</style>
#""")

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

    
col3, col4 = st.columns([0.5, 0.5])
with col3:   
    st.write('\n')
    # Own LSTM EUR/PLN D+5 prediction model
    st.subheader('Own LSTM EUR/PLN D+5 prediction model', divider ='blue')  
    val_D5E = pd.read_pickle('D5_eur_tabel.pkl')
    val_D5EP = val_D5E[['Date','Day + 5 Prediction']][-100:]
    val_D5EU = pd.read_pickle('D1_EUR_a.pkl')
    val_D5EUR = val_D5EU[['Date','EUR/PLN']][-100:]
    day_es = val_D5EUR.shape[0]

    st.write(f'Predictions for the last {day_es} days')  
    fig_D5E = px.line(val_D5EP, x='Date', y=['Day + 5 Prediction'],color_discrete_map={'Day + 5 Prediction':'red'}, width=1500, height=500)
    fig_D5E.add_trace(go.Scatter(x=val_D5EUR['Date'], y=val_D5EUR['EUR/PLN'], mode='lines', name='EUR/PLN', line=dict(color='blue')))
    fig_D5E.update_layout(plot_bgcolor='white',showlegend=True,xaxis=dict(showgrid=True, gridwidth=0.5, gridcolor='Lightgrey'),
                          yaxis=dict(showgrid=True, gridwidth=0.5, gridcolor='Lightgrey'))
    #fig_D5E.add_vline(x = today,line_width=1, line_dash="dash", line_color="black")
    st.plotly_chart(fig_D5E)

    # Own EUR/PLN LSTM prediction model (D+1)    
    st.subheader('EUR/PLN exchange rate (D+1) predictions', divider ='blue')
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
    st.subheader('USD/PLN exchange rate (D+1) predictions', divider ='blue')
    val_s = pd.read_pickle('D1_USD_a.pkl')
    val_s1 = val_s[['Date','USD/PLN','Day + 1 Prediction']][-100:]      #.iloc[:-1]
    day_s1 = val_s1.shape[0]
    st.write(f'Predictions for the last {day_s1} days')
    fig_vals = px.line(val_s1, x='Date', y=['USD/PLN','Day + 1 Prediction'],color_discrete_map={
                     'USD/PLN':'#800080','Day + 1 Prediction':'green'}, width=1000, height=500 ) # #89CFF0
    fig_vals.update_layout(plot_bgcolor='white',showlegend=True,xaxis=dict(showgrid=True, gridwidth=0.5, gridcolor='Lightgrey'),
                          yaxis=dict(showgrid=True, gridwidth=0.5, gridcolor='Lightgrey'))   
    st.plotly_chart(fig_vals, use_container_width=True)

with col4:
    st.write('\n')
    st.subheader('Arima benchmark' , divider ='blue')
    col5, col6 = st.columns([0.5, 0.5])
    with col5:
        bench = st.radio('Benchmark for:', list(bench_dict.values()), horizontal=True, key="<co2>")
        
    with col6:
        size_a = st.slider('Forecast length', 1, 10, 1, key="<co1>")
        
    
    submit = st.button(f"Submit {bench} Benchmark forecast")
    if submit:
        curr_f(bench)
        Arima_f(bench, size_a)
        arima_chart_dff = pd.read_pickle('arima_chart_dff.pkl')
        fig_ar = px.line(arima_chart_dff, x='Date', y=['High', 'Close', 'Predicted Close'], color_discrete_map={
                              'High': 'orange', 'Close': 'black', 'Predicted Close': 'red'}, width=1000, height=500)
        fig_ar.add_vline(x = today,line_width=3, line_dash="dash", line_color="green")
        fig_ar.update_layout(xaxis=None, yaxis=None)
        st.plotly_chart(fig_ar, use_container_width=True)
        
