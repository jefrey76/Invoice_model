import streamlit as st
import re
import pandas as pd
from datetime import datetime
from invoice_analyser import read_config


def get_data_transformer_function(func_key: str, ai_response: str):

    function_dict = {
        "get_date_details": get_date_details,
        "get_invoice_details": get_invoice_details,
        "all_items": get_all_items,
        "chart": prepare_bar_chart
        }
    function_dict[func_key](ai_response)


def get_date_details(data: str):

    df_date: list = []
    df_invoice: list = []
    df_delay: list = []
    today = datetime.now()
    invoice_dict: dict = eval(data)

    for key, value in invoice_dict.items():
        try:
            date_diff = today - datetime.strptime(value, "%d-%m-%Y")
        except:
            date_diff = today - datetime.strptime(value, "%Y-%m-%d")
        if int(date_diff.days) > 0:
            df_invoice.append(key)
            df_date.append(value)
            df_delay.append(int(date_diff.days))

    if not df_date:
        st.write("There is nothing to display as it seems there are no delays")
    else:
        df = pd.DataFrame(
            {
                "invoice": df_invoice,
                "date": df_date,
                "delay": df_delay
            }
        )
        st.header(f'Order delay verified at: {datetime.strftime(datetime.now(),"%d-%m-%Y")}',divider=True)
        st.dataframe(
            df,
            width=1200,
            column_config={
                "invoice": "Invoice Id",
                "date": "Order Date",
                "delay": "Overdue (days)"
            },
            hide_index = True
        )
        col1, col2 = st.columns(spec=[0.4, 0.6])
        label = configure_parameters_label()
        with col1:
            with st.expander("See model settings", expanded=False):
                st.text(body=label)
            
                     
def get_invoice_details(data: str):

    st.header("Invoice details generated by AI:", divider=True)
    st.write(data)
    col1, col2 = st.columns(spec=[0.4, 0.6])
    label = configure_parameters_label()
    with col1:
        with st.expander("See model settings", expanded=False):
            st.text(body=label)


def get_all_items(data: str):

    st.header("Item details generated by AI:", divider=True)
    st.write(data)
    col1, col2 = st.columns(spec=[0.4, 0.6])
    label = configure_parameters_label()
    with col1:
        with st.expander("See model settings", expanded=False):
            st.text(body=label)


def prepare_bar_chart(data: str):

    st.write(data)
    col1, col2 = st.columns(spec=[0.4, 0.6])
    label = configure_parameters_label()
    with col1:
        with st.expander("See model settings", expanded=False):
            st.text(body=label)


def configure_parameters_label()->st:

    config = read_config()
    label_string = f'Model: {config['model']}\nTemperature: {config['temperature']}\nChunk size: {config['chunk_size']}\nOverlap: {config['chunk_overlap']}'
    return label_string