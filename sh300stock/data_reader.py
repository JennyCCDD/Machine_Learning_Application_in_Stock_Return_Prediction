#coding=utf-8

__author__ = "Mengxuan Chen"
__email__  = "chenmx19@mails.tsinghua.edu.cn"
__date__   = "20200102"

import pandas as pd
import datetime

class Para:
    path_data = '.\\input\\'
    path_results = '.\\output\\'
para = Para()

def get_price(end_date):
    open_price = pd.read_excel(para.path_data+'data.xlsx', sheet_name='open', index_col='Date', parse_dates=True)
    open_price = open_price.T.fillna(open_price.mean(axis=1)).T
    close = pd.read_excel(para.path_data+'data.xlsx', sheet_name='close', index_col='Date', parse_dates=True)
    close = close.T.fillna(close.mean(axis=1)).T
    low = pd.read_excel(para.path_data+'data.xlsx', sheet_name='low', index_col='Date', parse_dates=True)
    low = low.T.fillna(low.mean(axis=1)).T
    high = pd.read_excel(para.path_data+'data.xlsx', sheet_name='high', index_col='Date', parse_dates=True)
    high = high.T.fillna(high.mean(axis=1)).T
    avg_price = pd.read_excel(para.path_data+'data.xlsx', sheet_name='average', index_col='Date', parse_dates=True)
    avg_price = avg_price.T.fillna(avg_price.mean(axis=1)).T
    prev_close = pd.read_excel(para.path_data+'data.xlsx', sheet_name='prev-close', index_col='Date', parse_dates=True)
    prev_close = prev_close.T.fillna(prev_close.mean(axis=1)).T
    volume = pd.read_excel(para.path_data+'data.xlsx', sheet_name='volumn', index_col='Date', parse_dates=True)
    volume = volume.T.fillna(volume.mean(axis=1)).T
    amount = pd.read_excel(para.path_data+'data.xlsx', sheet_name='turnover', index_col='Date', parse_dates=True)
    amount = amount.T.fillna(amount.mean(axis=1)).T

    begin_date = (datetime.datetime.strptime(end_date, '%Y/%m/%d') - datetime.timedelta(days=50)).strftime(
        '%Y/%m/%d')

    price = {
        'open_price': open_price[begin_date:end_date],
        'high': high[begin_date:end_date],
        'low': low[begin_date:end_date],
        'close': close[begin_date:end_date],
        'avg_price': avg_price[begin_date:end_date],
        'prev_close': prev_close[begin_date:end_date],
        'volume': volume[begin_date:end_date],
        'amount': amount[begin_date:end_date]
    }
    print(price)
    panel = pd.Panel(price)
    return panel
