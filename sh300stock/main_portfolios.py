#coding=utf-8

__author__ = "Mengxuan Chen"
__email__  = "chenmx19@mails.tsinghua.edu.cn"
__date__   = "20200102"


#--* pakages*--
import os
import pandas as pd
import numpy as np
import math
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
from portfolio_functions import performance,performance_anl,portfolio_EW
from sklearn import preprocessing
from matplotlib import pyplot as plt
from pylab import rcParams
from keras import models
from keras import layers
from keras.optimizers import *

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

class Para:
    path_data = '.\\input\\'
    path_results = '.\\output\\'
    rolling_time = 21
para = Para()

class main():
    def __init__(self):
        return
    def lstm(self,i):
        # ## read data
        df = pd.read_csv(list[i])
        for t in range(df.shape[0]):
            for s in range(df.shape[1]):
                if str(df.iat[t,s])=='inf':
                    df.iat[t,s]=0
                if str(df.iat[t,s])=='#NAME?':
                    df.iat[t,s]=0
                if str(df.iat[t,s])=='-inf':
                    df.iat[t, s] = 0
        #print(df.shape)
        df.fillna(df.mean(axis=0), inplace=True)
        print(df.shape)
        #df.head()
        # 有2601个交易日，165-11=154个features

        for j in range(2):
            train = df[250*j:442+250*j]#[250:692]  # [0:442]
            cv = df[442+250*j:692+250*j]#[692:942]  # [442:692]
            test = df[692+250*j:942+250*j]#[942:1129]  # [692:942]

            # In[ ]:

            dff = pd.read_csv('shnew112.csv')
            indica = dff.columns.tolist()
            features_name = indica[1:]
            print(features_name)
            target_name = ['return']

            X_train = train[features_name]
            #where_are_inf = np.isinf(X_train)
            #X_train[where_are_inf] = 0
            minmax = preprocessing.MinMaxScaler()
            X_train = minmax.fit_transform(X_train).copy()
            y_train = train[target_name]
            y_train.dropna(inplace=True)

            X_cv = cv[features_name]
            #where_are_inf = np.isinf(X_cv)
            #X_cv[where_are_inf] = 0
            y_cv = cv[target_name]
            X_cv = minmax.fit_transform(X_cv).copy()
            y_cv.dropna(inplace=True)

            X_test = test[features_name]
            y_test = test[target_name]
            #where_are_inf = np.isinf(X_test)
            #X_test[where_are_inf] = 0
            X_test = minmax.fit_transform(X_test).copy()
            y_test.dropna(inplace=True)

            # ## common functions
            def get_mape(y_true, y_pred):
                y_true, y_pred = np.array(y_true), np.array(y_pred)
                return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


            def linear_model_fit(model, model_name):
                est = model.predict(X_train)

                # Calculate RMSE
                print(stockno[i] + '_' + '%s' % model_name + "_in-sample MSE = " + str(mean_squared_error(y_train, est)))
                print(stockno[i] + '_' + '%s' % model_name + "_in-sample RMSE = " + str(
                    math.sqrt(mean_squared_error(y_train, est))))
                print(stockno[i] + '_' + '%s' % model_name + "_in-sample MAPE = " + str(get_mape(y_train, est)))
                print(stockno[i] + '_' + '%s' % model_name + "_in-sample R2 = " + str(r2_score(y_train, est)))

                # Plot adjusted close over time
                rcParams['figure.figsize'] = 10, 8  # width 10, height 8

                est_df = pd.DataFrame({'est': est.T.tolist()[0],
                                       'Date': train['Date']})

                ax = train.plot(x='Date', y='return', style='b-', grid=True)
                ax = cv.plot(x='Date', y='return', style='y-', grid=True, ax=ax)
                # ax = test.plot(x='Date', y='return', style='g-', grid=True, ax=ax)
                ax = est_df.plot(x='Date', y='est', style='r-', grid=True, ax=ax)
                ax.legend(['train', 'dev', 'test', 'est'])
                ax.set_xlabel("Date")
                ax.set_ylabel("%")
                plt.savefig(para.path_results +  stockno[i] + '_'+str(j)+ '_'+ 'in_sample_result_%s.png' % model_name)
                # plt.show()
                # Do prediction on test set
                est_ = model.predict(X_test)
                #

                # Calculate RMSE
                print(
                    stockno[i] + '_' + '%s' % model_name + "_out-of-sample MSE = " + str(mean_squared_error(y_test, est_)))
                print(stockno[i] + '_' + '%s' % model_name + "_out-of-sample RMSE = " + str(
                    math.sqrt(mean_squared_error(y_test, est_))))
                print(stockno[i] + '_' + '%s' % model_name + "_out-of-sample MAPE = " + str(get_mape(y_test, est_)))
                print(stockno[i] + '_' + '%s' % model_name + "_out-of-sample R2 = " + str(r2_score(y_test, est_)))
                # Plot adjusted close over time
                rcParams['figure.figsize'] = 10, 8  # width 10, height 8
                #matplotlib.rcParams.update({'font.size': 14})

                est_df_ = pd.DataFrame({'est': est_.T.tolist()[0],
                                        'Date': test['Date']})

                est_df_.to_csv(para.path_result + 'final1\\'  + stockno[i] + '_'+str(j)+ '_'+ 'out_of_sample_result_%s.csv' % model_name,
                                    index=False)
                print(est_df_)

                # ax = train.plot(x='Date', y='return', style='b-', grid=True)
                # ax = cv.plot(x='Date', y='return', style='y-', grid=True, ax=ax)
                ax = test.plot(x='Date', y='return', style='g-', grid=True)
                ax = est_df_.plot(x='Date', y='est', style='r-', grid=True, ax=ax)
                ax.legend(['test', 'predictions'])
                ax.set_xlabel("Date")
                ax.set_ylabel("%")
                plt.savefig(para.path_results +  stockno[i] + '_'+str(j)+ '_'+ 'out_of_sample_result_%s.png' % model_name)

                #plt.show()
                return est_df, est_df_


            # reshape input to be 3D [samples, timesteps, features]
            X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
            X_test = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))
            print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)


            # design network
            model = models.Sequential()
            model.add(layers.LSTM(50, input_shape=(X_train.shape[1], X_train.shape[2])))
            model.add(layers.Dense(1))
            model.compile(loss='mse', optimizer='adam')
            # fit network
            history = model.fit(X_train, y_train,
                                epochs=40, batch_size=72,
                                validation_data=(X_test, y_test), verbose=2, shuffle=False)
            score = model.evaluate(X_test, y_test, verbose=2)
            print("\nScore MSE:", score)
            # plot history
            plt.plot(history.history['loss'], label='train')
            plt.plot(history.history['val_loss'], label='test')
            plt.legend()
            plt.savefig(para.path_results + stockno[i] + '_'+str(j)+ '_' + 'LSTM_MSE_history.png')
            # plt.show()
            LSTM_MAE_model_fit = linear_model_fit(model, 'LSTM_MSE')

if __name__ == '__main__':
    csvstock = pd.read_csv(para.path_data + 'stockno.csv', header=None)
    stockno = np.array(csvstock)  # 转换为 ndarray [[1], [2], [3]]
    stockno = stockno.reshape(1, len(stockno)).tolist()
    path = para.path_data+'result\\'


    def get_file():
        files =os.listdir(path)
        files.sort() #排序
        list= []
        for file in files:
            f_name = str(file)
            filename = path + f_name
            list.append(filename)
        return list
    list = get_file()


    csvstock = pd.read_csv('stockno.csv', header=None)
    stockno = np.array(csvstock)
    stockno = stockno.reshape(1, len(stockno)).tolist()
    stockno = stockno[0]

    for i in range(300):
        main().lstm(i)
    ###############################################
    def get_file():
        files = os.listdir(para.path_data+'final\\')
        files.sort()
        list = []
        for file in files:
            f_name = str(file)
            filename = para.path_data + 'final\\' + f_name
            list.append(filename)
        return list
    list = get_file()

    timesteps = pd.read_csv(list[0],index_col=0, parse_dates=True)
    timesteps= timesteps.index.tolist()

    for timestep in range(0,int(500/para.rolling_time)*para.rolling_time,para.rolling_time):
        single_time = pd.DataFrame()
        for stock in range(len(list)):
            single_stock = pd.read_csv(list[stock],index_col=0, parse_dates=True)
            single_stock['pre_month_close']=single_stock['close'].shift(para.rolling_time)
            single_stock['return'] =((single_stock['close']-single_stock['pre_month_close'])/\
                                     single_stock['pre_month_close'])
            single_stock['prediction'] = pow(single_stock['est']+1,para.rolling_time)-1
            single_stock.dropna(inplace=True)
            single_stock_time = single_stock.iloc[timestep,:]
            single_time = pd.concat([single_time,single_stock_time],axis = 1)
            print(single_time)
            single_time.to_csv(para.path_results+'%s'%(timestep+1)+'.csv')

    RR_EW = []
    RR_VW = []
    RR_EW_Long = []
    RR_EW_Short = []
    RR_VW_Long = []
    RR_VW_Short = []
    for timestep in range(0,int(500/para.rolling_time)*para.rolling_time,para.rolling_time):
        single = pd.read_csv(para.path_results+'final\\'+'%s'%(timestep+1)+'.csv', index_col=0, parse_dates=True)
        single_T = single.T
        print(single_T)
        total_sort_, long_ew, short_ew, long_short_ew = portfolio_EW(single_T,0.1)
        RR_EW_Long.append(long_ew)
        RR_EW_Short.append(short_ew)
        RR_EW.append(long_short_ew)
    # main output part
    strategy_RR_EW_Long = pd.DataFrame(RR_EW_Long, columns=['monthly'])
    strategy_RR_EW_Long.index = pd.DatetimeIndex(strategy_RR_EW_Long.index)
    strategy_RR_EW_Long['nav'] = (strategy_RR_EW_Long['monthly'] + 1).cumprod()
    strategy_RR_EW_Long.to_csv(para.path_results + 'strategy_RR_EW_Long_.csv')
    print(strategy_RR_EW_Long)

    print('__________________strategy_RR_EW_Long__________________')
    performance(strategy_RR_EW_Long)
    performance_anl(strategy_RR_EW_Long)

    strategy_RR_EW_Short = pd.DataFrame(RR_EW_Short, columns=['monthly'])
    strategy_RR_EW_Short.index = pd.DatetimeIndex(strategy_RR_EW_Short.index)
    strategy_RR_EW_Short['nav'] = (strategy_RR_EW_Short['monthly'] + 1).cumprod()
    strategy_RR_EW_Short.to_csv(para.path_results + 'strategy_RR_EW_Short_.csv')
    print('__________________strategy_RR_EW_Short__________________')
    performance(strategy_RR_EW_Short)
    performance_anl(strategy_RR_EW_Short)

    strategy_RR_EW = pd.DataFrame(RR_EW, columns=['monthly'])
    strategy_RR_EW.index = pd.DatetimeIndex(strategy_RR_EW.index)
    strategy_RR_EW['nav'] = (strategy_RR_EW['monthly'] + 1).cumprod()
    strategy_RR_EW.to_csv(para.path_results + 'strategy_RR_EW_.csv')
    print('__________________strategy_RR_EW__________________')
    performance(strategy_RR_EW)
    performance_anl(strategy_RR_EW)

    strategy_RR_VW_Long = pd.DataFrame(RR_VW_Long, columns=['monthly'])
    strategy_RR_VW_Long.index = pd.DatetimeIndex(strategy_RR_VW_Long.index)
    strategy_RR_VW_Long['nav'] = (strategy_RR_VW_Long['monthly'] + 1).cumprod()
    strategy_RR_VW_Long.to_csv(para.path_results + 'strategy_RR_VW_Long_.csv')