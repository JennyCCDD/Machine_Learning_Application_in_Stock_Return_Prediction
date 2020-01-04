#coding=utf-8
# python35 pandas0.21.0 for alpha generating part

__author__ = "Mengxuan Chen"
__email__  = "chenmx19@mails.tsinghua.edu.cn"
__date__   = "20200102"

#--* pakages*--
####### normal pakages
import pandas as pd
import numpy as np
import scipy as sp
from scipy.stats import linregress
from matplotlib import pyplot as plt
from pylab import rcParams
####### something from sklearn
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.feature_selection import RFE,RFECV
from matplotlib import pyplot
from sklearn import preprocessing
from sklearn.feature_selection import VarianceThreshold
#sel = VarianceThreshold(threshold=(.8 * (1 - .8)))
#sel.fit_transform(X)
from sklearn import linear_model,kernel_ridge
from sklearn.ensemble import GradientBoostingRegressor, \
    RandomForestRegressor, ExtraTreesRegressor,AdaBoostRegressor
from sklearn import svm
from sklearn import neighbors
from sklearn import gaussian_process
from sklearn import tree
from sklearn import neural_network
from sklearn.metrics import mean_squared_error, r2_score
####### pakages for other models
import xgboost as xgb
import lightgbm as lgb
####### pakages for Deep Learning models
from keras import models
from keras import layers
from keras.models import Sequential
from keras.layers import Convolution1D, MaxPooling1D, Flatten
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import *
####### functions to evaluate the models
from functions import generate_a_line,data_preprocess,get_mape, \
    linear_model_fit,linear_model_fit_no_feature_importance,model_fit,linear_model_fit_dl

#--* parameters*--
class Para:
    path_data = '.\\input\\'
    path_results = '.\\output\\'
    rolling_time = 21
para = Para()

#--* main function*--
class main():
    def __init__(self):
        pass
    def shalpha_function(self):
        # Attention!!!!!!!!!!!!
        # python35 pandas0.21.0 for alpha generating part
        orginal_data = pd.read_csv(para.path_data + 'sh.csv')
        for i in range(len(orginal_data)):
            dataframe = pd.DataFrame()
            print(i)
            handled = generate_a_line(52 + i)
            self.shalpha = pd.concat([dataframe, handled], axis=0)
        # if you have already get the alpha data, just run this line below
        # self.shalpha = pd.read_csv(para.path_data + 'sh112.csv', index=None)
        return self.shalpha

    def shalphas_function(self):
        # construct flat time series features(21 work days* 112 alphas)
        # load the 112 alpha data
        shalpha = pd.read_csv(para.path_data + 'sh112.csv')
        df = pd.DataFrame(shalpha)
        datajoint = df
        for i in range(para.rolling_time-1):
            dfe = df.shift(i)
            datajoint = pd.concat([datajoint, dfe], axis=1)

        for j in range(para.rolling_time-2):
            datajoint = datajoint.drop(j)

        datelist = datajoint['Date']
        datelist = datelist.iloc[:, 0]
        datajoint = datajoint.drop('Date', axis=1, index=None)

        rename = pd.read_csv(para.path_data + 'rename.csv', header=None)
        dff = pd.DataFrame(rename)
        dfa = np.array(dff)
        dfa = dfa.reshape(1, len(dfa)).tolist()
        dfa = dfa[0]
        datajoint.columns = dfa
        datajointt = pd.concat([datelist, datajoint], axis=1)

        return datajointt

    def feaature_selection_df(self):
        # load the 112*21t alpha data
        datajointt = pd.read_csv(para.path_data + 'sh112_21t.csv', low_memory=False)
        datajoint_fs = datajointt.fillna(0)
        csv_return = pd.read_csv(para.path_data+'sh.csv', low_memory=False)
        total = pd.merge(datajoint_fs, csv_return)
        total['return'] = total['close'].pct_change()
        total.fillna(0, inplace=True)
        total.drop(['amount', 'pre_close', 'avg_price', 'turn', 'volume', 'close', 'low', 'high', 'open'], axis=1,
                   inplace=True)
        rename = pd.read_csv(para.path_data + 'rename.csv', header=None)
        features = np.array(pd.DataFrame(rename).T).tolist()

        # split into input and output
        X = total.drop(columns=['return', 'Date'])
        y = total['return']

        # fit gbdt model
        model = RFE(estimator=GradientBoostingRegressor(), n_features_to_select=300, step=0.5)
        model.fit(X, y)
        coef = pd.DataFrame(model.support_, index=features, columns=['fs_results'])
        coef.sort_index(ascending=False, inplace=True)
        coef = pd.DataFrame(coef)
        coef.to_csv(para.path_data + "ftselected300.csv")

        ftseleted = []

        for i in coef:
            if i[1] == False:
                ftseleted.append(i[0])
        for j in ftseleted:
            datajoint_fs = datajoint_fs.drop(columns=[j])

        return datajoint_fs


if __name__ == '__main__':
    # Attention!!!!!!!!!!!!
    # python35 pandas0.21.0 for alpha generating part
    sh112_df = main().shalpha_function()
    sh112_df.to_csv(para.path_data + 'sh112.csv', index=None)
    # if you have already get the alpha data, just run this line below
    #sh112_df = pd.read_csv(para.path_data+'sh112.csv')
    ########################################################
    # construct flat time series features(21 work days* 112 alphas)
    datajointt_= main().shalphas_function()
    datajointt_.to_csv(para.path_data+'sh112_21t.csv',index=False)
    ########################################################
    # feature selection
    datajoint_fs_ = main().feaature_selection_df()
    datajoint_fs_.to_csv(para.path_data + "sh112_21t_after_fs.csv", index=False)
    ########################################################
    # split into train, cv test
    datajointt_=pd.read_csv(para.path_data+'sh112_21t.csv')
    csv_return = pd.read_csv(para.path_data + 'sh.csv', low_memory=False)
    total = pd.merge(datajointt_, csv_return)
    X_train, y_train,X_cv,y_cv,X_test, y_test, features_name,train,cv,test = data_preprocess(total)
    ########################################################
    # # Timing Models without feature selection
    # ## Linear Regression
    linreg = linear_model.LinearRegression()
    linreg_model_fit = linear_model_fit(linreg, 'Linear_Regression',X_train, y_train,X_cv,y_cv,X_test, y_test, features_name,train,cv,test)

    # ## ridge regression
    ridgereg = linear_model.Ridge()
    ridgereg_model_fit = linear_model_fit(ridgereg, 'Rdige_Regression',X_train, y_train,X_cv,y_cv,X_test, y_test, features_name,train,cv,test)

    # ## Orthognoal Matching Pursuit Regression
    n_nonzero_coefs = 17
    ompreg = linear_model.OrthogonalMatchingPursuit(n_nonzero_coefs=n_nonzero_coefs)
    ompreg_model_fit = model_fit(ompreg, 'Orthognoal_Matching_Pursuit_Regression',X_train, y_train,X_cv,y_cv,X_test, y_test, features_name,train,cv,test)

    coef = pd.DataFrame(ompreg.coef_, index=features_name, columns=['features_importance'])
    coef.sort_index(ascending=False, inplace=True)
    print(coef.head(10).round(6))
    coef.to_csv(para.path_results + "features_importance_ompreg.csv")

    # ## Bayesian Ridge Regression
    Bayesreg = linear_model.BayesianRidge()
    Bayesreg_model_fit = model_fit(Bayesreg, 'Bayesian_Ridge_Regression',X_train, y_train,X_cv,y_cv,X_test, y_test, features_name,train,cv,test)
    coef = pd.DataFrame(Bayesreg.coef_, index=features_name, columns=['features_importance'])
    coef.sort_index(ascending=False, inplace=True)
    print(coef.head(10).round(6))
    coef.to_csv(para.path_results + "features_importance_Bayesreg.csv")

    # ## ARD Regression

    ardreg = linear_model.ARDRegression()
    ardreg_model_fit = model_fit(ardreg, 'ARD_Regression',X_train, y_train,X_cv,y_cv,X_test, y_test, features_name,train,cv,test)

    coef = pd.DataFrame(ardreg.coef_, index=features_name, columns=['features_importance'])
    coef.sort_index(ascending=False, inplace=True)
    print(coef.head(10).round(6))
    coef.to_csv(para.path_results + "features_importance_ardreg.csv")

    # ## TheilSen Regression

    theilsenreg = linear_model.TheilSenRegressor()
    theilsenreg_model_fit = model_fit(theilsenreg, 'TheilSen_Regression',X_train, y_train,X_cv,y_cv,X_test, y_test, features_name,train,cv,test)

    coef = pd.DataFrame(theilsenreg.coef_, index=features_name, columns=['features_importance'])
    coef.sort_index(ascending=False, inplace=True)
    print(coef.head(10).round(3))
    coef.to_csv(para.path_results + "features_importance_theilsenreg.csv")

    # ## Decision Tree Regression
    treereg = tree.DecisionTreeRegressor()
    treereg_model_fit = model_fit(treereg, 'Decision Tree Regression',X_train, y_train,X_cv,y_cv,X_test, y_test, features_name,train,cv,test)
    coef = pd.DataFrame(treereg.feature_importances_, index=features_name, columns=['features_importance'])
    coef.sort_index(ascending=False, inplace=True)
    print(coef.head(10).round(3))
    coef.to_csv(para.path_results + "features_importance_treereg.csv")

    # ## Random Forest Regression
    rfref = RandomForestRegressor()
    rfref_model_fit = model_fit(rfref, 'Random Forest Regression',X_train, y_train,X_cv,y_cv,X_test, y_test, features_name,train,cv,test)
    coef = pd.DataFrame(rfref.feature_importances_, index=features_name, columns=['features_importance'])
    coef.sort_index(ascending=False, inplace=True)
    print(coef.head(10).round(3))
    coef.to_csv(para.path_results + "features_importance_rfref.csv")

    # ## AdoBoost Regression
    adgbreg = AdaBoostRegressor()
    adgbreg_model_fit = model_fit(adgbreg, 'AdoBoost Regression',X_train, y_train,X_cv,y_cv,X_test, y_test, features_name,train,cv,test)
    coef = pd.DataFrame(adgbreg.feature_importances_, index=features_name, columns=['features_importance'])
    coef.sort_index(ascending=False, inplace=True)
    print(coef.head(10).round(3))
    coef.to_csv(para.path_results + "features_importance_adgbreg.csv")

    # ## Extra Tree Regression
    extreg = ExtraTreesRegressor()
    extreg_model_fit = model_fit(extreg, 'Extra_Tree_Regression',X_train, y_train,X_cv,y_cv,X_test, y_test, features_name,train,cv,test)
    coef = pd.DataFrame(extreg.feature_importances_, index=features_name, columns=['features_importance'])
    coef.sort_index(ascending=False, inplace=True)
    print(coef.head(10).round(3))
    coef.to_csv(para.path_results + "features_importance_extreg.csv")

    # ## GBRT
    gbdtreg = GradientBoostingRegressor()
    gbdtreg_model_fit = model_fit(gbdtreg, 'GBRT',X_train, y_train,X_cv,y_cv,X_test, y_test, features_name,train,cv,test)
    coef = pd.DataFrame(gbdtreg.feature_importances_, index=features_name, columns=['features_importance'])
    coef.sort_index(ascending=False, inplace=True)
    print(coef.head(10).round(3))
    coef.to_csv(para.path_results + "features_importance_gbdtreg.csv")

    # ## XGBoost
    model_seed = 100
    n_estimators = 100
    max_depth = 3
    learning_rate = 0.1
    min_child_weight = 1

    # Create the model
    xgbreg = xgb.XGBRegressor(seed=model_seed,
                              n_estimators=n_estimators,
                              max_depth=max_depth,
                              learning_rate=learning_rate,
                              min_child_weight=min_child_weight)

    xgbreg_model_fit = model_fit(xgbreg, 'XGBoost',X_train, y_train,X_cv,y_cv,X_test, y_test, features_name,train,cv,test)
    coef = pd.DataFrame(xgbreg.feature_importances_, index=features_name, columns=['features_importance'])
    coef.sort_index(ascending=False, inplace=True)
    print(coef.head(10).round(3))
    coef.to_csv(para.path_results + "features_importance_xgbreg.csv")

    # ## lightBGM
    lgbreg = lgb.LGBMRegressor(silent=False)
    lgbreg_model_fit = model_fit(lgbreg, 'lightBGM',X_train, y_train,X_cv,y_cv,X_test, y_test, features_name,train,cv,test)
    coef = pd.DataFrame(lgbreg.feature_importances_, index=features_name, columns=['features_importance'])
    coef.sort_index(ascending=False, inplace=True)
    print(coef.head(10).round(3))
    coef.to_csv(para.path_results + "features_importance_lgbreg.csv")
    ########################################################
    
    model = Sequential()
    model.add(Dense(500, input_dim= X_train.shape[1]))
    model.add(Activation('relu'))
    model.add(Dropout(0.25))
    model.add(Dense(250))
    model.add(Activation('relu'))
    model.add(Dense(1))
    model.add(Activation('linear'))
    model.compile(optimizer='adam',
                  loss='mse')
    # fit network
    history = model.fit(X_train,
              y_train,
              epochs=40,
              batch_size = 72,
              verbose=2,
              validation_data=(X_test, y_test))
    score = model.evaluate(X_test, y_test, verbose=2)
    print("\nScore MSE:",score)
    # plot history
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='test')
    plt.legend()
    plt.savefig(para.path_results+'MLP_MSE_history.jpg')
    plt.show()
    MLP_MSE_model_fit = linear_model_fit_dl(model, 'MLP_MSE',X_train, y_train,X_cv,y_cv,X_test, y_test, features_name,train,cv,test)


    # ### loss = MAE
    model = Sequential()
    model.add(Dense(500, input_dim= X_train.shape[1]))
    model.add(Activation('relu'))
    model.add(Dropout(0.25))
    model.add(Dense(250))
    model.add(Activation('relu'))
    model.add(Dense(1))
    model.add(Activation('linear'))
    model.compile(optimizer='adam',
                  loss='mae')
    # fit network
    history = model.fit(X_train,
              y_train,
              epochs=40,
              batch_size = 72,
              verbose=2,
              validation_data=(X_test, y_test))
    score = model.evaluate(X_test, y_test, verbose=2)
    print("\nScore MAE:",score)
    # plot history
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='test')
    plt.legend()
    plt.savefig(para.path_results+'MLP_MAE_history.jpg')
    plt.show()
    MLP_MAE_model_fit = linear_model_fit_dl(model, 'MLP_MAE',X_train, y_train,X_cv,y_cv,X_test, y_test, features_name,train,cv,test)

    
    # ## LSTM
    # ### loss = MSE
    # reshape input to be 3D [samples, timesteps, features]
    X_train_LSTM = X_train.reshape((X_train.shape[0], 1,X_train.shape[1]))
    X_cv_LSTM = X_cv.reshape((X_cv.shape[0], 1,X_cv.shape[1]))
    X_test_LSTM = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))
    print(X_train_LSTM.shape, y_train.shape, X_test_LSTM.shape, y_test.shape)
    # design network
    model = models.Sequential()
    model.add(layers.LSTM(50, input_shape=(X_train_LSTM.shape[1], X_train_LSTM.shape[2])))
    model.add(layers.Dense(1))
    model.compile(loss='mse', optimizer='adam')
    # fit network
    history = model.fit(X_train_LSTM, y_train,
                        epochs=40, batch_size=72,
                        validation_data=(X_cv_LSTM, y_cv), verbose=2, shuffle=False)
    score = model.evaluate(X_test_LSTM, y_test, verbose=2)
    print("\nScore MSE:",score)
    # plot history
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='test')
    plt.legend()
    plt.savefig(para.path_results+'LSTM_MSE_history.jpg')
    plt.show()
    LSTM_MAE_model_fit = linear_model_fit_dl(model, 'LSTM_MSE',X_train_LSTM, y_train,X_cv,y_cv,X_test_LSTM, y_test, features_name,train,cv,test)


    # ### loss = MAE
    # design network
    model = models.Sequential()
    model.add(layers.LSTM(50, input_shape=(X_train_LSTM.shape[1], X_train_LSTM.shape[2])))
    model.add(layers.Dense(1))
    model.compile(loss='mae', optimizer='adam')
    # fit network
    history = model.fit(X_train_LSTM, y_train,
                        epochs=40, batch_size=72,
                        validation_data=(X_cv_LSTM, y_cv), verbose=2, shuffle=False)
    score = model.evaluate(X_test_LSTM, y_test, verbose=2)
    print("\nScore MAE:",score)
    # plot history
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='test')
    plt.legend()
    plt.savefig(para.path_results+'LSTM_MAE_history.jpg')
    plt.show()
    LSTM_MAE_model_fit = linear_model_fit_dl(model, 'LSTM_MAE',X_train_LSTM, y_train,X_cv_LSTM,y_cv,X_test_LSTM, y_test, features_name,train,cv,test)


    # ## CNN

    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    X_cv = X_cv.reshape((X_cv.shape[0], X_cv.shape[1], 1))
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
    print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)

    np.set_printoptions(threshold=25)
    window_size = 2360
    filter_length = 5
    nb_filter = 4
    nb_input_series = 1
    nb_outputs = 1
    nb_filter = 4

    # ### loss = MSE

    model = Sequential((Convolution1D(input_shape=(window_size, nb_input_series),
                                      kernel_size=filter_length, activation="relu", filters=nb_filter),
                        MaxPooling1D(),  # Downsample the output of convolution by 2X.
                        # Convolution1D(nb_filter=nb_filter, filter_length=filter_length, activation='relu'),
                        Convolution1D(kernel_size=filter_length, activation="relu", filters=nb_filter),
                        MaxPooling1D(),
                        Flatten(),
                        Dense(nb_outputs, activation='linear'),
                        # For binary classification, change the activation to 'sigmoid'
                        ))
    opt = Adam(lr=0.001)
    # model.compile(loss='mean_squared_error', optimizer=opt, metrics=['mae'])
    model.compile(loss='mse', optimizer='adam')
    # Fitting the RNN to the Training set
    history = model.fit(X_train, y_train,
                        epochs=40, batch_size=72, validation_data=(X_cv, y_cv),
                        verbose=2, shuffle=False)
    score = model.evaluate(X_test, y_test, verbose=2)
    print("\nScore MSE:", score)
    # plot history
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='test')
    plt.legend()
    plt.savefig(para.path_results + 'CNN_MSE_history.jpg')
    plt.show()
    CNN_MSE_model_fit = linear_model_fit_dl(model, 'CNN_MSE',X_train, y_train,X_cv,y_cv,X_test, y_test, features_name,train,cv,test)
    model = Sequential((Convolution1D(input_shape=(window_size, nb_input_series),
                                      kernel_size=filter_length, activation="relu", filters=nb_filter),
                        MaxPooling1D(),  # Downsample the output of convolution by 2X.
                        # Convolution1D(nb_filter=nb_filter, filter_length=filter_length, activation='relu'),
                        Convolution1D(kernel_size=filter_length, activation="relu", filters=nb_filter),
                        MaxPooling1D(),
                        Flatten(),
                        Dense(nb_outputs, activation='linear'),
                        # For binary classification, change the activation to 'sigmoid'
                        ))
    opt = Adam(lr=0.001)
    # model.compile(loss='mae', optimizer=opt, metrics=['mae'])
    model.compile(loss='mae', optimizer='adam')
    # Fitting the RNN to the Training set
    history = model.fit(X_train, y_train,
                        epochs=40, batch_size=72, validation_data=(X_test, y_test),
                        verbose=2, shuffle=False)
    score = model.evaluate(X_test, y_test, verbose=2)
    print("\nScore MAE:", score)
    # plot history
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='test')
    plt.legend()
    plt.savefig(para.path_results + 'CNN_MAE_history.jpg')
    plt.show()
    CNN_MAE_model_fit = linear_model_fit_dl(model, 'CNN_MAE',X_train, y_train,X_cv,y_cv,X_test, y_test, features_name,train,cv,test)


