#coding=utf-8
# python35 pandas0.21.0 !!important

__author__ = "Mengxuan Chen"
__email__  = "chenmx19@mails.tsinghua.edu.cn"
__date__   = "20200102"


#--* pakages*--
import math
import pandas as pd
import numpy as np
from datetime import datetime
import xlrd
import os
from Alpha_generator import Generator

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

    def generate_alpha(self,date):
        g = Generator(date)
        a1 = g.alpha_001()
        a2 = g.alpha_002()
        a3 = g.alpha_003()
        a4 = g.alpha_004()
        a5 = g.alpha_005()
        a6 = g.alpha_006()
        a7 = g.alpha_007()
        a8 = g.alpha_008()
        a9 = g.alpha_009()
        a10 = g.alpha_010()
        a11 = g.alpha_011()
        a12 = g.alpha_012()
        a13 = g.alpha_013()
        a14 = g.alpha_014()
        a15 = g.alpha_015()
        a16 = g.alpha_016()
        a17 = g.alpha_017()
        a18 = g.alpha_018()
        a19 = g.alpha_019()
        a20 = g.alpha_020()
        a21 = g.alpha_021()
        a22 = g.alpha_022()
        a23 = g.alpha_023()
        a24 = g.alpha_024()
        a25 = g.alpha_025()
        a26 = g.alpha_026()
        a27 = g.alpha_027()
        a28 = g.alpha_028()
        a29 = g.alpha_029()
        a30 = g.alpha_030()
        a31 = g.alpha_031()
        a32 = g.alpha_032()
        a33 = g.alpha_033()
        a34 = g.alpha_034()
        a35 = g.alpha_035()
        a36 = g.alpha_036()
        a37 = g.alpha_037()
        a38 = g.alpha_038()
        a39 = g.alpha_039()
        a40 = g.alpha_040()
        a41 = g.alpha_041()
        a42 = g.alpha_042()
        a43 = g.alpha_043()
        a44 = g.alpha_044()
        a45 = g.alpha_045()
        a46 = g.alpha_046()
        a47 = g.alpha_047()
        a48 = g.alpha_048()
        a49 = g.alpha_049()
        a52 = g.alpha_052()
        a53 = g.alpha_053()
        a54 = g.alpha_054()
        a56 = g.alpha_056()
        a57 = g.alpha_057()
        a58 = g.alpha_058()
        a59 = g.alpha_059()
        a60 = g.alpha_060()
        a61 = g.alpha_061()
        a62 = g.alpha_062()
        a63 = g.alpha_063()
        a64 = g.alpha_064()
        a65 = g.alpha_065()
        a66 = g.alpha_066()
        a67 = g.alpha_067()
        a68 = g.alpha_068()
        a70 = g.alpha_070()
        a71 = g.alpha_071()
        a72 = g.alpha_072()
        a74 = g.alpha_074()
        a76 = g.alpha_076()
        a77 = g.alpha_077()
        a78 = g.alpha_078()
        a79 = g.alpha_079()
        a80 = g.alpha_080()
        a81 = g.alpha_081()
        a82 = g.alpha_082()
        a83 = g.alpha_083()
        a84 = g.alpha_084()
        a85 = g.alpha_085()
        a86 = g.alpha_086()
        a87 = g.alpha_087()
        a89 = g.alpha_089()
        a90 = g.alpha_090()
        a91 = g.alpha_091()
        a93 = g.alpha_093()
        a94 = g.alpha_094()
        a95 = g.alpha_095()
        a96 = g.alpha_096()
        a97 = g.alpha_097()
        a98 = g.alpha_098()
        a99 = g.alpha_099()
        a100 = g.alpha_100()
        a101 = g.alpha_101()
        a102 = g.alpha_102()
        a104 = g.alpha_104()
        a105 = g.alpha_105()
        a106 = g.alpha_106()
        a107 = g.alpha_107()
        a108 = g.alpha_108()
        a109 = g.alpha_109()
        a110 = g.alpha_110()
        a111 = g.alpha_111()
        a112 = g.alpha_112()
        a113 = g.alpha_113()
        a114 = g.alpha_114()
        a116 = g.alpha_116()
        a117 = g.alpha_117()
        a118 = g.alpha_118()
        a120 = g.alpha_120()
        a122 = g.alpha_122()
        a123 = g.alpha_123()
        a124 = g.alpha_124()
        a125 = g.alpha_125()
        a126 = g.alpha_126()
        a129 = g.alpha_129()
        a130 = g.alpha_130()
        a132 = g.alpha_132()
        a134 = g.alpha_134()
        a135 = g.alpha_135()
        a136 = g.alpha_136()
        a139 = g.alpha_139()
        a141 = g.alpha_141()
        a142 = g.alpha_142()
        a144 = g.alpha_144()
        a145 = g.alpha_145()
        a148 = g.alpha_148()
        a150 = g.alpha_150()
        a152 = g.alpha_152()
        a153 = g.alpha_153()
        a154 = g.alpha_154()
        a155 = g.alpha_155()
        a158 = g.alpha_158()
        a159 = g.alpha_159()
        a160 = g.alpha_160()
        a161 = g.alpha_161()
        a162 = g.alpha_162()
        a163 = g.alpha_163()
        a164 = g.alpha_164()
        a167 = g.alpha_167()
        a168 = g.alpha_168()
        a169 = g.alpha_169()
        a170 = g.alpha_170()
        a171 = g.alpha_171()
        a172 = g.alpha_172()
        a173 = g.alpha_173()
        a174 = g.alpha_174()
        a176 = g.alpha_176()
        a178 = g.alpha_178()
        a179 = g.alpha_179()
        a180 = g.alpha_180()
        a184 = g.alpha_184()
        a185 = g.alpha_185()
        a186 = g.alpha_186()
        a187 = g.alpha_187()
        a188 = g.alpha_188()
        a189 = g.alpha_189()
        a191 = g.alpha_191()

        b = pd.concat(
            [a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14, a15, a16, a17, a18, a19, a20, a21, a22, a23, a24,a25, a26, a28, a29, a31, a32, a33, a34, a35, a36, a37, a38, a39, a40, a41, a42, a43, a44, a45, a46,a47, a48, a49, a52, a53, a54, a56, a57, a58, a59, a60, a61, a62, a63, a64, a65, a66, a67, a68, a70, a71, a72,a74, a76, a77, a78, a79, a80, a81, a82, a83, a84, a85, a86, a87, a89, a90, a91, a93, a94,a95, a96, a97, a98, a99, a100, a101, a102, a104, a105, a106, a107, a108, a109,a110,a111, a112,a113,a114, a116, a117, a118, a120, a122, a123, a124, a125, a126, a129, a130, a132, a134, a135, a136, a139, a141, a142, a144, a145, a148, a150, a152, a153,a154, a155, a158, a159, a160, a161, a162, a163, a164, a167, a168, a169, a170, a171,a172, a173, a174, a176, a178, a179, a180, a184, a185, a186, a187, a188, a189, a191],
            axis=1)

        b.columns = ['alpha1', 'alpha2', 'alpha3', 'alpha4', 'alpha5', 'alpha6', 'alpha7', 'alpha8', 'alpha9', 'alpha10',
                     'alpha11', 'alpha12', 'alpha13', 'alpha14', 'alpha15', 'alpha16', 'alpha17', 'alpha18', 'alpha19',
                     'alpha20', 'alpha21', 'alpha22', 'alpha23', 'alpha24', 'alpha25', 'alpha26',
                     'alpha28', 'alpha29', 'alpha31', 'alpha32', 'alpha33', 'alpha34', 'alpha35',
                     'alpha36', 'alpha37', 'alpha38', 'alpha39', 'alpha40', 'alpha41', 'alpha42', 'alpha43', 'alpha44',
                     'alpha45', 'alpha46', 'alpha47', 'alpha48', 'alpha49', 'alpha52', 'alpha53', 'alpha54', 'alpha56',
                     'alpha57', 'alpha58', 'alpha59', 'alpha60', 'alpha61', 'alpha62', 'alpha63', 'alpha64', 'alpha65',
                     'alpha66', 'alpha67', 'alpha68', 'alpha70', 'alpha71', 'alpha72', 'alpha74',
                     'alpha76', 'alpha77', 'alpha78', 'alpha79', 'alpha80', 'alpha81', 'alpha82', 'alpha83', 'alpha84',
                     'alpha85', 'alpha86', 'alpha87', 'alpha89', 'alpha90', 'alpha91', 'alpha93', 'alpha94', 'alpha95',
                     'alpha96', 'alpha97', 'alpha98', 'alpha99', 'alpha100', 'alpha101', 'alpha102',
                     'alpha104', 'alpha105', 'alpha106', 'alpha107', 'alpha108', 'alpha109', 'alpha110',
                     'alpha111','alpha112', 'alpha113', 'alpha114', 'alpha116', 'alpha117', 'alpha118', 'alpha120', 'alpha122',
                     'alpha123', 'alpha124', 'alpha125', 'alpha126', 'alpha129', 'alpha130', 'alpha132', 'alpha134',
                     'alpha135', 'alpha136', 'alpha139', 'alpha141', 'alpha142', 'alpha144', 'alpha145', 'alpha148',
                     'alpha150', 'alpha152', 'alpha153', 'alpha154', 'alpha155', 'alpha158', 'alpha159', 'alpha160',
                     'alpha161', 'alpha162', 'alpha163', 'alpha164', 'alpha167', 'alpha168', 'alpha169', 'alpha170',
                     'alpha171', 'alpha172', 'alpha173', 'alpha174', 'alpha176', 'alpha178', 'alpha179', 'alpha180',
                     'alpha184', 'alpha185', 'alpha186', 'alpha187', 'alpha188', 'alpha189', 'alpha191']

        return b
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
        df.head()
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

                est_df_.to_csv(para.path_predictcsv + stockno[i] + '_'+str(j)+ '_'+ 'out_of_sample_result_%s.csv' % model_name,
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
    ########################################################
    date1=pd.read_csv(para.path_data+'datelist.csv')
    date = pd.DataFrame(date1)
    datelist = date.iloc[:, 0]
    for i in range(len(date)):
        current_datet=str(datelist[i])
        current_date=current_datet[0:10]
        data = main().generate_alpha(current_date)
        print(current_date)
        data.to_csv(para.path_data+'result\\' + current_datet + '.csv')

    ########################################################
    path = para.path_data+'result\\'

    def get_file():
        files =os.listdir(path)
        files.sort() #排序
        list= []
        for file in files:
            if not  os.path.isdir(path +file):
                f_name = str(file)
                filename = path + f_name
                list.append(filename)
        return list
    list = get_file()

    csvdate = pd.read_csv(para.path_data+'datelisttest.csv', header=None)#若step1完全运行，文件data/result中生成全部数据后可将该读入文件改为datelist，此处datelisttest仅为测试用
    datelist = np.array(csvdate)  # 转换为 ndarray [[1], [2], [3]]
    datelist = datelist.reshape(1, len(datelist)).tolist()  # 转换成 List [[1, 2, 3]]

    csvstock = pd.read_csv(para.path_data+'stockno.csv', header=None)
    stockno = np.array(csvstock)  # 转换为 ndarray [[1], [2], [3]]
    stockno = stockno.reshape(1, len(stockno)).tolist()# 转换成 List [[1, 2, 3]]
    stockno=stockno[0]


    for j in range (len(list)):
        data = pd.read_csv(list[j])
        df = pd.DataFrame(data)
        expand = pd.DataFrame(df.iloc[j])

        for i in range(len(list) - 1):
            data2 = pd.read_csv(list[i + 1])
            df2 = pd.DataFrame(data2)
            expandhelp = pd.DataFrame(df2.iloc[j])
            expand = pd.concat([expand, expandhelp], axis=1)
        stockalpha = expand.iloc[1:]

        stockalpha.columns = datelist
        stockalpha = stockalpha.T
        print(stockalpha)
        stockalpha.to_csv(para.path_data+'result_stock\\'+stockno[j]+'.csv')
    ########################################################
    path = para.path_data+'result_stock\\'
    def get_file():
        files = os.listdir(path)
        files.sort()  # 排序
        list = []
        for file in files:
            f_name = str(file)
            filename = path + f_name
            list.append(filename)
        return list


    list = get_file()

    csvstock = pd.read_csv(para.path_data+'stockno.csv', header=None)
    stockno = np.array(csvstock)
    stockno = stockno.reshape(1, len(stockno)).tolist()
    stockno = stockno[0]
    csvdate = pd.read_csv(para.path_data+'totalclose.csv')

    for i in range(len(list)):
        data = pd.read_csv(list[i])
        df = pd.DataFrame(data)
        # df.sort_values(by, inplace=True, ascending=True)
        df.dropna(axis=1, how='all', inplace=True)
        df.fillna(0, inplace=True)
        # print(df.isnull().any())

        stockclose = csvdate.loc[:, [stockno[i]]]
        print(stockclose)
        stockclose.columns = ['close']
        dff = pd.concat([df, stockclose], axis=1)
        dff['return'] = dff['close'].pct_change()
        dff.drop(['close'], axis=1, inplace=True)
        # print(dff.isnull().any())
        dff.fillna(0, inplace=True)
        print(dff)

        dff.rename(columns={'Unnamed: 0': 'Date'}, inplace=True)
        dff.fillna(0, inplace=True)
        dffd = dff
        print(dffd)

        dffd.to_csv(para.path_data+'sh300alphainputs\\' + stockno[i] + '.csv',
                    index=False)