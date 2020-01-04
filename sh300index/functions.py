#coding=utf-8

__author__ = "Mengxuan Chen"
__email__  = "chenmx19@mails.tsinghua.edu.cn"
__date__   = "20200102"

#--* pakages*--
import math
import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns
sns.set()

from sklearn.metrics import mean_squared_error, r2_score
from sklearn import preprocessing
from matplotlib import pyplot as plt
from pylab import rcParams
#%matplotlib inline
from alpha_generator import Generator
import warnings
warnings.filterwarnings("ignore")

#--* parameters*--
class Para:
    path_data = '.\\input\\'
    path_results = '.\\output\\'
para = Para()

#--* generate the alphas*--
def generate_a_line(end_index):
    g = Generator(end_index)
    #a1 = g.alpha_001()
    a2 = g.alpha_002()
    a3 = g.alpha_003()
    a4 = g.alpha_004()
    a5 = g.alpha_005()
    #a6 = g.alpha_006()
    #a7 = g.alpha_007()
    #a8 = g.alpha_008()
    a9 = g.alpha_009()
    #a10 = g.alpha_010()
    a11 = g.alpha_011()
    #a12 = g.alpha_012()
    a13 = g.alpha_013()
    a14 = g.alpha_014()
    a15 = g.alpha_015()
    #a16 = g.alpha_016()
    #a17 = g.alpha_017()
    a18 = g.alpha_018()
    a19 = g.alpha_019()
    a20 = g.alpha_020()
    a21 = g.alpha_021()
    a22 = g.alpha_022()
    a23 = g.alpha_023()
    a24 = g.alpha_024()
    a25 = g.alpha_025()
    #a26 = g.alpha_026()
    #a27 = g.alpha_027()
    a28 = g.alpha_028()
    a29 = g.alpha_029()
    #a30 = g.alpha_030()
    a31 = g.alpha_031()
    #a32 = g.alpha_032()
    #a33 = g.alpha_033()
    a34 = g.alpha_034()
    a35 = g.alpha_035()
    #a36 = g.alpha_036()
    a37 = g.alpha_037()
    a38 = g.alpha_038()
    #a39 = g.alpha_039()
    a40 = g.alpha_040()
    #a41 = g.alpha_041()
    a42 = g.alpha_042()
    a43 = g.alpha_043()
    a44 = g.alpha_044()
    #a45 = g.alpha_045()
    a46 = g.alpha_046()
    a47 = g.alpha_047()
    a48 = g.alpha_048()
    a49 = g.alpha_049()
    a52 = g.alpha_052()
    a53 = g.alpha_053()
    #a54 = g.alpha_054()
    #a56 = g.alpha_056()
    a57 = g.alpha_057()
    a58 = g.alpha_058()
    a59 = g.alpha_059()
    a60 = g.alpha_060()
    #a61 = g.alpha_061()
    #a62 = g.alpha_062()
    a63 = g.alpha_063()
    #a64 = g.alpha_064()
    a65 = g.alpha_065()
    a66 = g.alpha_066()
    a67 = g.alpha_067()
    a68 = g.alpha_068()
    a70 = g.alpha_070()
    a71 = g.alpha_071()
    a72 = g.alpha_072()
    #a74 = g.alpha_074()
    a76 = g.alpha_076()
    #a77 = g.alpha_077()
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
    #a90 = g.alpha_090()
    #a91 = g.alpha_091()
    a93 = g.alpha_093()
    a94 = g.alpha_094()
    a95 = g.alpha_095()
    a96 = g.alpha_096()
    a97 = g.alpha_097()
    a98 = g.alpha_098()
    #a99 = g.alpha_099()
    a100 = g.alpha_100()
    #a101 = g.alpha_101()
    a102 = g.alpha_102()
    a104 = g.alpha_104()
    #a105 = g.alpha_105()
    a106 = g.alpha_106()
    #a107 = g.alpha_107()
    #a108 = g.alpha_108()
    a109 = g.alpha_109()
    #a110 = g.alpha_110()
    a111 = g.alpha_111()
    a112 = g.alpha_112()
    a113 = g.alpha_113()
    a114 = g.alpha_114()
    a116 = g.alpha_116()
    a117 = g.alpha_117()
    a118 = g.alpha_118()
    #a120 = g.alpha_120()
    a122 = g.alpha_122()
    #a123 = g.alpha_123()
    a124 = g.alpha_124()
    #a125 = g.alpha_125()
    a126 = g.alpha_126()
    a129 = g.alpha_129()
    #a130 = g.alpha_130()
    #a132 = g.alpha_132()
    a134 = g.alpha_134()
    a135 = g.alpha_135()
    a136 = g.alpha_136()
    a139 = g.alpha_139()
    #a141 = g.alpha_141()
    #a142 = g.alpha_142()
    a144 = g.alpha_144()
    a145 = g.alpha_145()
    #a148 = g.alpha_148()
    a150 = g.alpha_150()
    a152 = g.alpha_152()
    a153 = g.alpha_153()
    #a154 = g.alpha_154()
    a155 = g.alpha_155()
    a158 = g.alpha_158()
    a159 = g.alpha_159()
    a160 = g.alpha_160()
    a161 = g.alpha_161()
    a162 = g.alpha_162()
    #a163 = g.alpha_163()
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
    #a179 = g.alpha_179()
    a180 = g.alpha_180()
    #a184 = g.alpha_184()
    #a185 = g.alpha_185()
    a186 = g.alpha_186()
    a187 = g.alpha_187()
    a188 = g.alpha_188()
    a189 = g.alpha_189()
    a191 = g.alpha_191()
    a_date = pd.Series(g.end_date, index=None)
    a_date.index = ['a']
    b = pd.concat(
        [a_date, a2, a3, a4, a5,  a9,  a11,  a13, a14, a15,  a18, a19,a20, a21,a22, a23, a24,
         a25,  a28, a29, a31,  a34, a35,  a37, a38,  a40, a42, a43, a44, a46,
         a47, a48, a49, a52, a53,  a57, a58, a59, a60, a63,  a65, a66, a67, a68, a70, a71, a72,
          a76,  a78, a79, a80, a81, a82, a83, a84, a85, a86, a87, a89, a93, a94,
         a95, a96, a97, a98,  a100, a102, a104,  a106,  a109,  a111, a112,
         a113, a114, a116, a117, a118, a122,  a124,  a126, a129,
         a134, a135, a136, a139,  a144, a145,a150, a152, a153,
         a155, a158, a159, a160, a161, a162,  a164, a167, a168, a169, a170, a171,
         a172, a173, a174, a176, a178,  a180,  a186, a187, a188, a189, a191],
        axis=1)

    b.columns = ['Date',  'alpha2', 'alpha3', 'alpha4', 'alpha5',  'alpha9','alpha11',  'alpha13', 'alpha14', 'alpha15',  'alpha18', 'alpha19','alpha20', 'alpha21', 'alpha22', 'alpha23', 'alpha24', 'alpha25',
                 'alpha28', 'alpha29', 'alpha31','alpha34', 'alpha35','alpha37', 'alpha38',  'alpha40', 'alpha42', 'alpha43', 'alpha44',
                  'alpha46', 'alpha47', 'alpha48', 'alpha49', 'alpha52', 'alpha53', 'alpha57', 'alpha58', 'alpha59', 'alpha60', 'alpha63',  'alpha65',
                 'alpha66', 'alpha67', 'alpha68', 'alpha70', 'alpha71', 'alpha72', 'alpha76', 'alpha78', 'alpha79', 'alpha80', 'alpha81', 'alpha82', 'alpha83', 'alpha84',
                 'alpha85', 'alpha86', 'alpha87', 'alpha89', 'alpha93', 'alpha94', 'alpha95','alpha96', 'alpha97', 'alpha98',  'alpha100', 'alpha102',
                 'alpha104', 'alpha106',  'alpha109', 'alpha111','alpha112', 'alpha113', 'alpha114', 'alpha116', 'alpha117', 'alpha118', 'alpha122',
                 'alpha124',  'alpha126', 'alpha129',  'alpha134','alpha135', 'alpha136', 'alpha139',  'alpha144', 'alpha145',
                 'alpha150', 'alpha152', 'alpha153',  'alpha155', 'alpha158', 'alpha159', 'alpha160',
                 'alpha161', 'alpha162', 'alpha164', 'alpha167', 'alpha168', 'alpha169', 'alpha170',
                 'alpha171', 'alpha172', 'alpha173', 'alpha174', 'alpha176', 'alpha178', 'alpha180',
                  'alpha186', 'alpha187', 'alpha188', 'alpha189', 'alpha191']

    return b

#--* data_preprocess to split the data into train, cv,test*--
def data_preprocess(df):
    df['return'] = df['close'].pct_change()
    df.dropna(axis=1, how='all', inplace=True)
    df.fillna(0, inplace=True)  # print(df.isnull())
    test_size = 0.2
    cv_size = 0.2
    Nmax = 30
    num_cv = int(cv_size * len(df))
    num_test = int(test_size * len(df))
    num_train = len(df) - num_cv - num_test

    train = df[:num_train]
    cv = df[num_train:num_train + num_cv]
    train_cv = df[:num_train + num_cv]
    test = df[num_train + num_cv:]

    # Split into X and y
    indica = df.columns.tolist()
    indica = df.columns.tolist()
    indica.remove('Date')
    indica.remove('return')
    indica.remove('close')
    features = indica.copy()
    target = ['return']

    X_train = train[features]
    quantile_transformer = preprocessing.QuantileTransformer(output_distribution='normal',
                                                             random_state=0)
    X_train = quantile_transformer.fit_transform(X_train).copy()
    y_train = train[target]
    y_train = y_train.fillna(0)

    X_cv = cv[features]
    y_cv = cv[target]
    quantile_transformer = preprocessing.QuantileTransformer(output_distribution='normal',
                                                             random_state=0)
    X_cv = quantile_transformer.fit_transform(X_cv).copy()

    X_test = test[features]
    y_test = test[target]
    quantile_transformer = preprocessing.QuantileTransformer(output_distribution='normal',
                                                             random_state=0)
    X_test = quantile_transformer.fit_transform(X_test).copy()
    y_test = y_test.fillna(0)
    return X_train, y_train,X_cv,y_cv,X_test, y_test, features,train,cv,test

#--* calculate the MAPE*--
def get_mape(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

#--* fit  the linear model with feature importance output*--
def linear_model_fit(model, model_name,X_train, y_train,X_cv,y_cv,X_test, y_test, features_name,train,cv,test):
    # Train the regressor
    model.fit(X_train, y_train)

    # coeff
    if model_name == 'Linear_Regression' or 'Rdige_Regression':
        coef = pd.DataFrame(model.coef_, columns=features_name, index=['features_importance'])
    elif model_name == 'RANSAC_Regression' or 'Nearest Neighbor Regression':
        coef = pd.DataFrame(model.score(X_train, y_train), columns=features_name, index=['features_importance'])

    else:
        coef = pd.DataFrame(model.feature_importances_, columns=features_name, index=['features_importance'])
    model_coef = coef.T.copy()
    model_coef.sort_index(ascending=False, inplace=True)
    print(model_coef.head(10).round(3))
    model_coef.to_csv(para.path_results + "features_importance%s" % model_name + '.csv')
    # Do prediction on train set
    est = model.predict(X_train)

    # Calculate RMSE
    print('%s' % model_name + "_in-sample MSE = " + str(mean_squared_error(y_train, est)))
    print('%s' % model_name + "_in-sample RMSE = " + str(math.sqrt(mean_squared_error(y_train, est))))
    print('%s' % model_name + "_in-sample MAPE = " + str(get_mape(y_train, est)))
    print('%s' % model_name + "_in-sample R2 = " + str(r2_score(y_train, est)))

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
    plt.savefig(para.path_results + 'in_sample_result_%s' % model_name)
    plt.show()
    # Do prediction on test set
    est_ = model.predict(X_test)

    # Calculate RMSE
    print('%s' % model_name + "_out-of-sample MSE = " + str(mean_squared_error(y_test, est_)))
    print('%s' % model_name + "_out-of-sample RMSE = " + str(math.sqrt(mean_squared_error(y_test, est_))))
    print('%s' % model_name + "_out-of-sample MAPE = " + str(get_mape(y_test, est_)))
    print('%s' % model_name + "_out-of-sample R2 = " + str(r2_score(y_test, est_)))
    # Plot adjusted close over time
    rcParams['figure.figsize'] = 10, 8  # width 10, height 8
    matplotlib.rcParams.update({'font.size': 14})

    est_df_ = pd.DataFrame({'est': est_.T.tolist()[0],
                            'Date': test['Date']})

    # ax = train.plot(x='Date', y='return', style='b-', grid=True)
    # ax = cv.plot(x='Date', y='return', style='y-', grid=True, ax=ax)
    ax = test.plot(x='Date', y='return', style='g-', grid=True)
    ax = est_df_.plot(x='Date', y='est', style='r-', grid=True, ax=ax)
    ax.legend(['test', 'predictions'])
    ax.set_xlabel("Date")
    ax.set_ylabel("%")
    plt.savefig(para.path_results + 'out_of_sample_result_%s' % model_name)
    plt.show()
    return est_df, est_df_

#--* fit  the linear model without feature importance output*--
def linear_model_fit_no_feature_importance(model, model_name,X_train, y_train,X_cv,y_cv,X_test, y_test, features_name,train,cv,test):

    # Train the regressor
    model.fit(X_train, y_train)
    # Do prediction on train set
    est = model.predict(X_train)

    # Calculate RMSE
    print('%s' % model_name + "_in-sample MSE = " + str(mean_squared_error(y_train, est)))
    print('%s' % model_name + "_in-sample RMSE = " + str(math.sqrt(mean_squared_error(y_train, est))))
    print('%s' % model_name + "_in-sample MAPE = " + str(get_mape(y_train, est)))
    print('%s' % model_name + "_in-sample R2 = " + str(r2_score(y_train, est)))

    # Plot adjusted close over time
    rcParams['figure.figsize'] = 10, 8  # width 10, height 8

    est_df = pd.DataFrame({'est': est.T.tolist()[0],
                           'Date': train['Date']})

    ax = train.plot(x='Date', y='return', style='b-', grid=True)
    ax = cv.plot(x='Date', y='return', style='y-', grid=True, ax=ax)
    #ax = test.plot(x='Date', y='return', style='g-', grid=True, ax=ax)
    ax = est_df.plot(x='Date', y='est', style='r-', grid=True, ax=ax)
    ax.legend(['train', 'dev', 'test', 'est'])
    ax.set_xlabel("Date")
    ax.set_ylabel("%")
    plt.savefig(para.path_results + 'in_sample_result_%s' % model_name)
    plt.show()
    # Do prediction on test set
    est_ = model.predict(X_test)

    # Calculate RMSE
    print('%s' % model_name + "_out-of-sample MSE = " + str(mean_squared_error(y_test, est_)))
    print('%s' % model_name + "_out-of-sample RMSE = " + str(math.sqrt(mean_squared_error(y_test, est_))))
    print('%s' % model_name + "_out-of-sample MAPE = " + str(get_mape(y_test, est_)))
    print('%s' % model_name + "_out-of-sample R2 = " + str(r2_score(y_test, est_)))
    # Plot adjusted close over time
    rcParams['figure.figsize'] = 10, 8  # width 10, height 8
    matplotlib.rcParams.update({'font.size': 14})

    est_df_ = pd.DataFrame({'est': est_.T.tolist()[0],
                            'Date': test['Date']})

    #ax = train.plot(x='Date', y='return', style='b-', grid=True)
    #ax = cv.plot(x='Date', y='return', style='y-', grid=True, ax=ax)
    ax = test.plot(x='Date', y='return', style='g-', grid=True)
    ax = est_df_.plot(x='Date', y='est', style='r-', grid=True, ax=ax)
    ax.legend([ 'test', 'predictions'])
    ax.set_xlabel("Date")
    ax.set_ylabel("%")
    plt.savefig(para.path_results + 'out_of_sample_result_%s' % model_name)
    plt.show()
    return est_df, est_df_

#--* fit  the model without feature importance output*--
def model_fit(model, model_name,X_train, y_train,X_cv,y_cv,X_test, y_test, features_name,train,cv,test):

    # Train the regressor
    model.fit(X_train, y_train)
    # Do prediction on train set
    est = model.predict(X_train)
    # Calculate RMSE
    print('%s' % model_name + "_in-sample MSE = " + str(mean_squared_error(y_train, est)))
    print('%s' % model_name + "_in-sample RMSE = " + str(math.sqrt(mean_squared_error(y_train, est))))
    print('%s' % model_name + "_in-sample MAPE = " + str(get_mape(y_train, est)))
    print('%s' % model_name + "_in-sample R2 = " + str(r2_score(y_train, est)))
    # Plot adjusted close over time
    rcParams['figure.figsize'] = 10, 8  # width 10, height 8

    est_df = pd.DataFrame({'est': est,
                           'Date': train['Date']})

    ax = train.plot(x='Date', y='return', style='b-', grid=True)
    ax = cv.plot(x='Date', y='return', style='y-', grid=True, ax=ax)
    #ax = test.plot(x='Date', y='return', style='g-', grid=True, ax=ax)
    ax = est_df.plot(x='Date', y='est', style='r-', grid=True, ax=ax)
    ax.legend(['train', 'dev', 'test', 'est'])
    ax.set_xlabel("Date")
    ax.set_ylabel("RMB")
    plt.savefig(para.path_results + 'in_sample_result_%s' % model_name)
    plt.show()
    # Do prediction on test set
    est_ = model.predict(X_test)

    # Calculate RMSE
    print('%s' % model_name + "_out-of-sample MSE = " + str(mean_squared_error(y_test, est_)))
    print('%s' % model_name + "_out-of-sample RMSE = " + str(math.sqrt(mean_squared_error(y_test, est_))))
    print('%s' % model_name + "_out-of-sample MAPE = " + str(get_mape(y_test, est_)))
    print('%s' % model_name + "_out-of-sample R2 = " + str(r2_score(y_test, est_)))
    # Plot adjusted close over time
    rcParams['figure.figsize'] = 10, 8  # width 10, height 8
    matplotlib.rcParams.update({'font.size': 14})

    est_df_ = pd.DataFrame({'est': est_,
                            'Date': test['Date']})

    #ax = train.plot(x='Date', y='return', style='b-', grid=True)
    #ax = cv.plot(x='Date', y='return', style='y-', grid=True, ax=ax)
    ax = test.plot(x='Date', y='return', style='g-', grid=True)
    ax = est_df_.plot(x='Date', y='est', style='r-', grid=True, ax=ax)
    ax.legend([ 'test', 'predictions'])
    ax.set_xlabel("Date")
    ax.set_ylabel("RMB")
    plt.savefig(para.path_results + 'out_of_sample_result_%s' % model_name)
    plt.show()
    return est_df, est_df_

#--* fit  the deep learning model*--
def linear_model_fit_dl(model, model_name,X_train, y_train,X_cv,y_cv,X_test, y_test, features_name,train,cv,test):
    est = model.predict(X_train)

    # Calculate RMSE
    print('%s' % model_name + "_in-sample MSE = " + str(mean_squared_error(y_train, est)))
    print('%s' % model_name + "_in-sample RMSE = " + str(math.sqrt(mean_squared_error(y_train, est))))
    print('%s' % model_name + "_in-sample MAPE = " + str(get_mape(y_train, est)))
    print('%s' % model_name + "_in-sample R2 = " + str(r2_score(y_train, est)))

    # Plot adjusted close over time
    rcParams['figure.figsize'] = 10, 8  # width 10, height 8

    est_df = pd.DataFrame({'est': est.T.tolist()[0],
                           'Date': train['Date']})

    ax = train.plot(x='Date', y='return', style='b-', grid=True)
    ax = cv.plot(x='Date', y='return', style='y-', grid=True, ax=ax)
    #ax = test.plot(x='Date', y='return', style='g-', grid=True, ax=ax)
    ax = est_df.plot(x='Date', y='est', style='r-', grid=True, ax=ax)
    ax.legend(['train', 'dev', 'test', 'est'])
    ax.set_xlabel("Date")
    ax.set_ylabel("%")
    plt.savefig(para.path_results + 'in_sample_result_%s' % model_name)
    plt.show()
    # Do prediction on test set
    est_ = model.predict(X_test)

    # Calculate RMSE
    print('%s' % model_name + "_out-of-sample MSE = " + str(mean_squared_error(y_test, est_)))
    print('%s' % model_name + "_out-of-sample RMSE = " + str(math.sqrt(mean_squared_error(y_test, est_))))
    print('%s' % model_name + "_out-of-sample MAPE = " + str(get_mape(y_test, est_)))
    print('%s' % model_name + "_out-of-sample R2 = " + str(r2_score(y_test, est_)))
    # Plot adjusted close over time
    rcParams['figure.figsize'] = 10, 8  # width 10, height 8
    matplotlib.rcParams.update({'font.size': 14})

    est_df_ = pd.DataFrame({'est': est_.T.tolist()[0],
                            'Date': test['Date']})

    #ax = train.plot(x='Date', y='return', style='b-', grid=True)
    #ax = cv.plot(x='Date', y='return', style='y-', grid=True, ax=ax)
    ax = test.plot(x='Date', y='return', style='g-', grid=True)
    ax = est_df_.plot(x='Date', y='est', style='r-', grid=True, ax=ax)
    ax.legend(['test', 'predictions'])
    ax.set_xlabel("Date")
    ax.set_ylabel("%")
    plt.savefig(para.path_results + 'out_of_sample_result_%s' % model_name)
    plt.show()
    return est_df, est_df_