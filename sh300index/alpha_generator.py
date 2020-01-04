#coding=utf-8

__author__ = "Mengxuan Chen"
__email__  = "chenmx19@mails.tsinghua.edu.cn"
__date__   = "20200102"

#--* pakages*--
import pandas as pd
import numpy as np
import scipy as sp
from scipy.stats import linregress

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
class Para:
    path_data = '.\\input\\'
    path_results = '.\\output\\'
    rolling_time = 21
para = Para()
def get_data(end_index):
    data = pd.read_csv(para.path_data+'sh.csv',index_col=0)
    return data[end_index-50:end_index]

class Generator:
    def __init__(self, end_index):
        # price = pd.read_csv('sh.csv',index_col=0)
        price = get_data(end_index)
        self.end_date = price.index[-1]
        self.open_price = pd.DataFrame(price['open'])
        self.high = pd.DataFrame(price['high'])
        self.low = pd.DataFrame(price['low'])
        self.close = pd.DataFrame(price['close'])
        self.avg_price = pd.DataFrame(price['avg_price'])
        self.prev_close = pd.DataFrame(price['pre_close'])
        self.volume = pd.DataFrame(price['volume'])
        self.amount = pd.DataFrame(price['amount'])
        self.open_price.columns = ['a']
        self.high.columns = ['a']
        self.low.columns = ['a']
        self.close.columns = ['a']
        self.avg_price.columns = ['a']
        self.prev_close.columns = ['a']
        self.volume.columns = ['a']
        self.amount.columns = ['a']

    def alpha_001(self):
        data1 = self.volume.diff(periods=1).rank(axis=1, pct=True)
        data2 = ((self.close - self.open_price) / self.open_price).rank(axis=1, pct=True)
        alpha = -data1.iloc[-6:, :].corrwith(data2.iloc[-6:, :]).dropna()
        alpha = alpha.dropna()
        return alpha

    def alpha_002(self):
        ##### -1 * delta((((close-low)-(high-close))/((high-low)),1))####
        result = ((self.close - self.low) - (self.high - self.close)) / ((self.high - self.low)).diff()
        m = result.iloc[-1, :].dropna()
        alpha = m[(m < np.inf) & (m > -np.inf)]
        return alpha.dropna()

        ################################################################

    def alpha_003(self):
        delay1 = self.close.shift()
        condtion1 = (self.close == delay1)
        condition2 = (self.close > delay1)
        condition3 = (self.close < delay1)

        part2 = (self.close - np.minimum(delay1[condition2], self.low[condition2])).iloc[-6:, :]  # 取最近的6位数据
        part3 = (self.close - np.maximum(delay1[condition3], self.low[condition3])).iloc[-6:, :]

        result = part2.fillna(0) + part3.fillna(0)
        alpha = result.sum()
        return alpha.dropna()

    ########################################################################
    def alpha_004(self):
        condition1 = (pd.rolling_std(self.close, 8) < pd.rolling_sum(self.close, 2) / 2)
        condition2 = (pd.rolling_sum(self.close, 2) / 2 < (
                pd.rolling_sum(self.close, 8) / 8 - pd.rolling_std(self.close, 8)))
        condition3 = (1 <= self.volume / pd.rolling_mean(self.volume, 20))
        # condition3

        indicator1 = pd.DataFrame(np.ones(self.close.shape), index=self.close.index,
                                  columns=self.close.columns)  # [condition2]
        indicator2 = -pd.DataFrame(np.ones(self.close.shape), index=self.close.index,
                                   columns=self.close.columns)  # [condition3]

        part0 = pd.rolling_sum(self.close, 8) / 8
        part1 = indicator2[condition1].fillna(0)
        part2 = (indicator1[~condition1][condition2]).fillna(0)
        part3 = (indicator1[~condition1][~condition2][condition3]).fillna(0)
        part4 = (indicator2[~condition1][~condition2][~condition3]).fillna(0)

        result = part0 + part1 + part2 + part3 + part4
        alpha = result.iloc[-1, :]
        return alpha.dropna()

    ################################################################
    def alpha_005(self):
        ts_volume = (self.volume.iloc[-7:, :]).rank(axis=0, pct=True)
        ts_high = (self.high.iloc[-7:, :]).rank(axis=0, pct=True)
        corr_ts = pd.rolling_corr(ts_high, ts_volume, 5)
        alpha = corr_ts.max().dropna()
        alpha = alpha[(alpha < np.inf) & (alpha > -np.inf)]
        return alpha

        ###############################################################

    def alpha_006(self):
        condition1 = ((self.open_price * 0.85 + self.high * 0.15).diff(4) > 1)
        condition2 = ((self.open_price * 0.85 + self.high * 0.15).diff(4) == 1)
        condition3 = ((self.open_price * 0.85 + self.high * 0.15).diff(4) < 1)
        indicator1 = pd.DataFrame(np.ones(self.close.shape), index=self.close.index, columns=self.close.columns)
        indicator2 = pd.DataFrame(np.zeros(self.close.shape), index=self.close.index, columns=self.close.columns)
        indicator3 = -pd.DataFrame(np.ones(self.close.shape), index=self.close.index, columns=self.close.columns)
        part1 = indicator1[condition1].fillna(0)
        part2 = indicator2[condition2].fillna(0)
        part3 = indicator3[condition3].fillna(0)
        result = part1 + part2 + part3
        alpha = (result.rank(axis=1, pct=True)).iloc[-1, :]  # cross section rank
        return alpha.dropna()

    ##################################################################
    def alpha_007(self):
        part1 = (np.maximum(self.avg_price - self.close, 3)).rank(axis=1, pct=True)
        part2 = (np.minimum(self.avg_price - self.close, 3)).rank(axis=1, pct=True)
        part3 = (self.volume.diff(3)).rank(axis=1, pct=True)
        result = part1 + part2 * part3
        alpha = result.iloc[-1, :]
        return alpha.dropna()

    ##################################################################
    def alpha_008(self):
        temp = (self.high + self.low) * 0.2 / 2 + self.avg_price * 0.8
        result = -temp.diff(4)
        alpha = result.rank(axis=1, pct=True)
        alpha = alpha.iloc[-1, :]
        return alpha.dropna()

    ##################################################################
    def alpha_009(self):
        temp = (self.high + self.low) * 0.5 - (self.high.shift() + self.low.shift()) * 0.5 * (
                self.high - self.low) / self.volume  # 计算close_{i-1}
        result = pd.ewma(temp, alpha=2 / 7)
        alpha = result.iloc[-1, :]
        return alpha.dropna()

    ##################################################################
    def alpha_010(self):
        ret = self.close.pct_change()
        condtion = (ret < 0)
        part1 = (pd.rolling_std(ret, 20)[condtion]).fillna(0)
        part2 = (self.close[~condtion]).fillna(0)
        result = np.maximum((part1 + part2) ** 2, 5)
        alpha = result.rank(axis=1, pct=True)
        alpha = alpha.iloc[-1, :]
        return alpha.dropna()

    ##################################################################
    def alpha_011(self):
        temp = ((self.close - self.low) - (self.high - self.close)) / (self.high - self.low)
        result = temp * self.volume
        alpha = result.iloc[-6:, :].sum()
        return alpha.dropna()

    ##################################################################
    def alpha_012(self):
        vwap10 = pd.rolling_sum(self.avg_price, 10) / 10
        temp1 = self.open_price - vwap10
        part1 = temp1.rank(axis=1, pct=True)
        temp2 = (self.close - self.avg_price).abs()
        part2 = -temp2.rank(axis=1, pct=True)
        result = part1 * part2
        alpha = result.iloc[-1, :]
        return alpha.dropna()

    ##################################################################
    def alpha_013(self):
        result = ((self.high - self.low) ** 0.5) - self.avg_price
        alpha = result.iloc[-1, :]
        return alpha.dropna()

    ##################################################################
    def alpha_014(self):
        result = self.close - self.close.shift(5)
        alpha = result.iloc[-1, :]
        return alpha.dropna()

    ##################################################################
    def alpha_015(self):
        result = self.open_price / self.close.shift() - 1
        alpha = result.iloc[-1, :]
        return alpha.dropna()

    ##################################################################
    def alpha_016(self):
        temp1 = self.volume.rank(axis=1, pct=True)
        temp2 = self.avg_price.rank(axis=1, pct=True)
        part = pd.rolling_corr(temp1, temp2, 5)  #
        part = part[(part < np.inf) & (part > -np.inf)]
        result = part.iloc[-5:, :]
        result = result.dropna(axis=1)
        alpha = -result.max()
        return alpha.dropna()

    ##################################################################
    def alpha_017(self):
        temp1 = pd.rolling_max(self.avg_price, 15)
        temp2 = (self.close - temp1).dropna()
        part1 = temp2.rank(axis=1, pct=True)
        part2 = self.close.diff(5)
        result = part1 ** part2
        alpha = result.iloc[-1, :]
        return alpha.dropna()

    ##################################################################
    def alpha_018(self):
        delay5 = self.close.shift(5)
        alpha = self.close / delay5
        alpha = alpha.iloc[-1, :]
        return alpha.dropna()

    ##################################################################
    def alpha_019(self):
        delay5 = self.close.shift(5)
        condition1 = (self.close < delay5)
        condition3 = (self.close > delay5)
        part1 = (self.close[condition1] - delay5[condition1]) / delay5[condition1]
        part1 = part1.fillna(0)
        part2 = (self.close[condition3] - delay5[condition3]) / self.close[condition3]
        part2 = part2.fillna(0)
        result = part1 + part2
        alpha = result.iloc[-1, :]
        return alpha.dropna()

    ##################################################################
    def alpha_020(self):
        delay6 = self.close.shift(6)
        result = (self.close - delay6) * 100 / delay6
        alpha = result.iloc[-1, :]
        return alpha.dropna()

    ##################################################################
    def alpha_021(self):
        A = pd.rolling_mean(self.close, 6).iloc[-6:, :]
        B = np.arange(1, 7)
        temp = A.apply(lambda x: linregress(x, B), axis=0)
        drop_list = [i for i in range(len(temp)) if temp[i][3] > 0.05]
        temp.drop(temp.index[drop_list], inplace=True)
        beta_list = [temp[i].slope for i in range(len(temp))]
        alpha = pd.Series(beta_list, index=temp.index)
        return alpha.dropna()

    ##################################################################
    def alpha_022(self):
        part1 = (self.close - pd.rolling_mean(self.close, 6)) / pd.rolling_mean(self.close, 6)
        temp = (self.close - pd.rolling_mean(self.close, 6)) / pd.rolling_mean(self.close, 6)
        part2 = temp.shift(3)
        result = part1 - part2
        result = pd.ewma(result, alpha=1.0 / 12)
        alpha = result.iloc[-1, :]
        return alpha.dropna()

        ##################################################################

    def alpha_023(self):
        condition1 = (self.close > self.close.shift())
        temp1 = pd.rolling_std(self.close, 20)[condition1]
        temp1 = temp1.fillna(0)
        temp2 = pd.rolling_std(self.close, 20)[~condition1]
        temp2 = temp2.fillna(0)
        part1 = pd.ewma(temp1, alpha=1.0 / 20)
        part2 = pd.ewma(temp2, alpha=1.0 / 20)
        result = part1 * 100 / (part1 + part2)
        alpha = result.iloc[-1, :]
        return alpha.dropna()

    ##################################################################
    def alpha_024(self):
        delay5 = self.close.shift(5)
        result = self.close - delay5
        result = pd.ewma(result, alpha=1.0 / 5)
        alpha = result.iloc[-1, :]
        return alpha.dropna()

    ##################################################################

    def alpha_025(self):
        n = 9
        part1 = (self.close.diff(7)).rank(axis=1, pct=True)
        part1 = part1.iloc[-1, :]
        temp = self.volume / pd.rolling_mean(self.volume, 20)
        temp1 = temp.iloc[-9:, :]
        seq = [2 * i / (n * (n + 1)) for i in range(1, n + 1)]
        weight = np.array(seq)

        temp1 = temp1.apply(lambda x: x * weight)
        ret = self.close.pct_change()
        rank_sum_ret = (ret.sum()).rank(pct=True)
        part2 = 1 - temp1.sum()
        part3 = 1 + rank_sum_ret
        alpha = -part1 * part2 * part3
        return alpha.dropna()

    ##################################################################
    def alpha_026(self):
        part1 = pd.rolling_sum(self.close, 7) / 7 - self.close
        part1 = part1.iloc[-1, :]
        delay5 = self.close.shift(5)
        part2 = pd.rolling_corr(self.avg_price, delay5, 230)
        part2 = part2.iloc[-1, :]
        alpha = part1 + part2
        return alpha.dropna()

    ##################################################################
    def alpha_027(self):
        return 0

    ##################################################################
    def alpha_028(self):
        temp1 = self.close - pd.rolling_min(self.low, 9)
        temp2 = pd.rolling_max(self.high, 9) - pd.rolling_min(self.low, 9)
        part1 = 3 * pd.ewma(temp1 * 100 / temp2, alpha=1.0 / 3)
        temp3 = pd.ewma(temp1 * 100 / temp2, alpha=1.0 / 3)
        part2 = 2 * pd.ewma(temp3, alpha=1.0 / 3)
        result = part1 - part2
        alpha = result.iloc[-1, :]
        return alpha.dropna()

    ##################################################################
    def alpha_029(self):
        delay6 = self.close.shift(6)
        result = (self.close - delay6) * self.volume / delay6
        alpha = result.iloc[-1, :]
        return alpha.dropna()

    ##################################################################
    def alpha_030(self):
        return 0

    ##################################################################
    def alpha_031(self):
        result = (self.close - pd.rolling_mean(self.close, 12)) * 100 / pd.rolling_mean(self.close, 12)
        alpha = result.iloc[-1, :]
        return alpha.dropna()

    ##################################################################
    def alpha_032(self):
        temp1 = self.high.rank(axis=1, pct=True)
        temp2 = self.volume.rank(axis=1, pct=True)
        temp3 = pd.rolling_corr(temp1, temp2, 3)
        temp3 = temp3[(temp3 < np.inf) & (temp3 > -np.inf)].fillna(0)
        result = (temp3.rank(axis=1, pct=True)).iloc[-3:, :]
        alpha = -result.sum()
        return alpha.dropna()

    ##################################################################
    def alpha_033(self):
        ret = self.close.pct_change()
        temp1 = pd.rolling_min(self.low, 5)  # TS_MIN
        part1 = temp1.shift(5) - temp1
        part1 = part1.iloc[-1, :]
        temp2 = (pd.rolling_sum(ret, 240) - pd.rolling_sum(ret, 20)) / 220
        part2 = temp2.rank(axis=1, pct=True)
        part2 = part2.iloc[-1, :]
        temp3 = self.volume.iloc[-5:, :]
        part3 = temp3.rank(axis=0, pct=True)  # TS_RANK
        part3 = part3.iloc[-1, :]
        alpha = part1 + part2 + part3
        return alpha.dropna()

    ##################################################################
    def alpha_034(self):
        result = pd.rolling_mean(self.close, 12) / self.close
        alpha = result.iloc[-1, :]
        return alpha.dropna()

    ##################################################################
    def alpha_035(self):
        n = 15
        m = 7
        temp1 = self.open_price.diff()
        temp1 = temp1.iloc[-n:, :]
        seq1 = [2 * i / (n * (n + 1)) for i in range(1, n + 1)]
        seq2 = [2 * i / (m * (m + 1)) for i in range(1, m + 1)]
        weight1 = np.array(seq1)
        weight2 = np.array(seq2)
        part1 = temp1.apply(lambda x: x * weight1)
        part1 = part1.rank(axis=1, pct=True)

        temp2 = 0.65 * self.open_price + 0.35 * self.open_price
        temp2 = pd.rolling_corr(temp2, self.volume, 17)
        temp2 = temp2.iloc[-m:, :]
        part2 = temp2.apply(lambda x: x * weight2)
        alpha = np.minimum(part1.iloc[-1, :], -part2.iloc[-1, :])
        alpha = alpha[(alpha < np.inf) & (alpha > -np.inf)]
        alpha = alpha.dropna()
        return alpha

    ##################################################################
    def alpha_036(self):
        temp1 = self.volume.rank(axis=1, pct=True)
        temp2 = self.avg_price.rank(axis=1, pct=True)
        part1 = pd.rolling_corr(temp1, temp2, 6)
        result = pd.rolling_sum(part1, 2)
        result = result.rank(axis=1, pct=True)
        alpha = result.iloc[-1, :]
        return alpha.dropna()

    ##################################################################
    def alpha_037(self):
        ret = self.close.pct_change()
        temp = pd.rolling_sum(self.open_price, 5) * pd.rolling_sum(ret, 5)
        part1 = temp.rank(axis=1, pct=True)
        part2 = temp.shift(10)
        result = -part1 - part2
        alpha = result.iloc[-1, :].dropna()
        return alpha

    ##################################################################
    def alpha_038(self):
        sum_20 = pd.rolling_sum(self.high, 20) / 20
        delta2 = self.high.diff(2)
        condition = (sum_20 < self.high)
        result = -delta2[condition].fillna(0)
        alpha = result.iloc[-1, :]
        return alpha

    ##################################################################
    def alpha_039(self):
        n = 8
        m = 12
        temp1 = self.close.diff(2)
        temp1 = temp1.iloc[-n:, :]
        seq1 = [2 * i / (n * (n + 1)) for i in range(1, n + 1)]
        seq2 = [2 * i / (m * (m + 1)) for i in range(1, m + 1)]

        weight1 = np.array(seq1)
        weight2 = np.array(seq2)
        part1 = temp1.apply(lambda x: x * weight1)
        part1 = part1.rank(axis=1, pct=True)

        temp2 = 0.3 * self.avg_price + 0.7 * self.open_price
        volume_180 = pd.rolling_mean(self.volume, 180)
        sum_vol = pd.rolling_sum(volume_180, 37)
        temp3 = pd.rolling_corr(temp2, sum_vol, 14)
        temp3 = temp3.iloc[-m:, :]
        part2 = -temp3.apply(lambda x: x * weight2)
        part2.rank(axis=1, pct=True)
        result = part1.iloc[-1, :] - part2.iloc[-1, :]
        alpha = result
        alpha = alpha[(alpha < np.inf) & (alpha > -np.inf)]
        alpha = alpha.dropna()
        return alpha

    ##################################################################
    def alpha_040(self):
        delay1 = self.close.shift()
        condition = (self.close > delay1)
        vol = self.volume[condition].fillna(0)
        vol_sum = pd.rolling_sum(vol, 26)
        vol1 = self.volume[~condition].fillna(0)
        vol1_sum = pd.rolling_sum(vol1, 26)
        result = 100 * vol_sum / vol1_sum
        result = result.iloc[-1, :]
        alpha = result
        alpha = alpha[(alpha < np.inf) & (alpha > -np.inf)]
        alpha = alpha.dropna()
        return alpha

    ##################################################################
    def alpha_041(self):
        delta_avg = self.avg_price.diff(3)
        part = np.maximum(delta_avg, 5)
        result = -part.rank(axis=1, pct=True)
        alpha = result.iloc[-1, :]
        alpha = alpha.dropna()
        return alpha

    ##################################################################
    def alpha_042(self):
        part1 = pd.rolling_corr(self.high, self.volume, 10)
        part2 = pd.rolling_std(self.high, 10)
        part2 = part2.rank(axis=1, pct=True)
        result = -part1 * part2
        alpha = result.iloc[-1, :]
        alpha = alpha[(alpha < np.inf) & (alpha > -np.inf)]
        alpha = alpha.dropna()
        return alpha

    ##################################################################
    def alpha_043(self):
        delay1 = self.close.shift()
        condition1 = (self.close > delay1)
        condition2 = (self.close < delay1)
        temp1 = self.volume[condition1].fillna(0)
        temp2 = -self.volume[condition2].fillna(0)
        result = temp1 + temp2
        result = pd.rolling_sum(result, 6)
        alpha = result.iloc[-1, :].dropna()
        return alpha

    ##################################################################
    def alpha_044(self):
        part1 = self.open_price * 0.4 + self.close * 0.6
        n = 6
        m = 10
        temp1 = pd.rolling_corr(self.low, pd.rolling_mean(self.volume, 10), 7)
        temp1 = temp1.iloc[-n:, :]
        seq1 = [2 * i / (n * (n + 1)) for i in range(1, n + 1)]
        seq2 = [2 * i / (m * (m + 1)) for i in range(1, m + 1)]
        weight1 = np.array(seq1)
        weight2 = np.array(seq2)
        part1 = temp1.apply(lambda x: x * weight1)
        part1 = part1.iloc[-4:, ].rank(axis=0, pct=True)

        temp2 = self.avg_price.diff(3)
        temp2 = temp2.iloc[-m:, :]
        part2 = temp2.apply(lambda x: x * weight2)
        part2 = part1.iloc[-5:, ].rank(axis=0, pct=True)
        alpha = part1.iloc[-1, :] + part2.iloc[-1, :]
        alpha = alpha.dropna()
        return alpha

    ##################################################################
    def alpha_045(self):
        temp1 = self.close * 0.6 + self.open_price * 0.4
        part1 = temp1.diff()
        part1 = part1.rank(axis=1, pct=True)
        temp2 = pd.rolling_mean(self.volume, 150)
        part2 = pd.rolling_corr(self.avg_price, temp2, 15)
        part2 = part2.rank(axis=1, pct=True)
        result = part1 * part2
        alpha = result.iloc[-1, :].dropna()
        return alpha

    ##################################################################
    def alpha_046(self):
        part1 = pd.rolling_mean(self.close, 3)
        part2 = pd.rolling_mean(self.close, 6)
        part3 = pd.rolling_mean(self.close, 12)
        part4 = pd.rolling_mean(self.close, 24)
        result = (part1 + part2 + part3 + part4) * 0.25 / self.close
        alpha = result.iloc[-1, :].dropna()
        return alpha

    ##################################################################
    def alpha_047(self):
        part1 = pd.rolling_max(self.high, 6) - self.close
        part2 = pd.rolling_max(self.high, 6) - pd.rolling_min(self.low, 6)
        result = pd.ewma(100 * part1 / part2, alpha=1.0 / 9)
        alpha = result.iloc[-1, :].dropna()
        return alpha

        ##################################################################

    def alpha_048(self):
        condition1 = (self.close > self.close.shift())
        condition2 = (self.close.shift() > self.close.shift(2))
        condition3 = (self.close.shift(2) > self.close.shift(3))

        indicator1 = pd.DataFrame(np.ones(self.close.shape), index=self.close.index, columns=self.close.columns)[
            condition1].fillna(0)
        indicator2 = pd.DataFrame(np.ones(self.close.shape), index=self.close.index, columns=self.close.columns)[
            condition2].fillna(0)
        indicator3 = pd.DataFrame(np.ones(self.close.shape), index=self.close.index, columns=self.close.columns)[
            condition3].fillna(0)

        indicator11 = -pd.DataFrame(np.ones(self.close.shape), index=self.close.index, columns=self.close.columns)[
            (~condition1) & (self.close != self.close.shift())].fillna(0)
        indicator22 = -pd.DataFrame(np.ones(self.close.shape), index=self.close.index, columns=self.close.columns)[
            (~condition2) & (self.close.shift() != self.close.shift(2))].fillna(0)
        indicator33 = -pd.DataFrame(np.ones(self.close.shape), index=self.close.index, columns=self.close.columns)[
            (~condition3) & (self.close.shift(2) != self.close.shift(3))].fillna(0)

        summ = indicator1 + indicator2 + indicator3 + indicator11 + indicator22 + indicator33
        result = -summ * pd.rolling_sum(self.volume, 5) / pd.rolling_sum(self.volume, 20)
        alpha = result.iloc[-1, :].dropna()
        return alpha

    ##################################################################
    def alpha_049(self):
        delay_high = self.high.shift()
        delay_low = self.low.shift()
        condition1 = (self.high + self.low >= delay_high + delay_low)
        condition2 = (self.high + self.low <= delay_high + delay_low)
        part1 = np.maximum(np.abs(self.high - delay_high), np.abs(self.low - delay_low))
        part1 = part1[~condition1]
        part1 = part1.iloc[-12:, :].sum()

        part2 = np.maximum(np.abs(self.high - delay_high), np.abs(self.low - delay_low))
        part2 = part2[~condition2]
        part2 = part2.iloc[-12:, :].sum()
        result = part1 / (part1 + part2)
        alpha = result.dropna()
        return alpha

    ##################################################################
    def alpha_052(self):
        delay = ((self.high + self.low + self.close) / 3).shift()
        part1 = (np.maximum(self.high - delay, 0)).iloc[-26:, :]

        part2 = (np.maximum(delay - self.low, 0)).iloc[-26:, :]
        alpha = part1.sum() + part2.sum()
        return alpha

    ##################################################################
    def alpha_053(self):
        delay = self.close.shift()
        condition = self.close > delay
        result = self.close[condition].iloc[-12:, :]
        alpha = result.count() * 100 / 12
        return alpha.dropna()

    ##################################################################
    def alpha_054(self):
        part1 = (self.close - self.open_price).abs()
        part1 = part1.std()
        part2 = (self.close - self.open_price).iloc[-1, :]
        part3 = self.close.iloc[-10:, :].corrwith(self.open_price.iloc[-10:, :])
        result = (part1 + part2 + part3).dropna()
        alpha = result.rank(pct=True)
        return alpha.dropna()

    ##################################################################
    def alpha_056(self):
        part1 = self.open_price.iloc[-1, :] - self.open_price.iloc[-12:, :].min()
        part1 = part1.rank(pct=1)
        temp1 = (self.high + self.low) / 2
        temp1 = pd.rolling_sum(temp1, 19)
        temp2 = pd.rolling_sum(pd.rolling_mean(self.volume, 40), 19)
        part2 = temp1.iloc[-13:, :].corrwith(temp2.iloc[-13:, :])
        part2 = (part2.rank(pct=1)) ** 5
        part2 = part2.rank(pct=1)

        part1[part1 < part2] = 1
        part1 = part1.apply(lambda x: 0 if x < 1 else None)
        alpha = part1.fillna(1)
        return alpha.dropna()

    ##################################################################
    def alpha_057(self):
        part1 = self.close - pd.rolling_min(self.low, 9)
        part2 = pd.rolling_max(self.high, 9) - pd.rolling_min(self.low, 9)
        result = pd.ewma(100 * part1 / part2, alpha=1.0 / 3)
        alpha = result.iloc[-1, :]
        return alpha.dropna()

    ##################################################################
    def alpha_058(self):
        delay = self.close.shift()
        condition = self.close > delay
        result = self.close[condition].iloc[-20:, :]
        alpha = result.count() * 100 / 20
        return alpha.dropna()

        ##################################################################

    def alpha_059(self):
        delay = self.close.shift()
        condition1 = (self.close > delay)
        condition2 = (self.close < delay)
        part1 = np.minimum(self.low[condition1], delay[condition1]).fillna(0)
        part2 = np.maximum(self.high[condition2], delay[condition2]).fillna(0)
        part1 = part1.iloc[-20:, :]
        part2 = part2.iloc[-20:, :]
        result = self.close - part1 - part2
        alpha = result.sum()
        return alpha

    ##################################################################
    def alpha_060(self):
        part1 = (self.close.iloc[-20:, :] - self.low.iloc[-20:, :]) - (
                self.high.iloc[-20:, :] - self.close.iloc[-20:, :])
        part2 = self.high.iloc[-20:, :] - self.low.iloc[-20:, :]
        result = self.volume.iloc[-20:, :] * part1 / part2
        alpha = result.sum()
        return alpha

    ##################################################################
    def alpha_061(self):
        n = 12
        m = 17
        temp1 = self.avg_price.diff()
        temp1 = temp1.iloc[-n:, :]
        seq1 = [2 * i / (n * (n + 1)) for i in range(1, n + 1)]
        seq2 = [2 * i / (m * (m + 1)) for i in range(1, m + 1)]

        weight1 = np.array(seq1)
        weight2 = np.array(seq2)
        part1 = temp1.apply(lambda x: x * weight1)
        part1 = part1.rank(axis=1, pct=True)

        temp2 = self.low
        temp2 = pd.rolling_corr(temp2, pd.rolling_mean(self.volume, 80), 8)
        temp2 = temp2.rank(axis=1, pct=1)
        temp2 = temp2.iloc[-m:, :]
        part2 = temp2.apply(lambda x: x * weight2)
        part2 = -part2.rank(axis=1, pct=1)
        alpha = np.maximum(part1.iloc[-1, :], part2.iloc[-1, :])
        alpha = alpha[(alpha < np.inf) & (alpha > -np.inf)]
        alpha = alpha.dropna()
        return alpha

    ##################################################################
    def alpha_062(self):
        volume_rank = self.volume.rank(axis=1, pct=1)
        result = self.high.iloc[-5:, :].corrwith(volume_rank.iloc[-5:, :])
        alpha = -result
        return alpha.dropna()

        ##################################################################

    def alpha_063(self):
        part1 = np.maximum(self.close - self.close.shift(), 0)
        part1 = pd.ewma(part1, alpha=1.0 / 6)
        part2 = (self.close - self.close.shift()).abs()
        part2 = pd.ewma(part2, alpha=1.0 / 6)
        result = part1 * 100 / part2
        alpha = result.iloc[-1, :]
        return alpha.dropna()

    ##################################################################
    def alpha_064(self):
        n = 4
        m = 14
        temp1 = pd.rolling_corr(self.avg_price.rank(axis=1, pct=1), self.volume.rank(axis=1, pct=1), 4)
        temp1 = temp1.iloc[-n:, :]
        seq1 = [2 * i / (n * (n + 1)) for i in range(1, n + 1)]
        seq2 = [2 * i / (m * (m + 1)) for i in range(1, m + 1)]
        weight1 = np.array(seq1)
        weight2 = np.array(seq2)
        part1 = temp1.apply(lambda x: x * weight1)
        part1 = part1.rank(axis=1, pct=True)

        temp2 = self.close.rank(axis=1, pct=1)
        temp2 = pd.rolling_corr(temp2, pd.rolling_mean(self.volume, 60), 4)
        temp2 = np.maximum(temp2, 13)
        temp2 = temp2.iloc[-m:, :]
        part2 = temp2.apply(lambda x: x * weight2)
        part2 = -part2.rank(axis=1, pct=1)
        alpha = np.maximum(part1.iloc[-1, :], part2.iloc[-1, :])
        alpha = alpha[(alpha < np.inf) & (alpha > -np.inf)]
        alpha = alpha.dropna()
        return alpha

    ##################################################################
    def alpha_065(self):
        part1 = self.close.iloc[-6:, :]
        alpha = part1.mean() / self.close.iloc[-1, :]
        return alpha.dropna()

    ##################################################################
    def alpha_066(self):
        part1 = self.close.iloc[-6:, :]
        alpha = (self.close.iloc[-1, :] - part1.mean()) / part1.mean()
        return alpha

    ##################################################################
    def alpha_067(self):
        temp1 = self.close - self.close.shift()
        part1 = np.maximum(temp1, 0)
        part1 = pd.ewma(part1, alpha=1.0 / 24)
        temp2 = temp1.abs()
        part2 = pd.ewma(temp2, alpha=1.0 / 24)
        result = part1 * 100 / part2
        alpha = result.iloc[-1, :].dropna()
        return alpha

    ##################################################################
    def alpha_068(self):
        part1 = (self.high + self.low) / 2 - self.high.shift()
        part2 = 0.5 * self.low.shift() * (self.high - self.low) / self.volume
        result = (part1 + part2) * 100
        result = pd.ewma(result, alpha=2.0 / 15)
        alpha = result.iloc[-1, :].dropna()
        return alpha

    ##################################################################
    def alpha_070(self):
        #### STD(AMOUNT, 6)
        ##
        alpha = self.amount.iloc[-6:, :].std().dropna()
        return alpha

    #############################################################################
    def alpha_071(self):
        # (CLOSE-MEAN(CLOSE,24))/MEAN(CLOSE,24)*100
        #
        data = self.close - pd.rolling_mean(self.close, 24) / pd.rolling_mean(self.close, 24)
        alpha = data.iloc[-1].dropna()
        return alpha

    #############################################################################
    def alpha_072(self):
        # SMA((TSMAX(HIGH,6)-CLOSE)/(TSMAX(HIGH,6)-TSMIN(LOW,6))*100,15,1)
        #
        data1 = pd.rolling_max(self.high, 6) - self.close
        data2 = pd.rolling_max(self.high, 6) - pd.rolling_min(self.low, 6)
        alpha = pd.ewma(data1 / data2 * 100, alpha=1 / 15).iloc[-1].dropna()
        return alpha

    #############################################################################
    def alpha_074(self):
        # (RANK(CORR(SUM(((LOW * 0.35) + (VWAP * 0.65)), 20), SUM(MEAN(VOLUME,40), 20), 7)) + RANK(CORR(RANK(VWAP), RANK(VOLUME), 6)))
        #
        data1 = pd.rolling_sum((self.low * 0.35 + self.avg_price * 0.65), window=20)
        data2 = pd.rolling_mean(self.volume, window=40)
        rank1 = pd.rolling_corr(data1, data2, window=7).rank(axis=1, pct=True)
        data3 = self.avg_price.rank(axis=1, pct=True)
        data4 = self.volume.rank(axis=1, pct=True)
        rank2 = pd.rolling_corr(data3, data4, window=6).rank(axis=1, pct=True)
        alpha = (rank1 + rank2).iloc[-1].dropna()
        return alpha

    #############################################################################
    def alpha_076(self):
        # STD(ABS((CLOSE/DELAY(CLOSE,1)-1))/VOLUME,20)/MEAN(ABS((CLOSE/DELAY(CLOSE,1)-1))/VOLUME,20)
        #

        data1 = abs((self.close / ((self.prev_close - 1) / self.volume).shift(20))).std()
        data2 = abs((self.close / ((self.prev_close - 1) / self.volume).shift(20))).mean()
        alpha = (data1 / data2).dropna()
        return alpha

    #############################################################################
    def alpha_077(self):
        # MIN(RANK(DECAYLINEAR(((((HIGH + LOW) / 2) + HIGH)  -  (VWAP + HIGH)), 20)), RANK(DECAYLINEAR(CORR(((HIGH + LOW) / 2), MEAN(VOLUME,40), 3), 6)))
        #

        data1 = ((self.high + self.low) / 2 + self.high - (self.avg_price + self.high)).iloc[-20:, :]
        decay_weights = np.arange(1, 20 + 1, 1)[::-1]
        decay_weights = decay_weights / decay_weights.sum()
        rank1 = data1.apply(lambda x: x * decay_weights).rank(axis=1, pct=True)
        data2 = pd.rolling_corr((self.high + self.low) / 2, pd.rolling_mean(self.volume, window=40), window=3).iloc[-6:,
                :]
        decay_weights2 = np.arange(1, 6 + 1, 1)[::-1]
        decay_weights2 = decay_weights2 / decay_weights2.sum()
        rank2 = data2.apply(lambda x: x * decay_weights2).rank(axis=1, pct=True)
        alpha = np.minimum(rank1.iloc[-1], rank2.iloc[-1])
        return alpha

    #############################################################################
    def alpha_078(self):
        # ((HIGH+LOW+CLOSE)/3-MA((HIGH+LOW+CLOSE)/3,12))/(0.015*MEAN(ABS(CLOSE-MEAN((HIGH+LOW+CLOSE)/3,12)),12))
        #
        data1 = (self.high + self.low + self.close) / 3 - pd.rolling_mean((self.high + self.low + self.close) / 3,
                                                                          window=12)
        data2 = abs(self.close - pd.rolling_mean((self.high + self.low + self.close) / 3, window=12))
        data3 = pd.rolling_mean(data2, window=12) * 0.015
        alpha = (data1 / data3).iloc[-1].dropna()
        return alpha

    #############################################################################
    def alpha_079(self):
        # SMA(MAX(CLOSE-DELAY(CLOSE,1),0),12,1)/SMA(ABS(CLOSE-DELAY(CLOSE,1)),12,1)*100
        #
        data1 = pd.ewma(np.maximum((self.close - self.prev_close), 0), alpha=1 / 12)
        data2 = pd.ewma(abs(self.close - self.prev_close), alpha=1 / 12)
        alpha = (data1 / data2 * 100).iloc[-1].dropna()
        return alpha

    #############################################################################
    def alpha_080(self):
        # (VOLUME-DELAY(VOLUME,5))/DELAY(VOLUME,5)*100
        #
        alpha = ((self.volume - self.volume.shift(5)) / self.volume.shift(5) * 100).iloc[-1].dropna()
        return alpha

    #############################################################################
    def alpha_081(self):
        result = pd.ewma(self.volume, alpha=2.0 / 21)
        alpha = result.iloc[-1, :].dropna()
        return alpha

    #############################################################################
    def alpha_082(self):
        part1 = pd.rolling_max(self.high, 6) - self.close
        part2 = pd.rolling_max(self.high, 6) - pd.rolling_min(self.low, 6)
        result = pd.ewma(100 * part1 / part2, alpha=1.0 / 20)
        alpha = result.iloc[-1, :].dropna()
        return alpha

    #############################################################################
    def alpha_083(self):
        part1 = self.high.rank(axis=0, pct=True)
        part1 = part1.iloc[-5:, :]
        part2 = self.volume.rank(axis=0, pct=True)
        part2 = part2.iloc[-5:, :]
        result = part1.corrwith(part2)
        alpha = -result
        return alpha.dropna()

    #############################################################################
    def alpha_084(self):
        condition1 = (self.close > self.close.shift())
        condition2 = (self.close < self.close.shift())
        part1 = self.volume[condition1].fillna(0)
        part2 = -self.volume[condition2].fillna(0)
        result = part1.iloc[-20:, :] + part2.iloc[-20:, :]
        alpha = result.sum().dropna()
        return alpha

    #############################################################################
    def alpha_085(self):
        temp1 = self.volume.iloc[-20:, :] / self.volume.iloc[-20:, :].mean()
        temp1 = temp1
        part1 = temp1.rank(axis=0, pct=True)
        part1 = part1.iloc[-1, :]

        delta = self.close.diff(7)
        temp2 = -delta.iloc[-8:, :]
        part2 = temp2.rank(axis=0, pct=True).iloc[-1, :]
        part2 = part2
        alpha = part1 * part2
        return alpha.dropna()

    #############################################################################
    def alpha_086(self):
        delay10 = self.close.shift(10)
        delay20 = self.close.shift(20)
        indicator1 = pd.DataFrame(-np.ones(self.close.shape), index=self.close.index, columns=self.close.columns)
        indicator2 = pd.DataFrame(np.ones(self.close.shape), index=self.close.index, columns=self.close.columns)

        temp = (delay20 - delay10) / 10 - (delay10 - self.close) / 10
        condition1 = (temp > 0.25)
        condition2 = (temp < 0)
        temp2 = (self.close - self.close.shift()) * indicator1

        part1 = indicator1[condition1].fillna(0)
        part2 = indicator2[~condition1][condition2].fillna(0)
        part3 = temp2[~condition1][~condition2].fillna(0)
        result = part1 + part2 + part3
        alpha = result.iloc[-1, :].dropna()

        return alpha

    #############################################################################
    def alpha_087(self):
        n = 7
        m = 11
        temp1 = self.avg_price.diff(4)
        temp1 = temp1.iloc[-n:, :]
        seq1 = [2 * i / (n * (n + 1)) for i in range(1, n + 1)]
        seq2 = [2 * i / (m * (m + 1)) for i in range(1, m + 1)]

        weight1 = np.array(seq1)
        weight2 = np.array(seq2)
        part1 = temp1.apply(lambda x: x * weight1)
        part1 = part1.rank(axis=1, pct=True)

        temp2 = self.low - self.avg_price
        temp3 = self.open_price - 0.5 * (self.high + self.low)
        temp2 = temp2 / (temp3 + 0.01)
        temp2 = temp2.iloc[-m:, :]
        part2 = -temp2.apply(lambda x: x * weight2)

        part2 = part2.rank(axis=0, pct=1)
        alpha = part1.iloc[-1, :] + part2.iloc[-1, :]
        alpha = alpha[(alpha < np.inf) & (alpha > -np.inf)]
        alpha = alpha.dropna()
        return alpha

    '''
    ########################################################################
    '''

    def alpha_088(self):
        # (close-delay(close,20))/delay(close,20)*100
        ####################
        data1 = self.close.iloc[-21, :]
        alpha = ((self.close.iloc[-1, :] - data1) / data1) * 100
        alpha = alpha.dropna()
        return alpha

    def alpha_089(self):
        # 2*(sma(close,13,2)-sma(close,27,2)-sma(sma(close,13,2)-sma(close,27,2),10,2))
        ######################
        data1 = pd.ewma(self.close, span=12, adjust=False)
        data2 = pd.ewma(self.close, span=26, adjust=False)
        data3 = pd.ewma(data1 - data2, span=9, adjust=False)
        alpha = ((data1 - data2 - data3) * 2).iloc[-1, :]
        alpha = alpha.dropna()
        return alpha

    def alpha_090(self):
        # (rank(corr(rank(vwap),rank(volume),5))*-1)
        #######################
        data1 = self.avg_price.rank(axis=1, pct=True)
        data2 = self.volume.rank(axis=1, pct=True)
        corr = data1.iloc[-5:, :].corrwith(data2.iloc[-5:, :])
        rank1 = corr.rank(pct=True)
        alpha = rank1 * -1
        alpha = alpha.dropna()
        return alpha

    def alpha_091(self):
        # ((rank((close-max(close,5)))*rank(corr((mean(volume,40)),low,5)))*-1)
        #################
        data1 = self.close
        cond = data1 > 5
        data1[~cond] = 5
        rank1 = ((self.close - data1).rank(axis=1, pct=True)).iloc[-1, :]
        mean = pd.rolling_mean(self.volume, window=40)
        corr = mean.iloc[-5:, :].corrwith(self.low.iloc[-5:, :])
        rank2 = corr.rank(pct=True)
        alpha = rank1 * rank2 * (-1)
        alpha = alpha.dropna()
        return alpha

    #
    def alpha_093(self):
        # SUM((OPEN>=DELAY(OPEN,1)?0:MAX((OPEN-LOW),(OPEN-DELAY(OPEN,1)))),20) #
        cond = self.open_price >= self.open_price.shift()
        data1 = self.open_price - self.low
        data2 = self.open_price - self.open_price.shift()
        cond_max = data1 > data2
        data2[cond_max] = data1[cond_max]
        data2[cond] = 0
        alpha = data2.iloc[-20:, :].sum()
        alpha = alpha.dropna()
        return alpha

    #
    def alpha_094(self):
        # SUM((CLOSE>DELAY(CLOSE,1)?VOLUME:(CLOSE<DELAY(CLOSE,1)?-VOLUME:0)),30) #
        cond1 = self.close > self.prev_close
        cond2 = self.close < self.prev_close
        value = -self.volume
        value[~cond2] = 0
        value[cond1] = self.volume[cond1]
        alpha = value.iloc[-30:, :].sum()
        alpha = alpha.dropna()
        return alpha

    #
    def alpha_095(self):
        # STD(AMOUNT,20) #
        alpha = self.amount.iloc[-20:, :].std()
        alpha = alpha.dropna()
        return alpha

    #
    def alpha_096(self):
        # SMA(SMA((CLOSE-TSMIN(LOW,9))/(TSMAX(HIGH,9)-TSMIN(LOW,9))*100,3,1),3,1) #
        sma1 = pd.ewma(
            100 * (self.close - self.low.rolling(9).min()) / (self.high.rolling(9).max() - self.low.rolling(9).min()),
            span=5, adjust=False)
        alpha = pd.ewma(sma1, span=5, adjust=False).iloc[-1, :]
        alpha = alpha.dropna()
        return alpha

    #
    def alpha_097(self):
        # STD(VOLUME,10) #
        alpha = self.volume.iloc[-10:, :].std()
        alpha = alpha.dropna()
        return alpha

    #
    def alpha_098(self):
        # ((((DELTA((SUM(CLOSE,100)/100),100)/DELAY(CLOSE,100))<0.05)||((DELTA((SUM(CLOSE,100)/100),100)/DELAY(CLOSE,100))==0.05))?(-1*(CLOSE-TSMIN(CLOSE,100))):(-1*DELTA(CLOSE,3))) #
        sum_close = self.close.rolling(100).sum()
        cond = (sum_close / 100 - (sum_close / 100).shift(100)) / self.close.shift(100) <= 0.05
        left_value = -(self.close - self.close.rolling(100).min())
        right_value = -(self.close - self.close.shift(3))
        right_value[cond] = left_value[cond]
        alpha = right_value.iloc[-1, :]
        alpha = alpha.dropna()
        return alpha

    #
    def alpha_099(self):
        # (-1 * RANK(COVIANCE(RANK(CLOSE), RANK(VOLUME), 5))) #
        alpha = (-pd.rolling_cov(self.close.rank(axis=1, pct=True), self.volume.rank(axis=1, pct=True), window=5).rank(
            axis=1, pct=True)).iloc[-1, :]
        alpha = alpha.dropna()
        return alpha

    #
    def alpha_100(self):
        # STD(VOLUME,20) #
        alpha = self.volume.iloc[-20:, :].std()
        alpha = alpha.dropna()
        return alpha

    #
    def alpha_101(self):
        # ((RANK(CORR(CLOSE,SUM(MEAN(VOLUME,30),37),15))<RANK(CORR(RANK(((HIGH*0.1)+(VWAP*0.9))),RANK(VOLUME),11)))*-1) #
        rank1 = (
            self.close.rolling(window=15).corr((self.volume.rolling(window=30).mean()).rolling(window=37).sum())).rank(
            axis=1, pct=True)
        rank2 = (self.high * 0.1 + self.avg_price * 0.9).rank(axis=1, pct=True)
        rank3 = self.volume.rank(axis=1, pct=True)
        rank4 = (rank2.rolling(window=11).corr(rank3)).rank(axis=1, pct=True)
        alpha = -(rank1 < rank4)
        alpha = alpha.iloc[-1, :].dropna()
        return alpha

    #
    def alpha_102(self):
        # SMA(MAX(VOLUME-DELAY(VOLUME,1),0),6,1)/SMA(ABS(VOLUME-DELAY(VOLUME,1)),6,1)*100 #
        max_cond = (self.volume - self.volume.shift()) > 0
        max_data = self.volume - self.volume.shift()
        max_data[~max_cond] = 0
        sma1 = pd.ewma(max_data, span=11, adjust=False)
        sma2 = pd.ewma((self.volume - self.volume.shift()).abs(), span=11, adjust=False)
        alpha = (sma1 / sma2 * 100).iloc[-1, :]
        alpha = alpha.dropna()
        return alpha

    #
    def alpha_104(self):
        # (-1*(DELTA(CORR(HIGH,VOLUME,5),5)*RANK(STD(CLOSE,20)))) #
        corr = self.high.rolling(window=5).corr(self.volume)
        alpha = (-(corr - corr.shift(5)) * ((self.close.rolling(window=20).std()).rank(axis=1, pct=True))).iloc[-1, :]
        alpha = alpha.dropna()
        return alpha

    #
    def alpha_105(self):
        # (-1*CORR(RANK(OPEN),RANK(VOLUME),10)) #
        alpha = -((self.open_price.rank(axis=1, pct=True)).iloc[-10:, :]).corrwith(
            self.volume.iloc[-10:, :].rank(axis=1, pct=True))
        alpha = alpha.dropna()
        return alpha

    #
    def alpha_106(self):
        # CLOSE-DELAY(CLOSE,20) #
        alpha = (self.close - self.close.shift(20)).iloc[-1, :]
        alpha = alpha.dropna()
        return alpha

    #
    def alpha_107(self):
        # (((-1*RANK((OPEN-DELAY(HIGH,1))))*RANK((OPEN-DELAY(CLOSE,1))))*RANK((OPEN-DELAY(LOW,1)))) #
        rank1 = -(self.open_price - self.high.shift()).rank(axis=1, pct=True)
        rank2 = (self.open_price - self.close.shift()).rank(axis=1, pct=True)
        rank3 = (self.open_price - self.low.shift()).rank(axis=1, pct=True)
        alpha = (rank1 * rank2 * rank3).iloc[-1, :]
        alpha = alpha.dropna()
        return alpha

    #
    def alpha_108(self):
        # ((RANK((HIGH-MIN(HIGH,2)))^RANK(CORR((VWAP),(MEAN(VOLUME,120)),6)))*-1) #
        min_cond = self.high > 2
        data = self.high
        data[min_cond] = 2
        rank1 = (self.high - data).rank(axis=1, pct=True)
        rank2 = (self.avg_price.rolling(window=6).corr(self.volume.rolling(window=120).mean())).rank(axis=1, pct=True)
        alpha = (-rank1 ** rank2).iloc[-1, :]
        alpha = alpha.dropna()
        return alpha

    #
    def alpha_109(self):
        # SMA(HIGH-LOW,10,2)/SMA(SMA(HIGH-LOW,10,2),10,2)#
        data = self.high - self.low
        sma1 = pd.ewma(data, span=9, adjust=False)
        sma2 = pd.ewma(sma1, span=9, adjust=False)
        alpha = (sma1 / sma2).iloc[-1, :]
        alpha = alpha.dropna()
        return alpha

    #
    def alpha_110(self):
        # SUM(MAX(0,HIGH-DELAY(CLOSE,1)),20)/SUM(MAX(0,DELAY(CLOSE,1)-LOW),20)*100 #
        data1 = self.high - self.prev_close
        data2 = self.prev_close - self.low
        max_cond1 = data1 < 0
        max_cond2 = data2 < 0
        data1[max_cond1] = 0
        data2[max_cond2] = 0
        sum1 = data1.rolling(window=20).sum()
        sum2 = data2.rolling(window=20).sum()
        alpha = sum1 / sum2 * 100
        alpha = alpha.dropna()
        return alpha.iloc[-1, :]

    def alpha_111(self):
        # sma(vol*((close-low)-(high-close))/(high-low),11,2)-sma(vol*((close-low)-(high-close))/(high-low),4,2)
        ######################
        data1 = self.volume * ((self.close - self.low) - (self.high - self.close)) / (self.high - self.low)
        x = pd.ewma(data1, span=10)
        y = pd.ewma(data1, span=3)
        alpha = (x - y).iloc[-1, :]
        alpha = alpha.dropna()
        return alpha

    #
    def alpha_112(self):
        # (SUM((CLOSE-DELAY(CLOSE,1)>0?CLOSE-DELAY(CLOSE,1):0),12)-SUM((CLOSE-DELAY(CLOSE,1)<0?ABS(CLOSE-DELAY(CLOSE,1)):0),12))/(SUM((CLOSE-DELAY(CLOSE,1)>0?CLOSE-DELAY(CLOSE,1):0),12)+SUM((CLOSE-DELAY(CLOSE,1)<0?ABS(CLOSE-DELAY(CLOSE,1)):0),12))*100 #
        cond1 = self.close > self.prev_close
        cond2 = self.close < self.prev_close
        data1 = self.close - self.prev_close
        data2 = self.close - self.prev_close
        data1[~cond1] = 0
        data2[~cond2] = 0
        data2 = data2.abs()
        sum1 = data1.rolling(window=12).sum()
        sum2 = data2.rolling(window=12).sum()
        alpha = ((sum1 - sum2) / (sum1 + sum2) * 100).iloc[-1, :]
        alpha = alpha.dropna()
        return alpha

    def alpha_113(self):
        # (-1*((rank((sum(delay(close,5),20)/20))*corr(close,volume,2))*rank(corr(sum(close,5),sum(close,20),2))))
        #####################
        data1 = self.close.iloc[:-5, :]
        rank1 = (pd.rolling_sum(data1, window=20) / 20).rank(axis=1, pct=True)
        corr1 = self.close.iloc[-2:, :].corrwith(self.volume.iloc[-2:, :])
        data2 = pd.rolling_sum(self.close, window=5)
        data3 = pd.rolling_sum(self.close, window=20)
        corr2 = data2.iloc[-2:, :].corrwith(data3.iloc[-2:, :])
        rank2 = corr2.rank(axis=0, pct=True)
        alpha = (-1 * rank1 * corr1 * rank2).iloc[-1, :]
        alpha = alpha.dropna()
        return alpha

    def alpha_114(self):
        # ((rank(delay(((high-low)/(sum(close,5)/5)),2))*rank(rank(volume)))/(((high-low)/(sum(close,5)/5))/(vwap-close)))
        #####################
        data1 = (self.high - self.low) / (pd.rolling_sum(self.close, window=5) / 5)
        rank1 = (data1.iloc[-2, :]).rank(axis=0, pct=True)
        rank2 = ((self.volume.rank(axis=1, pct=True)).rank(axis=1, pct=True)).iloc[-1, :]
        data2 = (((self.high - self.low) / (pd.rolling_sum(self.close, window=5) / 5)) / (
                self.avg_price - self.close)).iloc[-1, :]
        alpha = (rank1 * rank2) / data2
        alpha = alpha.dropna()
        return alpha

    #
    def alpha_116(self):
        # REGBETA(CLOSE,SEQUENCE,20) #
        sequence = pd.Series(range(1, 21), index=self.close.iloc[-20:, ].index)  # 1~20
        corr = self.close.iloc[-20:, :].corrwith(sequence)
        alpha = corr
        alpha = alpha.dropna()
        return alpha

    def alpha_117(self):
        #######((tsrank(volume,32)*(1-tsrank(((close+high)-low),16)))*(1-tsrank(ret,32)))
        ####################
        data1 = (self.close + self.high - self.low).iloc[-16:, :]
        data2 = 1 - data1.rank(axis=0, pct=True)
        data3 = (self.volume.iloc[-32:, :]).rank(axis=0, pct=True)
        ret = (self.close / self.close.shift() - 1).iloc[-32:, :]
        data4 = 1 - ret.rank(axis=0, pct=True)
        alpha = (data2.iloc[-1, :]) * (data3.iloc[-1, :]) * (data4.iloc[-1, :])
        alpha = alpha.dropna()
        return alpha

    def alpha_118(self):
        ######sum(high-open,20)/sum((open-low),20)*100
        ###################
        data1 = self.high - self.open_price
        data2 = self.open_price - self.low
        data3 = pd.rolling_sum(data1, window=20)
        data4 = pd.rolling_sum(data2, window=20)
        alpha = ((data3 / data4) * 100).iloc[-1, :]
        alpha = alpha.dropna()
        return alpha

    def alpha_120(self):
        ###############(rank(vwap-close))/(rank(vwap+close))
        ###################
        data1 = (self.avg_price - self.close).rank(axis=1, pct=True)
        data2 = (self.avg_price + self.close).rank(axis=1, pct=True)
        alpha = (data1 / data2).iloc[-1, :]
        alpha = alpha.dropna()
        return alpha

    def alpha_122(self):
        ##### (SMA(SMA(SMA(LOG(CLOSE),13,2),13,2),13,2) - DELAY(SMA(SMA(SMA(LOG(CLOSE),13,2),13,2),13,2),1))
        ##### / DELAY(SMA(SMA(SMA(LOG(CLOSE),13,2),13,2),13,2),1)
        ##
        log_close = np.log(self.close)
        data = pd.ewma(pd.ewma(pd.ewma(log_close, span=12, adjust=False), span=12, adjust=False), span=12, adjust=False)
        alpha = (data.iloc[-1, :] / data.iloc[-2, :]) - 1
        alpha = alpha.dropna()
        return alpha

    def alpha_123(self):
        #####((RANK(CORR(SUM(((HIGH+LOW)/2), 20), SUM(MEAN(VOLUME, 60), 20), 9)) < RANK(CORR(LOW, VOLUME, 6))) * -1)
        ##
        data1 = ((self.high + self.low) / 2).rolling(20).sum()
        data2 = self.volume.rolling(60).mean().rolling(20).sum()
        rank1 = data1.iloc[-9:, :].corrwith(data2.iloc[-9:, :]).dropna().rank(axis=0, pct=True)
        rank2 = self.low.iloc[-6:, :].corrwith(self.volume.iloc[-6:, :]).dropna().rank(axis=0, pct=True)
        rank1 = rank1[rank1.index.isin(rank2.index)]
        rank2 = rank2[rank2.index.isin(rank1.index)]
        alpha = (rank1 < rank2) * (-1)
        alpha = alpha.dropna()
        return alpha

    def alpha_124(self):
        ##### (CLOSE - VWAP) / DECAYLINEAR(RANK(TSMAX(CLOSE, 30)),2)
        ##
        data1 = self.close.rolling(30).max().rank(axis=1, pct=True)
        alpha = (self.close.iloc[-1, :] - self.avg_price.iloc[-1, :]) / (
                2. / 3 * data1.iloc[-2, :] + 1. / 3 * data1.iloc[-1, :])
        alpha = alpha.dropna()
        return alpha

    def alpha_125(self):
        ##### (RANK(DECAYLINEAR(CORR((VWAP), MEAN(VOLUME, 80), 17), 20)) / RANK(DECAYLINEAR(DELTA((CLOSE * 0.5 + VWAP * 0.5), 3), 16)))
        ##
        data1 = pd.rolling_corr(self.avg_price, self.volume.rolling(80).mean(), window=17)
        decay_weights = np.arange(1, 21, 1)[::-1]  # 倒序数组
        decay_weights = decay_weights / decay_weights.sum()
        rank1 = data1.iloc[-20:, :].mul(decay_weights, axis=0).sum().rank(axis=0, pct=True)

        data2 = (self.close * 0.5 + self.avg_price * 0.5).diff(3)
        decay_weights = np.arange(1, 17, 1)[::-1]  # 倒序数组
        decay_weights = decay_weights / decay_weights.sum()
        rank2 = data2.iloc[-16:, :].mul(decay_weights, axis=0).sum().rank(axis=0, pct=True)

        alpha = rank1 / rank2
        alpha = alpha.dropna()
        return alpha

    def alpha_126(self):
        #### (CLOSE + HIGH + LOW) / 3
        ##
        alpha = (self.close.iloc[-1, :] + self.high.iloc[-1, :] + self.low.iloc[-1, :]) / 3
        alpha = alpha.dropna()
        return alpha

    def alpha_129(self):
        #### SUM((CLOSE - DELAY(CLOSE, 1) < 0 ? ABS(CLOSE - DELAY(CLOSE, 1)):0), 12)
        ##
        data = self.close.diff(1)
        data[data >= 0] = 0
        data = abs(data)
        alpha = data.iloc[-12:, :].sum()
        alpha = alpha.dropna()
        return alpha

    def alpha_130(self):
        #### alpha_130
        #### (RANK(DELCAYLINEAR(CORR(((HIGH + LOW) / 2), MEAN(VOLUME, 40), 9), 10)) / RANK(DELCAYLINEAR(CORR(RANK(VWAP), RANK(VOLUME), 7), 3)))
        ##
        data1 = (self.high + self.low) / 2
        data2 = self.volume.rolling(40).mean()
        data3 = pd.rolling_corr(data1, data2, window=9)
        decay_weights = np.arange(1, 11, 1)[::-1]  # 倒序数组
        decay_weights = decay_weights / decay_weights.sum()
        rank1 = data3.iloc[-10:, :].mul(decay_weights, axis=0).sum().rank(axis=0, pct=True)

        data1 = self.avg_price.rank(axis=1, pct=True)
        data2 = self.volume.rank(axis=1, pct=True)
        data3 = pd.rolling_corr(data1, data2, window=7)
        decay_weights = np.arange(1, 4, 1)[::-1]  # 倒序数组
        decay_weights = decay_weights / decay_weights.sum()
        rank2 = data3.iloc[-3:, :].mul(decay_weights, axis=0).sum().rank(axis=0, pct=True)

        alpha = (rank1 / rank2).dropna()
        return alpha

    def alpha_132(self):
        #### MEAN(AMOUNT, 20)
        ##
        alpha = self.amount.iloc[-20:, :].mean()
        alpha = alpha.dropna()
        return alpha

    def alpha_134(self):
        #### (CLOSE - DELAY(CLOSE, 12)) / DELAY(CLOSE, 12) * VOLUME
        ##
        alpha = ((self.close.iloc[-1, :] / self.close.iloc[-13, :] - 1) * self.volume.iloc[-1, :])
        alpha = alpha.dropna()
        return alpha

    def alpha_135(self):
        #### SMA(DELAY(CLOSE / DELAY(CLOSE, 20), 1), 20, 1)
        ##
        def rolling_div(na):
            return na[-1] / na[-21]

        data1 = self.close.rolling(21).apply(rolling_div).shift(periods=1)
        alpha = pd.ewma(data1, com=19, adjust=False).iloc[-1, :]
        alpha = alpha.dropna()
        return alpha

    def alpha_136(self):
        #### ((-1 * RANK(DELTA(RET, 3))) * CORR(OPEN, VOLUME, 10))
        ##
        data1 = -(self.close / self.prev_close - 1).diff(3).rank(axis=1, pct=True)
        data2 = self.open_price.iloc[-10:, :].corrwith(self.volume.iloc[-10:, :])
        alpha = (data1.iloc[-1, :] * data2).dropna()

        return alpha

    #
    def alpha_139(self):
        #### (-1 * CORR(OPEN, VOLUME, 10))
        ##
        alpha = - self.open_price.iloc[-10:, :].corrwith(self.volume.iloc[-10:, :]).dropna()
        return alpha

    def alpha_141(self):
        #### (RANK(CORR(RANK(HIGH), RANK(MEAN(VOLUME, 15)), 9))* -1)
        ##
        df1 = self.high.rank(axis=1, pct=True)
        df2 = self.volume.rolling(15).mean().rank(axis=1, pct=True)
        alpha = -df1.iloc[-9:, :].corrwith(df2.iloc[-9:, :]).rank(pct=True)
        alpha = alpha.dropna()
        return alpha

    def alpha_142(self):
        #### (((-1 * RANK(TSRANK(CLOSE, 10))) * RANK(DELTA(DELTA(CLOSE, 1), 1))) * RANK(TSRANK((VOLUME/MEAN(VOLUME, 20)), 5)))
        ##

        rank1 = self.close.iloc[-10:, :].rank(axis=0, pct=True).iloc[-1, :].rank(pct=True)
        rank2 = self.close.diff(1).diff(1).iloc[-1, :].rank(pct=True)
        rank3 = (self.volume / self.volume.rolling(20).mean()).iloc[-5:, :].rank(axis=0, pct=True).iloc[-1, :].rank(
            pct=True)

        alpha = -(rank1 * rank2 * rank3).dropna()
        alpha = alpha.dropna()
        return alpha

    def alpha_144(self):
        #### SUMIF(ABS(CLOSE/DELAY(CLOSE, 1) - 1)/AMOUNT, 20, CLOSE < DELAY(CLOSE, 1))/COUNT(CLOSE < DELAY(CLOSE, 1), 20)
        ##
        df1 = self.close < self.prev_close
        sumif = ((abs(self.close / self.prev_close - 1) / self.amount) * df1).iloc[-20:, :].sum()
        count = df1.iloc[-20:, :].sum()

        alpha = (sumif / count).dropna()
        alpha = alpha.dropna()
        return alpha

    def alpha_145(self):
        #### (MEAN(VOLUME, 9) - MEAN(VOLUME, 26)) / MEAN(VOLUME, 12) * 100
        ##

        alpha = (self.volume.iloc[-9:, :].mean() - self.volume.iloc[-26:, :].mean()) / self.volume.iloc[-12:,
                                                                                       :].mean() * 100
        alpha = alpha.dropna()
        return alpha

    def alpha_148(self):
        #### ((RANK(CORR((OPEN), SUM(MEAN(VOLUME, 60), 9), 6)) < RANK((OPEN - TSMIN(OPEN, 14)))) * -1)
        ##
        df1 = self.volume.rolling(60).mean().rolling(9).sum()
        rank1 = self.open_price.iloc[-6:, :].corrwith(df1.iloc[-6:, :]).rank(pct=True)
        rank2 = (self.open_price - self.open_price.rolling(14).min()).iloc[-1, :].rank(pct=True)

        alpha = -1 * (rank1 < rank2)
        alpha = alpha.dropna()
        return alpha

    def alpha_150(self):
        #### (CLOSE + HIGH + LOW)/3 * VOLUME
        ##

        alpha = ((self.close + self.high + self.low) / 3 * self.volume).iloc[-1, :]
        alpha = alpha.dropna()
        return alpha

    ######################## alpha_152 #######################
    #
    def alpha_152(self):
        # SMA(MEAN(DELAY(SMA(DELAY(CLOSE/DELAY(CLOSE,9),1),9,1),1),12)-MEAN(DELAY(SMA(DELAY(CLOSE/DELAY(CLOSE,9),1),9,1),1),26),9,1) #
        #
        data1 = pd.rolling_mean((pd.ewma(((self.close / self.close.shift(9)).shift()), span=17, adjust=False)).shift(),
                                12)
        data2 = pd.rolling_mean((pd.ewma(((self.close / self.close.shift(9)).shift()), span=17, adjust=False)).shift(),
                                26)
        alpha = (pd.ewma(data1 - data2, span=17, adjust=False)).iloc[-1, :]
        alpha = alpha.dropna()
        return alpha

    ######################## alpha_153 #######################
    #
    def alpha_153(self):
        # (MEAN(CLOSE,3)+MEAN(CLOSE,6)+MEAN(CLOSE,12)+MEAN(CLOSE,24))/4 #
        alpha = ((pd.rolling_mean(self.close, 3) + pd.rolling_mean(self.close, 6) + pd.rolling_mean(self.close,
                                                                                                    12) + pd.rolling_mean(
            self.close, 24)) / 4).iloc[-1, :]
        alpha = alpha.dropna()
        return alpha

    ######################## alpha_154 #######################
    #
    def alpha_154(self):
        # (((VWAP-MIN(VWAP,16)))<(CORR(VWAP,MEAN(VOLUME,180),18))) #
        alpha = (self.avg_price - pd.rolling_min(self.avg_price, 16)).iloc[-1, :] < self.avg_price.iloc[-18:,
                                                                                    :].corrwith(
            (pd.rolling_mean(self.volume, 180)).iloc[-18:, :])
        alpha = alpha.dropna()
        return alpha

    ######################## alpha_155 #######################
    #
    def alpha_155(self):
        # SMA(VOLUME,13,2)-SMA(VOLUME,27,2)-SMA(SMA(VOLUME,13,2)-SMA(VOLUME,27,2),10,2) #
        sma1 = pd.ewma(self.volume, span=12, adjust=False)
        sma2 = pd.ewma(self.volume, span=26, adjust=False)
        sma = pd.ewma(sma1 - sma2, span=9, adjust=False)
        alpha = sma.iloc[-1, :]
        alpha = alpha.dropna()
        return alpha

    ######################## alpha_158 #######################
    #
    def alpha_158(self):
        # ((HIGH-SMA(CLOSE,15,2))-(LOW-SMA(CLOSE,15,2)))/CLOSE #
        alpha = (((self.high - pd.ewma(self.close, span=14, adjust=False)) - (
                self.low - pd.ewma(self.close, span=14, adjust=False))) / self.close).iloc[-1, :]
        alpha = alpha.dropna()
        return alpha

    def alpha_159(self):
        #########((close-sum(min(low,delay(close,1)),6))/sum(max(high,delay(close,1))-min(low,delay(close,1)),6)*12*24+(close-sum(min(low,delay(close,1)),12))/sum(max(high,delay(close,1))-min(low,delay(close,1)),12)*6*24+(close-sum(min(low,delay(close,1)),24))/sum(max(high,delay(close,1))-min(low,delay(close,1)),24)*6*24)*100/(6*12+6*24+12*24)
        ###################
        data1 = self.low
        data2 = self.close.shift()
        cond = data1 > data2
        data1[cond] = data2
        data3 = self.high
        data4 = self.close.shift()
        cond = data3 > data4
        data3[~cond] = data4
        # 计算出公式核心部分x
        x = ((self.close - pd.rolling_sum(data1, 6)) / pd.rolling_sum((data2 - data1), 6)) * 12 * 24
        # 计算出公式核心部分y
        y = ((self.close - pd.rolling_sum(data1, 12)) / pd.rolling_sum((data2 - data1), 12)) * 6 * 24
        # 计算出公式核心部分z
        z = ((self.close - pd.rolling_sum(data1, 24)) / pd.rolling_sum((data2 - data1), 24)) * 6 * 24
        data5 = (x + y + z) * (100 / (6 * 12 + 12 * 24 + 6 * 24))
        alpha = data5.iloc[-1, :]
        alpha = alpha.dropna()
        return alpha

    def alpha_160(self):
        ################
        ############sma((close<=delay(close,1)?std(close,20):0),20,1)
        data1 = pd.rolling_std(self.close, 20)
        cond = self.close <= self.close.shift(0)
        data1[~cond] = 0
        data2 = pd.ewma(data1, span=39)
        alpha = data2.iloc[-1, :]
        alpha = alpha.dropna()
        return alpha

    def alpha_161(self):
        ###########mean((max(max(high-low),abs(delay(close,1)-high)),abs(delay(close,1)-low)),12)
        ################
        data1 = (self.high - self.low)
        data2 = pd.Series.abs(self.close.shift() - self.high)
        cond = data1 > data2
        data1[~cond] = data2
        data3 = pd.Series.abs(self.close.shift() - self.low)
        cond = data1 > data3
        data1[~cond] = data3
        alpha = (pd.rolling_mean(data1, 12)).iloc[-1, :]
        alpha = alpha.dropna()
        return alpha

    def alpha_162(self):
        ###############(sma(max(close-delay(close,1),0),12,1)/sma(abs(close-delay(close,1)),12,1)*100-min(sma(max(close-delay(close,1),0),12,1)/sma(abs(close-delay(close,1)),12,1)*100,12))/(max(sma(max(close-delay(close,1),0),12,1)/sma(abs(close-delay(close,1)),12,1)*100),12)-min(sma(max(close-delay(close,1),0),12,1)/sma(abs(close-delay(close,1)),12,1)*100),12))
        #################
        # 算出公式核心部分X
        data1 = self.close - self.close.shift()
        cond = data1 > 0
        data1[~cond] = 0
        x = pd.ewma(data1, span=23)
        data2 = pd.Series.abs(self.close - self.close.shift())
        y = pd.ewma(data2, span=23)
        z = (x / y) * 100
        cond = z > 12
        z[cond] = 12
        c = (x / y) * 100
        cond = c > 12
        c[~cond] = 12
        data3 = (x / y) * 100 - (z / c) - c
        alpha = data3.iloc[-1, :]
        alpha = alpha.dropna()
        return alpha

    def alpha_163(self):
        ################
        #######rank(((((-1*ret)*,ean(volume,20))*vwap)*(high-close)))
        data1 = (-1) * (self.close / self.close.shift() - 1) * pd.rolling_mean(self.volume, 20) * self.avg_price * (
                self.high - self.close)
        data2 = (data1.rank(axis=1, pct=True)).iloc[-1, :]
        alpha = data2
        alpha = alpha.dropna()
        return alpha

    def alpha_164(self):
        ################
        ############sma((((close>delay(close,1))?1/(close-delay(close,1)):1)-min(((close>delay(close,1))?1/(close/delay(close,1)):1),12))/(high-low)*100,13,2)
        cond = self.close > self.close.shift()
        data1 = 1 / (self.close - self.close.shift())
        data1[~cond] = 1
        data2 = 1 / (self.close - self.close.shift())
        cond = data2 > 12
        data2[cond] = 12
        data3 = data1 - data2 / ((self.high - self.low) * 100)
        alpha = (pd.ewma(data3, span=12)).iloc[-1, :]
        alpha = alpha.dropna()
        return alpha

    def alpha_167(self):
        ##
        ####sum(((close-delay(close,1)>0)?(close-delay(close,1)):0),12)####
        data1 = self.close - self.close.shift()
        cond = (data1 < 0)
        data1[cond] = 0
        data2 = (pd.rolling_sum(data1, 12)).iloc[-1, :]
        alpha = data2
        alpha = alpha.dropna()
        return alpha

    def alpha_168(self):
        ##
        #####-1*volume/mean(volume,20)####
        data1 = (-1 * self.volume) / pd.rolling_mean(self.volume, 20)
        alpha = data1.iloc[-1, :]
        alpha = alpha.dropna()
        return alpha

    def alpha_169(self):
        ##
        ###sma(mean(delay(sma(close-delay(close,1),9,1),1),12)-mean(delay(sma(close-delay(close,1),1,1),1),26),10,1)#####
        data1 = self.close - self.close.shift()
        data2 = (pd.ewma(data1, span=17)).shift()
        data3 = pd.rolling_mean(data2, 12) - pd.rolling_mean(data2, 26)
        alpha = (pd.ewma(data3, span=19)).iloc[-1, :]
        alpha = alpha.dropna()
        return alpha

    def alpha_170(self):
        ##
        #####((((rank((1/close))*volume)/mean(volume,20))*((high*rank((high-close)))/(sum(high,5)/5)))-rank((vwap-delay(vwap,5))))####
        data1 = (1 / self.close).rank(axis=0, pct=True)
        data2 = pd.rolling_mean(self.volume, 20)
        x = (data1 * self.volume) / data2
        data3 = (self.high - self.close).rank(axis=0, pct=True)
        data4 = pd.rolling_mean(self.high, 5)
        y = (data3 * self.high) / data4
        z = (self.avg_price.iloc[-1, :] - self.avg_price.iloc[-5, :]).rank(axis=0, pct=True)
        alpha = (x * y - z).iloc[-1, :]
        alpha = alpha.dropna()
        return alpha

    def alpha_171(self):
        ##
        ####(((low-close)*open^5)*-1)/((close-high)*close^5)#####
        data1 = -1 * (self.low - self.close) * (self.open_price ** 5)
        data2 = (self.close - self.high) * (self.close ** 5)
        alpha = (data1 / data2).iloc[-1, :]
        alpha = alpha.dropna()
        return alpha

    #
    def alpha_172(self):
        # MEAN(ABS(SUM((LD>0&LD>HD)?LD:0,14)*100/SUM(TR,14)-SUM((HD>0&HD>LD)?HD:0,14)*100/(SUM((LD>0&LD>HD)?LD:0,14)*100/SUM(TR,14)+SUM(TR,14)+SUM((HD>0&HD>LD)?HD:0,14)*100/SUM(TR,14))*100,6) #
        hd = self.high - self.high.shift()
        ld = self.low.shift() - self.low
        temp1 = self.high - self.low
        temp2 = (self.high - self.close.shift()).abs()
        cond1 = temp1 > temp2
        temp2[cond1] = temp1[cond1]
        temp3 = (self.low - self.close.shift()).abs()
        cond2 = temp2 > temp3
        temp3[cond2] = temp2[cond2]
        tr = temp3  # MAX(MAX(HIGH-LOW,ABS(HIGH-DELAY(CLOSE,1))),ABS(LOW-DELAY(CLOSE,1)))
        sum_tr14 = pd.rolling_sum(tr, 14)
        cond3 = ld > 0
        cond4 = ld > hd
        cond3[~cond4] = False
        data1 = ld
        data1[~cond3] = 0
        sum1 = pd.rolling_sum(data1, 14) * 100 / sum_tr14
        cond5 = hd > 0
        cond6 = hd > ld
        cond5[~cond6] = False
        data2 = hd
        data2[~cond5] = 0
        sum2 = pd.rolling_sum(data2, 14) * 100 / sum_tr14
        alpha = pd.rolling_mean((sum1 - sum2).abs() / (sum1 + sum2) * 100, 6).iloc[-1, :]
        alpha = alpha.dropna()
        return alpha

    def alpha_173(self):
        ##
        ####3*sma(close,13,2)-2*sma(sma(close,13,2),13,2)+sma(sma(sma(log(close),13,2),13,2),13,2)#####
        data1 = pd.ewma(self.close, span=12)
        data2 = pd.ewma(data1, span=12)
        close_log = np.log(self.close)
        data3 = pd.ewma(close_log, span=12)
        data4 = pd.ewma(data3, span=12)
        data5 = pd.ewma(data4, span=12)
        alpha = (3 * data1 - 2 * data2 + data5).iloc[-1, :]
        alpha = alpha.dropna()
        return alpha

    def alpha_174(self):
        ##
        ####sma((close>delay(close,1)?std(close,20):0),20,1)#####
        cond = self.close > self.prev_close
        data2 = pd.rolling_std(self.close, 20)
        data2[~cond] = 0
        alpha = (pd.ewma(data2, span=39, adjust=False)).iloc[-1, :]
        alpha = alpha.dropna()
        return alpha

    def alpha_176(self):
        ##
        ######### #########corr(rank((close-tsmin(low,12))/(tsmax(high,12)-tsmin(low,12))),rank(volume),6)#############
        data1 = (self.close - pd.rolling_min(self.low, window=12)) / (
                pd.rolling_max(self.high, window=12) - pd.rolling_min(self.low, window=12))
        data2 = data1.rank(axis=0, pct=True)
        # 获取数据求出rank2
        data3 = self.volume.rank(axis=0, pct=True)
        corr = data2.iloc[-6:, :].corrwith(data3.iloc[-6:, :])
        alpha = corr
        alpha = alpha.dropna()
        return alpha

    def alpha_178(self):
        ##### (close-delay(close,1))/delay(close,1)*volume ####
        ##
        alpha = ((self.close - self.close.shift()) / self.close.shift() * self.volume).iloc[-1, :]
        alpha = alpha.dropna()
        return alpha

    def alpha_179(self):
        #####（rank(corr(vwap,volume,4))*rank(corr(rank(low),rank(mean(volume,50)),12))####
        ##
        rank1 = (self.avg_price.iloc[-4:, :].corrwith(self.volume.iloc[-4:, :])).rank(axis=0, pct=True)
        data2 = self.low.rank(axis=0, pct=True)
        data3 = (pd.rolling_mean(self.volume, window=50)).rank(axis=0, pct=True)
        rank2 = (data2.iloc[-12:, :].corrwith(data3.iloc[-12:, :])).rank(axis=0, pct=True)
        alpha = rank1 * rank2
        alpha = alpha.dropna()
        return alpha

        ##################### alpha_180 #######################

    #
    def alpha_180(self):
        ##### ((MEAN(VOLUME,20)<VOLUME)?((-1*TSRANK(ABS(DELTA(CLOSE,7)),60))*SIGN(DELTA(CLOSE,7)):(-1*VOLUME))) #####
        ma = pd.rolling_mean(self.volume, window=20)
        cond = (ma < self.volume).iloc[-20:, :]
        sign = delta_close_7 = self.close.diff(7)
        sign[sign.iloc[:, :] < 0] = -1
        sign[sign.iloc[:, :] > 0] = 1
        sign[sign.iloc[:, :] == 0] = 0
        left = (((self.close.diff(7).abs()).iloc[-60:, :].rank(axis=0, pct=True) * (-1)).iloc[-20:, :] * sign.iloc[-20:,
                                                                                                         :]).iloc[-20:,
               :]
        right = self.volume.iloc[-20:, :] * (-1)
        right[cond] = left[cond]
        alpha = right.iloc[-1, :]
        alpha = alpha.dropna()
        return alpha

    ######################## alpha_182 #######################
    #
    def count_cond_182(self, x):
        num = 0
        for i in x:
            if i == np.True_:
                num += 1
        return num

    def alpha_184(self):
        #####(rank(corr(delay((open-close),1),close,200))+rank((open-close))) ####
        ##
        data1 = self.open_price.shift() - self.close.shift()
        data2 = self.open_price.iloc[-1, :] - self.close.iloc[-1, :]
        corr = data1.iloc[-200:, :].corrwith(self.close.iloc[-200:, :])
        alpha = data2.rank(axis=0, pct=True) + corr.rank(axis=0, pct=True)
        alpha = alpha.dropna()
        return alpha

    def alpha_185(self):
        ##### RANK((-1 * ((1 - (OPEN / CLOSE))^2))) ####
        alpha = (-(1 - self.open_price / self.close) ** 2).rank(axis=1, pct=True).iloc[-1, :]
        alpha = alpha.dropna()
        return alpha

    #
    def alpha_186(self):
        # (MEAN(ABS(SUM((LD>0 & LD>HD)?LD:0,14)*100/SUM(TR,14)-SUM((HD>0 & HD>LD)?HD:0,14)*100/SUM(TR,14))/(SUM((LD>0 & LD>HD)?LD:0,14)*100/SUM(TR,14)+SUM((HD>0 & HD>LD)?HD:0,14)*100/SUM(TR,14))*100,6)+DELAY(MEAN(ABS(SUM((LD>0 & LD>HD)?LD:0,14)*100/SUM(TR,14)-SUM((HD>0 & HD>LD)?HD:0,14)*100/SUM(TR,14))/(SUM((LD>0 & LD>HD)?LD:0,14)*100/SUM(TR,14)+SUM((HD>0 & HD>LD)?HD:0,14)*100/SUM(TR,14))*100,6),6))/2 #
        hd = self.high - self.high.shift()
        ld = self.low.shift() - self.low
        temp1 = self.high - self.low
        temp2 = (self.high - self.close.shift()).abs()
        cond1 = temp1 > temp2
        temp2[cond1] = temp1[cond1]
        temp3 = (self.low - self.close.shift()).abs()
        cond2 = temp2 > temp3
        temp3[cond2] = temp2[cond2]
        tr = temp3  # MAX(MAX(HIGH-LOW,ABS(HIGH-DELAY(CLOSE,1))),ABS(LOW-DELAY(CLOSE,1)))
        sum_tr14 = pd.rolling_sum(tr, 14)
        cond3 = ld > 0
        cond4 = ld > hd
        cond3[~cond4] = False
        data1 = ld
        data1[~cond3] = 0
        sum1 = pd.rolling_sum(data1, 14) * 100 / sum_tr14
        cond5 = hd > 0
        cond6 = hd > ld
        cond5[~cond6] = False
        data2 = hd
        data2[~cond5] = 0
        sum2 = pd.rolling_sum(data2, 14) * 100 / sum_tr14
        mean1 = pd.rolling_mean((sum1 - sum2).abs() / (sum1 + sum2) * 100, 6)
        alpha = ((mean1 + mean1.shift(6)) / 2).iloc[-1, :]
        alpha = alpha.dropna()
        return alpha

    def alpha_187(self):
        ##### SUM((OPEN<=DELAY(OPEN,1)?0:MAX((HIGH-OPEN),(OPEN-DELAY(OPEN,1)))),20) ####

        cond = (self.open_price <= self.open_price.shift())
        data1 = self.high - self.low  # HIGH-LOW
        data2 = self.open_price - self.open_price.shift()  # OPEN-DELAY(OPEN,1)
        cond_max = data2 > data1
        data1[cond_max] = data2[cond_max]
        data1[cond] = 0
        alpha = data1.iloc[-20:, :].sum()
        alpha = alpha.dropna()
        return alpha

    def alpha_188(self):
        ##### ((HIGH-LOW–SMA(HIGH-LOW,11,2))/SMA(HIGH-LOW,11,2))*100 #####

        sma = pd.ewma(self.high - self.low, span=10, adjust=False)
        alpha = ((self.high - self.low - sma) / sma * 100).iloc[-1, :]
        alpha = alpha.dropna()
        return alpha

    def alpha_189(self):
        ##### mean(abs(close-mean(close,6),6)) ####
        ma6 = pd.rolling_mean(self.close, window=6)
        alpha = pd.rolling_mean((self.close - ma6).abs(), window=6).iloc[-1, :]
        alpha = alpha.dropna()
        return alpha

    def alpha_191(self):
        ##### (CORR(MEAN(VOLUME,20), LOW, 5) + ((HIGH + LOW) / 2)) - CLOSE ####
        # /chencheng
        volume_avg = pd.rolling_mean(self.volume, window=20)
        corr = volume_avg.iloc[-5:, :].corrwith(self.low.iloc[-5:, :])
        alpha = corr + (self.high.iloc[-1, :] + self.low.iloc[-1, :]) / 2 - self.close.iloc[-1, :]
        alpha = alpha.dropna()
        return alpha


if __name__ == '__main__':
    g = Generator(51)
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

    # d = pd.DataFrame(list(zip(a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14, a15)))
    # d.columns = ['alpha1', 'alpha2', 'alpha3', 'alpha4', 'alpha5', 'alpha6', 'alpha7', 'alpha8', 'alpha9', 'alpha10',
    #              'alpha11', 'alpha12', 'alpha13', 'alpha14', 'alpha15']
    # # d.reset_index(g.open_price.columns.values)
    # # d.set_index(g.open_price.columns.values)
    # d.index = g.open_price.columns.values
    # print(d)
    # a1.set_index('alpha')
    a_date = pd.Series(g.end_date, index=None)
    a_date.index=['a']
    b = pd.concat(
        [a_date, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14, a15, a16, a17, a18, a19, a20, a21, a22,
         a23, a24,
         a25, a26, a28, a29, a31, a32, a33, a34, a35, a36, a37, a38, a39, a40, a41, a42, a43, a44, a45, a46,
         a47, a48, a49, a52, a53, a54, a56, a57, a58, a59, a60, a61, a62, a63, a64, a65, a66, a67, a68, a70, a71, a72,
         a74, a76, a77, a78, a79, a80, a81, a82, a83, a84, a85, a86, a87, a89, a90, a91, a93, a94,
         a95, a96, a97, a98, a99, a100, a101, a102, a104, a105, a106, a107, a108, a109, a110, a111, a112,
         a113, a114, a116, a117, a118, a120, a122, a123, a124, a125, a126, a129, a130,
         a134, a135, a136, a139, a141, a142, a144, a145, a148, a150, a152, a153,
         a154, a155, a158, a159, a160, a161, a162, a163, a164, a167, a168, a169, a170, a171,
         a172, a173, a174, a176, a178, a179, a180, a184, a185, a186, a187, a188, a189, a191],
        axis=1)

    b.columns = ['Date', 'alpha1', 'alpha2', 'alpha3', 'alpha4', 'alpha5', 'alpha6', 'alpha7', 'alpha8', 'alpha9',
                 'alpha10',
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
                 'alpha104', 'alpha105', 'alpha106', 'alpha107', 'alpha108', 'alpha109', 'alpha110', 'alpha111',
                 'alpha112', 'alpha113', 'alpha114', 'alpha116', 'alpha117', 'alpha118', 'alpha120', 'alpha122',
                 'alpha123', 'alpha124', 'alpha125', 'alpha126', 'alpha129', 'alpha130', 'alpha134',
                 'alpha135', 'alpha136', 'alpha139', 'alpha141', 'alpha142', 'alpha144', 'alpha145', 'alpha148',
                 'alpha150', 'alpha152', 'alpha153', 'alpha154', 'alpha155', 'alpha158', 'alpha159', 'alpha160',
                 'alpha161', 'alpha162', 'alpha163', 'alpha164', 'alpha167', 'alpha168', 'alpha169', 'alpha170',
                 'alpha171', 'alpha172', 'alpha173', 'alpha174', 'alpha176', 'alpha178', 'alpha179', 'alpha180',
                 'alpha184', 'alpha185', 'alpha186', 'alpha187', 'alpha188', 'alpha189', 'alpha191']

    b.to_csv('sh191_2.csv', index=None)

# if __name__ == '__main__':
#     # df = pd.read_csv('sh.csv', index_col=0)
# '''
#     df['alpha001'] = stock.alpha_001()
#     df['alpha002'] = stock.alpha_002()
#     df['alpha003'] = stock.alpha_003()
#     df['alpha004'] = stock.alpha_004()
#     df['alpha005'] = stock.alpha_005()
#     df['alpha006'] = stock.alpha_006()
#     df['alpha007'] = stock.alpha_007()
#     df['alpha008'] = stock.alpha_008()
#     df['alpha009'] = stock.alpha_009()
#     df['alpha010'] = stock.alpha_010()
#     df['alpha011'] = stock.alpha_011()
#     df['alpha012'] = stock.alpha_012()
#     df['alpha013'] = stock.alpha_013()
#     df['alpha014'] = stock.alpha_014()
#     df['alpha015'] = stock.alpha_015()
#     df['alpha016'] = stock.alpha_016()
#     df['alpha017'] = stock.alpha_017()
#     df['alpha018'] = stock.alpha_018()
#     df['alpha019'] = stock.alpha_019()
#     df['alpha020'] = stock.alpha_020()
#     df['alpha021'] = stock.alpha_021()
#     df['alpha022'] = stock.alpha_022()
#     df['alpha023'] = stock.alpha_023()
#     df['alpha024'] = stock.alpha_024()
#     df['alpha025'] = stock.alpha_025()
#     df['alpha026'] = stock.alpha_026()
#     df['alpha027'] = stock.alpha_027()
#     df['alpha028'] = stock.alpha_028()
#     df['alpha029'] = stock.alpha_029()
#     df['alpha030'] = stock.alpha_030()
#     df['alpha031'] = stock.alpha_031()
#     df['alpha032'] = stock.alpha_032()
#     df['alpha033'] = stock.alpha_033()
#     df['alpha034'] = stock.alpha_034()
#     df['alpha035'] = stock.alpha_035()
#     df['alpha036'] = stock.alpha_036()
#     df['alpha037'] = stock.alpha_037()
#     df['alpha038'] = stock.alpha_038()
#     df['alpha039'] = stock.alpha_039()
#     df['alpha040'] = stock.alpha_040()
#     df['alpha041'] = stock.alpha_041()
#     df['alpha042'] = stock.alpha_042()
#     df['alpha043'] = stock.alpha_043()
#     df['alpha044'] = stock.alpha_044()
#     df['alpha045'] = stock.alpha_045()
#     df['alpha046'] = stock.alpha_046()
#     df['alpha047'] = stock.alpha_047()
#     df['alpha049'] = stock.alpha_049()
#     # df['alpha050']=stock.alpha_050()
#     # df['alpha051']=stock.alpha_051()
#     df['alpha052'] = stock.alpha_052()
#     df['alpha053'] = stock.alpha_053()
#     df['alpha054'] = stock.alpha_054()
#     # df['alpha055']=stock.alpha_055()
#     df['alpha057'] = stock.alpha_057()
#     df['alpha060'] = stock.alpha_060()
#     df['alpha061'] = stock.alpha_061()
#     df['alpha062'] = stock.alpha_062()
#     df['alpha064'] = stock.alpha_064()
#     df['alpha065'] = stock.alpha_065()
#     df['alpha066'] = stock.alpha_066()
#     df['alpha068'] = stock.alpha_068()
#     df['alpha071'] = stock.alpha_071()
#     df['alpha072'] = stock.alpha_072()
#     # df['alpha073']=stock.alpha_073()
#     df['alpha074'] = stock.alpha_074()
#     # df['alpha075']=stock.alpha_075()
#     df['alpha077'] = stock.alpha_077()
#     df['alpha078'] = stock.alpha_078()
#     df['alpha081'] = stock.alpha_081()
#     df['alpha083'] = stock.alpha_083()
#     df['alpha084'] = stock.alpha_084()
#     df['alpha085'] = stock.alpha_085()
#     df['alpha086'] = stock.alpha_086()
#     df['alpha088'] = stock.alpha_088()
#     # df['alpha092']=stock.alpha_092()
#     df['alpha094'] = stock.alpha_094()
#     df['alpha095'] = stock.alpha_095()
#     df['alpha096'] = stock.alpha_096()
#     df['alpha098'] = stock.alpha_098()
#     df['alpha099'] = stock.alpha_099()
#     df['alpha101'] = stock.alpha_101()
#     df['alpha102'] = stock.alpha_102()
#     df['alpha104'] = stock.alpha_104()
#     df['alpha105'] = stock.alpha_105()
#     df['alpha106'] = stock.alpha_106()
#     df['alpha107'] = stock.alpha_107()
#     df['alpha108'] = stock.alpha_108()
#     df['alpha109'] = stock.alpha_109()
#     df['alpha110'] = stock.alpha_110()
#     df['alpha111'] = stock.alpha_111()
#     df['alpha112'] = stock.alpha_112()
#     df['alpha113'] = stock.alpha_113()
#     df['alpha114'] = stock.alpha_114()
#     df['alpha116'] = stock.alpha_116()
#     df['alpha117'] = stock.alpha_117()
#     df['alpha118'] = stock.alpha_118()
#     df['alpha120'] = stock.alpha_120()
#     df['alpha122'] = stock.alpha_122()
#     df['alpha123'] = stock.alpha_123()
#     df['alpha124'] = stock.alpha_124()
#     df['alpha125'] = stock.alpha_125()
#     df['alpha126'] = stock.alpha_126()
#     df['alpha129'] = stock.alpha_129()
#     df['alpha130'] = stock.alpha_130()
#     df['alpha132'] = stock.alpha_132()
#     df['alpha134'] = stock.alpha_134()
#     df['alpha135'] = stock.alpha_135()
#     df['alpha136'] = stock.alpha_136()
#     df['alpha139'] = stock.alpha_139()
#     df['alpha141'] = stock.alpha_141()
#     df['alpha142'] = stock.alpha_142()
#     df['alpha144'] = stock.alpha_144()
#     df['alpha145'] = stock.alpha_145()
#     df['alpha148'] = stock.alpha_148()
#     df['alpha150'] = stock.alpha_150()
#     df['alpha152'] = stock.alpha_152()
#     df['alpha153'] = stock.alpha_153()
#     df['alpha154'] = stock.alpha_154()
#     df['alpha155'] = stock.alpha_155()
#     df['alpha158'] = stock.alpha_158()
#     df['alpha159'] = stock.alpha_159()
#     df['alpha160'] = stock.alpha_160()
#     df['alpha161'] = stock.alpha_161()
#     df['alpha162'] = stock.alpha_162()
#     df['alpha163'] = stock.alpha_163()
#     df['alpha164'] = stock.alpha_164()
#     df['alpha167'] = stock.alpha_167()
#     df['alpha168'] = stock.alpha_168()
#     df['alpha169'] = stock.alpha_169()
#     df['alpha170'] = stock.alpha_170()
#     df['alpha171'] = stock.alpha_171()
#     df['alpha172'] = stock.alpha_172()
#     df['alpha173 '] = stock.alpha_173()
#     df['alpha174'] = stock.alpha_174()
#     df['alpha176'] = stock.alpha_176()
#     df['alpha178'] = stock.alpha_178()
#     df['alpha179 '] = stock.alpha_179()
#     df['alpha180'] = stock.alpha_180()
#     df['alpha184'] = stock.alpha_184()
#     df['alpha185'] = stock.alpha_185()
#     df['alpha186'] = stock.alpha_186()
#     df['alpha187'] = stock.alpha_187()
#     df['alpha188'] = stock.alpha_188()
#     df['alpha189'] = stock.alpha_189()
#     df['alpha191'] = stock.alpha_191()
# '''
#     g = Generator(51)
#     a1 = g.alpha_001()
#     a2 = g.alpha_002()
#     a3 = g.alpha_003()
#     a4 = g.alpha_004()
#     a5 = g.alpha_005()
#     a6 = g.alpha_006()
#     a7 = g.alpha_007()
#     a8 = g.alpha_008()
#     a9 = g.alpha_009()
#     a10 = g.alpha_010()
#     a11 = g.alpha_011()
#     a12 = g.alpha_012()
#     a13 = g.alpha_013()
#     a14 = g.alpha_014()
#     a15 = g.alpha_015()
#     a16 = g.alpha_016()
#     a17 = g.alpha_017()
#     a18 = g.alpha_018()
#     a19 = g.alpha_019()
#     a20 = g.alpha_020()
#     a21 = g.alpha_021()
#     a22 = g.alpha_022()
#     a23 = g.alpha_023()
#     a24 = g.alpha_024()
#     a25 = g.alpha_025()
#     a26 = g.alpha_026()
#     a27 = g.alpha_027()
#     a28 = g.alpha_028()
#     a29 = g.alpha_029()
#     a30 = g.alpha_030()
#     a31 = g.alpha_031()
#     a32 = g.alpha_032()
#     a33 = g.alpha_033()
#     a34 = g.alpha_034()
#     a35 = g.alpha_035()
#     a36 = g.alpha_036()
#     a37 = g.alpha_037()
#     a38 = g.alpha_038()
#     a39 = g.alpha_039()
#     a40 = g.alpha_040()
#     a41 = g.alpha_041()
#     a42 = g.alpha_042()
#     a43 = g.alpha_043()
#     a44 = g.alpha_044()
#     a45 = g.alpha_045()
#     a46 = g.alpha_046()
#     a47 = g.alpha_047()
#     a48 = g.alpha_048()
#     a49 = g.alpha_049()
#     a52 = g.alpha_052()
#     a53 = g.alpha_053()
#     a54 = g.alpha_054()
#     a56 = g.alpha_056()
#     a57 = g.alpha_057()
#     a58 = g.alpha_058()
#     a59 = g.alpha_059()
#     a60 = g.alpha_060()
#     a61 = g.alpha_061()
#     a62 = g.alpha_062()
#     a63 = g.alpha_063()
#     a64 = g.alpha_064()
#     a65 = g.alpha_065()
#     a66 = g.alpha_066()
#     a67 = g.alpha_067()
#     a68 = g.alpha_068()
#     a70 = g.alpha_070()
#     a71 = g.alpha_071()
#     a72 = g.alpha_072()
#     a74 = g.alpha_074()
#     a76 = g.alpha_076()
#     a77 = g.alpha_077()
#     a78 = g.alpha_078()
#     a79 = g.alpha_079()
#     a80 = g.alpha_080()
#     a81 = g.alpha_081()
#     a82 = g.alpha_082()
#     a83 = g.alpha_083()
#     a84 = g.alpha_084()
#     a85 = g.alpha_085()
#     a86 = g.alpha_086()
#     a87 = g.alpha_087()
#     a89 = g.alpha_089()
#     a90 = g.alpha_090()
#     a91 = g.alpha_091()
#     a93 = g.alpha_093()
#     a94 = g.alpha_094()
#     a95 = g.alpha_095()
#     a96 = g.alpha_096()
#     a97 = g.alpha_097()
#     a98 = g.alpha_098()
#     a99 = g.alpha_099()
#     a100 = g.alpha_100()
#     a101 = g.alpha_101()
#     a102 = g.alpha_102()
#     a104 = g.alpha_104()
#     a105 = g.alpha_105()
#     a106 = g.alpha_106()
#     a107 = g.alpha_107()
#     a108 = g.alpha_108()
#     a109 = g.alpha_109()
#     a110 = g.alpha_110()
#     a111 = g.alpha_111()
#     a112 = g.alpha_112()
#     a113 = g.alpha_113()
#     a114 = g.alpha_114()
#     a116 = g.alpha_116()
#     a117 = g.alpha_117()
#     a118 = g.alpha_118()
#     a120 = g.alpha_120()
#     a122 = g.alpha_122()
#     a123 = g.alpha_123()
#     a124 = g.alpha_124()
#     a125 = g.alpha_125()
#     a126 = g.alpha_126()
#     a129 = g.alpha_129()
#     a130 = g.alpha_130()
#     a132 = g.alpha_132()
#     a134 = g.alpha_134()
#     a135 = g.alpha_135()
#     a136 = g.alpha_136()
#     a139 = g.alpha_139()
#     a141 = g.alpha_141()
#     a142 = g.alpha_142()
#     a144 = g.alpha_144()
#     a145 = g.alpha_145()
#     a148 = g.alpha_148()
#     a150 = g.alpha_150()
#     a152 = g.alpha_152()
#     a153 = g.alpha_153()
#     a154 = g.alpha_154()
#     a155 = g.alpha_155()
#     a158 = g.alpha_158()
#     a159 = g.alpha_159()
#     a160 = g.alpha_160()
#     a161 = g.alpha_161()
#     a162 = g.alpha_162()
#     a163 = g.alpha_163()
#     a164 = g.alpha_164()
#     a167 = g.alpha_167()
#     a168 = g.alpha_168()
#     a169 = g.alpha_169()
#     a170 = g.alpha_170()
#     a171 = g.alpha_171()
#     a172 = g.alpha_172()
#     a173 = g.alpha_173()
#     a174 = g.alpha_174()
#     a176 = g.alpha_176()
#     a178 = g.alpha_178()
#     a179 = g.alpha_179()
#     a180 = g.alpha_180()
#     a184 = g.alpha_184()
#     a185 = g.alpha_185()
#     a186 = g.alpha_186()
#     a187 = g.alpha_187()
#     a188 = g.alpha_188()
#     a189 = g.alpha_189()
#     a191 = g.alpha_191()
# # d = pd.DataFrame(list(zip(a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14, a15)))
# # d.columns = ['alpha1', 'alpha2', 'alpha3', 'alpha4', 'alpha5', 'alpha6', 'alpha7', 'alpha8', 'alpha9', 'alpha10',
# #              'alpha11', 'alpha12', 'alpha13', 'alpha14', 'alpha15']
# # # d.reset_index(g.open_price.columns.values)
# # # d.set_index(g.open_price.columns.values)
# # d.index = g.open_price.columns.values
# # print(d)
# # a1.set_index('alpha')
#     b = pd.concat(
#         [a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14, a15, a16, a17, a18, a19, a20, a21, a22, a23, a24,
#          a25, a26, a28, a29, a31, a32, a33, a34, a35, a36, a37, a38, a39, a40, a41, a42, a43, a44, a45, a46,
#          a47, a48, a49, a52, a53, a54, a56, a57, a58, a59, a60, a61, a62, a63, a64, a65, a66, a67, a68, a70, a71, a72,
#          a74, a76, a77, a78, a79, a80, a81, a82, a83, a84, a85, a86, a87, a89, a90, a91, a93, a94,
#          a95, a96, a97, a98, a99, a100, a101, a102, a104, a105, a106, a107, a108, a109, a110, a111, a112,
#          a113, a114, a116, a117, a118, a120, a122, a123, a124, a125, a126, a129, a130,
#          a134, a135, a136, a139, a141, a142, a144, a145, a148, a150, a152, a153,
#          a154, a155, a158, a159, a160, a161, a162, a163, a164, a167, a168, a169, a170, a171,
#          a172, a173, a174, a176, a178, a179, a180, a184, a185, a186, a187, a188, a189, a191],
#         axis=1)
#
#     b.index = ['alpha1', 'alpha2', 'alpha3', 'alpha4', 'alpha5', 'alpha6', 'alpha7', 'alpha8', 'alpha9', 'alpha10',
#                'alpha11', 'alpha12', 'alpha13', 'alpha14', 'alpha15', 'alpha16', 'alpha17', 'alpha18', 'alpha19',
#                'alpha20', 'alpha21', 'alpha22', 'alpha23', 'alpha24', 'alpha25', 'alpha26',
#                'alpha28', 'alpha29', 'alpha31', 'alpha32', 'alpha33', 'alpha34', 'alpha35',
#                'alpha36', 'alpha37', 'alpha38', 'alpha39', 'alpha40', 'alpha41', 'alpha42', 'alpha43', 'alpha44',
#                'alpha45', 'alpha46', 'alpha47', 'alpha48', 'alpha49', 'alpha52', 'alpha53', 'alpha54', 'alpha56',
#                'alpha57', 'alpha58', 'alpha59', 'alpha60', 'alpha61', 'alpha62', 'alpha63', 'alpha64', 'alpha65',
#                'alpha66', 'alpha67', 'alpha68', 'alpha70', 'alpha71', 'alpha72', 'alpha74',
#                'alpha76', 'alpha77', 'alpha78', 'alpha79', 'alpha80', 'alpha81', 'alpha82', 'alpha83', 'alpha84',
#                'alpha85', 'alpha86', 'alpha87', 'alpha89', 'alpha90', 'alpha91', 'alpha93', 'alpha94', 'alpha95',
#                'alpha96', 'alpha97', 'alpha98', 'alpha99', 'alpha100', 'alpha101', 'alpha102',
#                'alpha104', 'alpha105', 'alpha106', 'alpha107', 'alpha108', 'alpha109', 'alpha110', 'alpha111',
#                'alpha112', 'alpha113', 'alpha114', 'alpha116', 'alpha117', 'alpha118', 'alpha120', 'alpha122',
#                'alpha123', 'alpha124', 'alpha125', 'alpha126', 'alpha129', 'alpha130', 'alpha134',
#                'alpha135', 'alpha136', 'alpha139', 'alpha141', 'alpha142', 'alpha144', 'alpha145', 'alpha148',
#                'alpha150', 'alpha152', 'alpha153', 'alpha154', 'alpha155', 'alpha158', 'alpha159', 'alpha160',
#                'alpha161', 'alpha162', 'alpha163', 'alpha164', 'alpha167', 'alpha168', 'alpha169', 'alpha170',
#                'alpha171', 'alpha172', 'alpha173', 'alpha174', 'alpha176', 'alpha178', 'alpha179', 'alpha180',
#                'alpha184', 'alpha185', 'alpha186', 'alpha187', 'alpha188', 'alpha189', 'alpha191']
#
