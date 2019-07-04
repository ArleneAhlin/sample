#!/usr/bin/env python
# coding: utf-8

# In[55]:


# -*- coding: utf-8 -*-
"""
Created on Sat Jun 22 17:51:34 2019

@author: youji
"""

# 一般的なモジュール
import pandas as pd
import numpy  as np
import datetime
import seaborn
import matplotlib.pyplot as plt
from scipy.stats import linregress

# 投資分析にのみ必要なモジュール
import mpl_finance
from   oandapyV20 import API
import oandapyV20.endpoints.instruments as oandapy


import pickle

# For suppressing warning
# https://github.com/numpy/numpy/issues/11448
np.seterr(invalid='ignore')

class FXBase():
    candles = None
    def __init__(self):
        cols = ['col1', 'col2']
        FXBase.candles = pd.DataFrame(index=[], columns=cols)
        
class RequestRate(FXBase):
    def __init__(self, days=10, read_file=False):
        # request用パラメータ設定
        num_candles  = int(days * 24 * 12) # 24h * 60min / 5分刻み
        minutes      = num_candles * 720 # 60 = 12 * 5 分
        now          = datetime.datetime.now() - datetime.timedelta(hours=9) # 標準時に合わせる
        start_time   = now - datetime.timedelta(minutes=minutes)
        start_time   = start_time.strftime("%Y-%m-%dT%H:%M:00.000000Z")
        params       = {
            "alignmentTimezone": "Japan",
            "from":  start_time,
            "count": 5000,
            "granularity": "H1" # per 1h
        }
        access_token = "4122baed289c346a74b193c9bee3937a-b0772f2526c6f7c81eae4e3fc708e66a"
        api          = API(access_token = access_token, environment="practice")
        request      = oandapy.InstrumentsCandles(
            instrument = "USD_JPY",
            params     = params
        )

        if read_file == False:
            # request処理
            api.request(request)
    
            # request結果データの整形 / クラス外から呼出し可能にする
            candle         = pd.DataFrame.from_dict([ row['mid'] for row in request.response['candles'] ])
            
            
            # astype による cast を複数列へ https://qiita.com/driller/items/af1369a5c0fc2ec61af3
            candle         = candle.astype({'c': 'float64', 'l': 'float64', 'h': 'float64', 'o': 'float64'})
            candle.columns = ['close', 'high', 'low', 'open']
            candle['time'] = [ row['time'] for row in request.response['candles'] ]
            # 冗長な日時データを短縮整形 https://note.nkmk.me/python-pandas-datetime-timestamp/
            candle['time'] = pd.to_datetime(candle['time']).astype(str)
            FXBase.candles = candle
            
            # 読んだファイルを保存しておく
            f = open('FXBase.candles.binaryfile','wb')
            pickle.dump(FXBase.candles,f)
            f.close
        
        else:
            f = open('FXBase.candles.binaryfile','rb')
            FXBase.candles = pickle.load(f)
        

        # 表示可能な最大行数を設定
        pd.set_option("display.max_rows", num_candles)

## チャートを単回帰分析し、得られる単回帰直線よりも上（下）の値だけで再度単回帰分析...
## これを繰り返し、高値（安値）を2～3点に絞り込む

# 高値の始点/支点を取得
def get_highpoint(start, end):
    chart = FXBase.candles[start:end+1]
    while len(chart)>3:
        regression = linregress(
            x = chart['time_id'],
            y = chart['high'],
        )
        chart = chart.loc[chart['high'] > regression[0] * chart['time_id'] + regression[1]]
    return chart

# 安値の始点/支点を取得
def get_lowpoint(start, end):
    chart = FXBase.candles[start:end+1]
    while len(chart)>3:
        regression = linregress(
            x = chart['time_id'],
            y = chart['low'],
        )
        chart = chart.loc[chart['low'] < regression[0] * chart['time_id'] + regression[1]]
    return chart

def g_trendlines(span=20, min_interval=3):
    trendlines = []
    reglession_list = []
    

    # 高値の下降トレンドラインを生成
    for i in FXBase.candles.index[::int(span/2)]:
        highpoint = get_highpoint(i, i + span)
        # ポイントが2箇所未満だとエラーになるので回避する
        if len(highpoint) < 2:
            continue
        # 始点と支点が近過ぎたらトレンドラインとして引かない
        if abs(highpoint.index[0] - highpoint.index[1]) < min_interval:
            continue
        regression = linregress(
            x = highpoint['time_id'],
            y = highpoint['high'],
        )
        print(regression[0] < 0.0, 'reg_high: ', regression[0], ', ', regression[1], )

        # 下降してるときだけ
        if regression[0] < 0.0:
            trendlines.append(regression[0] * FXBase.candles['time_id'][i:i+span*2] + regression[1])
            reglession_list.append(regression)

    # 安値の上昇トレンドラインを生成
    for i in FXBase.candles.index[::int(span/2)]:
        
        lowpoint   = get_lowpoint(i, i + span)
        # ポイントが2箇所未満だとエラーになるので回避する
        if len(lowpoint) < 2:
            continue
        # 始点と支点が近過ぎたらトレンドラインとして引かない
        if abs(lowpoint.index[0] - lowpoint.index[1]) < min_interval:
            continue
        regression = linregress(
            x = lowpoint['time_id'],
            y = lowpoint['low'],
        )
        print(regression[0] > 0.0, 'reg_low: ', regression[0], ', ', regression[1], )

        # 上昇してるときだけ
        if regression[0] > 0.0:
            trendlines.append(regression[0] * FXBase.candles['time_id'][i:i+span*2] + regression[1])
            reglession_list.append(regression)
    
    
    feature_value_list = []
    if 1:
        Ntl = len(trendlines)
        for i in range(Ntl):
            for j in range(i+1, Ntl):
                
                line1 = trendlines[i]
                line2 = trendlines[j]
                # トレンドラインの交差範囲を求める -----------------------------------------
                # ライン1,2の交差範囲の下限x
                #x_min = line1[line1 == max(line1)].index.tolist()[0]
                lmin = max(line1[line1 == max(line1)].index.tolist()[0], 
                           line2[line2 == min(line2)].index.tolist()[0])
                
                lmax = min(line1[line1 == min(line1)].index.tolist()[0], 
                           line2[line2 == max(line2)].index.tolist()[0])
                
                cross_range = lmax - lmin
                cross_range_idx_list = list(range(lmin, lmax+1))

                all_range_min = min(line1[line1 == max(line1)].index.tolist()[0], 
                           line2[line2 == min(line2)].index.tolist()[0])
                
                all_range_max = max(line1[line1 == min(line1)].index.tolist()[0], 
                           line2[line2 == max(line2)].index.tolist()[0])
                
                all_range = all_range_max - all_range_min
                cross_rate = cross_range / all_range
                
                if cross_rate > 0.5:
                    # 交差範囲の後半に交点はあるか？
                    #　ライン1,2の差分絶対値リスト
                    
                    line12diff_abs = [abs(line1[j] - line2[j]) for j in cross_range_idx_list]
                    
                    #　ライン1,2の差分リスト
                    #line12diff = [(line1[j] - line2[j]) for j in cross_range_idx_list]
                    
                    # ライン交差期間、終点のx座標
                    max_index = cross_range_idx_list[len(cross_range_idx_list)-1]
                    # ライン交差期間、始点のx座標
                    min_index = cross_range_idx_list[0]
                    # ライン1,2の交点の座標
                    min_index_abs = cross_range_idx_list[line12diff_abs.index(min(line12diff_abs))]
                    
                    # 交点が存在する
                    if min_index < min_index_abs < max_index:
                        # 交点は三角持ち合い期間の、後半である
                        if   (lmin + lmax)/2 < min_index_abs:
                            # 交点のY座標
                            intersection_y = line1[min_index_abs]
                            # 交点の価格
                            intersection_value = FXBase.candles['high'][min_index_abs]
                            # 交点後の価格（最大か最小）　　予測したい価格
                            if intersection_y <= intersection_value:
                                after_intersection_value = max(FXBase.candles['high'][min_index_abs:min_index_abs + 50])
                                break_type = 0
                            else:
                                break_type = 1
                                after_intersection_value = min(FXBase.candles['high'][min_index_abs:min_index_abs + 50])
                                
                            # 三角持ち合いの期間
                            triangle_interval = min_index_abs - min_index
                            
                            feature_value_list_temp = [i,j,
                                                       break_type,
                                                       reglession_list[i][0], 
                                                       reglession_list[i][1], 
                                                       reglession_list[j][0], 
                                                       reglession_list[j][1], 
                                                       min_index, 
                                                       max_index, 
                                                       min_index_abs, 
                                                       triangle_interval, 
                                                       intersection_y, 
                                                       intersection_value, 
                                                       after_intersection_value 
                                                       ]
                            
                            feature_value_list.append(feature_value_list_temp)
                            
                
        feature_value_list_df = pd.DataFrame(feature_value_list)
        feature_value_list_df.columns = ['i', 'j', 
                                         'brak_type',
                                         'a1', 'b1', 'a2', 'b2',  
                                         'cross_range_start_x', 
                                         'cross_range_end_x', 
                                         'intersection_point_x',
                                         'triangle_interval_x',
                                         'intersection_y',
                                         'intersection_value',
                                         'after_intersection_value']
    else:
        feature_value_list_df = pd.DataFrame(feature_value_list)
    
    return trendlines, feature_value_list_df


if __name__ == '__main__':
    requester = RequestRate(days=2, read_file=False)
    #requester = RequestRate(days=2, read_file=True)
    FXBase.candles['time_id']= FXBase.candles.index + 1

    figure, (axis1, axis2) = plt.subplots(2, 1, figsize=(20,10), dpi=200, gridspec_kw = {'height_ratios':[3, 1]})
    
    
    
    # ローソク足
    mpl_finance.candlestick2_ohlc(
        axis1,
        opens  = FXBase.candles.open.values,
        highs  = FXBase.candles.high.values,
        lows   = FXBase.candles.low.values,
        closes = FXBase.candles.close.values,
        width=0.6, colorup='#77d879', colordown='#db3f3f'
    )
    
    # トレンドラインを引く
    tline_list, feature_value_list_df, = g_trendlines()
    
    feature_value_list_df.to_csv('feature_value_list_df.csv', index = False)
    
    for i, line in enumerate(tline_list):
        #if i==1 or i == 12:
        axis1.plot(line, label=i)
        
    # X軸の見た目を整える
    xticks_number  = 96 # 12本刻みに目盛りを書く
    xticks_index   = range(0, len(FXBase.candles), xticks_number)
    xticks_display = [FXBase.candles.time.values[i][11:16] for i in xticks_index] # 時間を切り出すため、先頭12文字目から取る
    
    # axis1を装飾 ( plt.sca(axis): set current axis )
    plt.sca(axis1)
    plt.xticks( xticks_index, xticks_display )
    plt.legend()
    plt.show()
    


# In[6]:


# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.externals import joblib
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn import linear_model
import numpy as np
from pandas import Series,DataFrame

# CSVファイルの読み込み
data = pd.read_csv("/Users/tomodaaya/Downloads/feature_value_list_df.csv", sep=',')

data['b'] = data['b1']-data['b2']
data['a'] = (data['a1']+data['a2'])/2


# 回帰モデルの呼び出し
clf = linear_model.LinearRegression()


# 説明変数
#幅、傾き、ライン交差期間、交点のX座標、交点のY座標
X = data.loc[:, ['b','a','triangle_interval_x','intersection_point_x','intersection_y',]].values

# 目的変数
#ブレイク後の値
Y = data['after_intersection_value'].values

# 予測モデルを作成（重回帰）
clf.fit(X, Y)

# 回帰係数と切片の抽出
a = clf.coef_
b = clf.intercept_  

# 回帰係数
print("回帰係数:", a) # 回帰係数: [ 0.70068905 -0.64667957]
print("切片:", b) # 切片: 12.184694815481187
print("決定係数:", clf.score(X, Y)) # 決定係数: 0.6624246214760455

# 学習結果を出力
joblib.dump(clf, 'multiple.learn') 

Y_predict = clf.predict(X)


plt.scatter(Y, Y_predict)
# 回帰直線

plt.title("Y Y_predict title")
plt.xlabel("Y")
plt.ylabel("Y_predict")
plt.grid()
plt.savefig('figure.png')
plt.show()

