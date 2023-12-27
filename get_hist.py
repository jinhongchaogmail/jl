#get日线数据到DB.h5

import pandas as pd
import xcsc_tushare as ts
import h5py
import time
import requests

ts.set_token('793aaae8a99da22beccae7bcb56c47fd0f90b84053a211e8462335e5')
pro = ts.pro_api(env='prd',server='http://116.128.206.39:7172')

hist_fields="trade_date,open,high,low,close,change,pct_chg,volume,amount"

def get_hist(ts_code):
    df = pro.daily(ts_code=ts_code, start_date="20220101", end_date="", fields=hist_fields)
    df = df.iloc[::-1]  # 翻转DataFrame
    if len(df) > 21:
        return ts_code, df

temp0 = pro.stock_basic(market="CS")
temp0 = temp0[temp0["delist_date"].isna()]
temp1 = temp0[temp0["list_board_name"] == "主板"]
ts_codes = temp1[["ts_code", "name"]]

with h5py.File("DB.h5", "w") as f:
    for x in ts_codes["ts_code"]:
        i = None
        while i is None: ##只要i是None就一直请求
            try:
                i = get_hist(x)
                if i is None:
                    print("请求失败，等待5秒")
                    break
            except requests.exceptions.ConnectionError:
                continue
            except Exception as e:
                if "抱歉,您每小时最多访问该接口4000次" in str(e):
                    print("达到请求限制，等待到下一个小时")
                    now = time.time()
                    next_hour = now + (60 * 60 - now % (60 * 60))
                    time.sleep(next_hour - now)
                  
        if i is not None:
            print(i[0])
            
            data = i[1].drop(columns=['trade_date']).to_numpy()
            f.create_dataset(i[0], data=data)
    print(list(f.keys()))