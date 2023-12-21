#!/usr/bin/env julia
#因为有每小时4000次请求的限制。还是改成单线程算了。

  using Pandas
  using JLD2
  using PyCall
  using ProgressMeter

  ts=pyimport("xcsc_tushare")
  ts.set_token("793aaae8a99da22beccae7bcb56c47fd0f90b84053a211e8462335e5")
  pro = ts.pro_api(env="prd",server="http://116.128.206.39:7172")
  hist_fields="trade_date,open,high,low,close,change,pct_chg,volume,amount"


  function get_hist(ts_code)	
	df = pro.daily(ts_code="$ts_code",start_date="20220101",end_date="",fields=hist_fields)
	a=Array(Pandas.DataFrame(df))
	if size(a)[1]>21
		return  ts_code=>a
	end 
       		
end
       

temp0=Pandas.DataFrame(pro.stock_basic(market="CS"))
#删除退市的 delist_date字段非空
a=temp0.delist_date
a1=[ isnothing(a[i]) for i in keys(Array(a))]
temp0=temp0[a1]
#选择主板
temp1=temp0[(temp0.list_board_name.=="主板")]
temp2=values(loc(temp1)[:, [:ts_code,:name]])
ts_codes=temp2#[1:500,:]   
	   





using PyCall

f = jldopen("DB.jld2", "w") 
@showprogress for x in ts_codes[:,1]
    i = nothing
    while isnothing(i)
        try
            i = get_hist(x)
            if isnothing(i)
                # get_hist(x)返回nothing，跳出while循环
                break
            end
        catch e
            if isa(e, PyCall.PyError) && occursin("requests.exceptions.ConnectionError", string(e.val))
                # 网络断开，立即重试
                continue
            elseif isa(e, PyCall.PyError) && occursin("Exception('抱歉，您每小时最多访问该接口4000次')", string(e.val))
                # 达到请求限制，等待到下一个小时
                println(" 达到请求限制，等待到下一个小时")
                now = time()
                next_hour = now + (60 * 60 - now % (60 * 60))
                sleep(next_hour - now)
            end
        end
    end

    if !isnothing(i)
        println(i.first)
        f[i.first] = i.second
    end
end
keys(f)
close(f)


dt_codes=Dict([ts_codes[i,1]=>ts_codes[i,2]  for i in 1:size(ts_codes)[1]])#转换成字典 
loc_ts_code=[i for i in keys(JLD2.load("DB.jld2"))]#所有本地数据的ts_code


ts_codes= [[i for i in loc_ts_code] [ dt_codes[i] for i in loc_ts_code]]#本地ts_codes和name字典，保存 

jldopen("DB.jld2","r+") do file
write(file,"ts_codes",ts_codes)

  end     
       







