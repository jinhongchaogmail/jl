


"设定周期内的总收益率，最大回撤，胜率,参数内置海选 ds平仓周期 us背离持续 ns 均线"

####在均线日连续上穿均线后。会如何？

function maintest(id="600000.SH",ns=[2:1:20;],us=[2:2:10;],ds=[10:1:90;])
         
         o,h,l,c,v,p=ohlc(id);


         log=zero(rand(10))

             for n in ns
                  m5=ma((c*2+l+h)/4,n);   #     N日均线  
                  for u in us
                      gongshi= GS_m5_c_bl(m5,c,u)                     
                      ix=try gongshi; catch; [] end  #利用已有选股公式提取出所有条件符合的位置 （bool量）
                      idx=findall(ix)                  #取得所有真值所在的时间点 （时间点）                                                     
                      for d in ds                          
                           if length(idx)!=0           #如果idx数量为零则跳过
                                endidx=idx[end]
                                idx= idx[pushfirst!(diff(idx),d+1).>d]     #diff(idx) idx点的间隔会少1。pushfirst,d+1放前保持第一个并剔除间隔小的。
                                if length(idx)!=0                # 如果idx数量为零则跳过
                                     while  idx[end]+d>=length(c)    #如果最后一个交易点+周期超过了总天数，就把最后一个交易点循环剔除
                                           idx=idx[1:end-1]
                                           if length(idx)==0          #如果处理后没有交易点，直接退出此循环
                                                 break                                        
                                            end                                        
                                      end
                                 
                                     if length(idx)>0
                                js=((c[idx.+d].-c[idx])./c[idx]).+1;      #收益率:周期到了价格减去交易点价格/交易点价格+1
                                hc=minimum(js)-1;                           #最大回撤
                                cishu=length(idx)
                                tianshu=length(o)
                                append!(log, [cishu, round(length(js[js.>1])./length(js),digits=4), round(prod(js),digits=4), round(hc,digits=4), d, u, n ,tianshu, round(cishu/tianshu,digits=4),endidx ])
                                     end 
                           
                           
                               
                                end
                           end             
                      end                          
                 end
            end

            x=10
            log=reshape(log,x,div(length(log),x))

            ah=log[:,log[2,:].*log[3,:].==maximum(log[2,:].*log[3,:])]#这个数字是指按胜率2或是按总收益3来取所有测试中最大的一个。
            #这里　　　　　需要再好好想想．选出最有价值的．
           #ah=log[:,log[3,:].==maximum(log[3,:])]#这个数字是指按总收益3来取所有测试中最大的一个。
            return ah[:,1]                        #考虑一下用胜率与总收益的积来做为参考。
          
end



allstocks=JLD2.load("DB.jld2","ts_codes")#
function Backtesting(dates=["20220109",""],stocks=allstocks)
		#现在认为起止与票都是本地数据库里有的，且时间准确
		minchen=stocks[:,2]
		index=stocks[:,1]
		array=pmap(maintest,index)　#数据处理
	         
	         
		 array=hcat(array...)'　# 综合转向
                 array= hcat(minchen,array)	#并入名称
 		 #array=sortslices(array, dims=1, lt=(x,y)->isless(x[4],y[4]))   #   按需排序          

                 df=DataFrame(array,:auto)
                 sort胜率=sort(df,[:x4,:x3,:x10],rev=true)	#   按需排序  
		out=rename!(sort胜率,:x1=>:股票名称,　:x2=>:次数,　:x3=>:胜率,　:x4=>:总收益,　:x5=>:最大回撤,　:x6=>:平仓周期,　:x7=>:背驰,　:x8=>:均线,  :x9=>:总天数,  :x10=>:命中率,  :x11=>:最后)
		filepath=string(pwd(),"/backtest.xlsx")
		XLSX.writetable(filepath, collect(DataFrames.eachcol(out)), DataFrames.names(out))
                 return out
                #return names!(sort胜率,[:名称,:交易次数,:胜率,:总收益,:最大回撤,:周期,:背驰,:均线]) #加上列名输出
end 
