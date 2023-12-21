#!/usr/bin/env julia
######################################################
#自定义函数
######################################################



"相当于python的list,把一个字串分成单个的数组"
list(x)=[i for i in x];
  

"移动平均 ma(x,n)"
function ma(x::Array{Float64,1},n::Int)#在cpu上计算一般移动平均
               w=ones(Float64,n)
              w /=sum(w)
              a=conv(x,w)
              a[1:n].=a[n]
              return a[1:end-n+1]
              end

"a金叉于b"
function xcrros(a::Array{Float64,1},b::Array{Float64,1})
           c1=a-b
           c2=Array{Bool}(length(c1))
           for i in 1:length(c1)-1
              if c1[i]<0 && c1[i+1]>0
              c2[i+1]=true
              else
              c2[i+1]=false
              end
              end
       return c2
       end

"a交叉于b"
function crros(a::Array{Float64,1},b::Array{Float64,1})#aXb bXa                               
                d=a.<b;
                d1=unshift!(d,pop!(d)); #第二天
                ix=(d1$d);
       return ix
       end

"数字转成日期"
function num2date(num)
           return Date(Dates.rata2datetime(num))
       end

"日期转成数字"
function date2num(date)
           return Float64(Dates.datetime2rata(DateTime(date)))
       end
	
"csv获取成交后8天数组"
function getfxb(csv::AbstractString)
       df=read(tdxdata[csv])
            o,h,l,c=df[:,2],df[:,3],df[:,4],df[:,5]
            m5=ma((c*2+l+h)/4,5)
            x=xcrros(c,m5)
            idx=x2idx(x)
            li=length(idx)
            return [c[idx] c[idx+1] c[idx+2] c[idx+3] c[idx+4] c[idx+5] c[idx+6] c[idx+7] c[idx+8]]
       end

"csv 获取tohlcvm,时间，开盘，最高，最低，收盘，成交量，成交额。"
function ohlc(id::AbstractString)
 	df=jldopen("DB.jld2","r") do file
       		read(file,id)
       end
               o,h,c,l,v,p=df[:,2],df[:,3],df[:,4],df[:,5],df[:,8],df[:,9]
               return o,h,l,c,v,p;
       end

"百分化，向量除以平均值"
function percent(a)
       return a/mean(a)
       end

"idx 整理，交易期内不重复"
function idxuu(idx,day)
       id=Array{Int32,1}();
       i1=0;
                     for i in idx
                     if i-i1>day 
                     append!(id,[i]);i1=i
                     end
       end
       return id
       end


function format_x(tick_val,tick_pos)
                  if Int(round(tick_val)) in x
                        return xlabels[Int(round(tick_val))]
                  else
                        return ""
                  end
                  
                  end
function format_y(tick_val,tick_pos)
                  if Int(round(tick_val)) in x
                        return ylabels[Int(round(tick_val))]
                  else
                        return ""
                  end
                  
                  end

