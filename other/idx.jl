#选股公式


"背离       idx=x2idx(m5_c_bl(m5,c,m=2))"
function GS_m5_c_bl(m5,c,m=2)
        len_c=length(c)
        b=falses(len_c)
        cm5=c.-m5
       for i in (1+m):len_c
                  if (cm5[i-m]<cm5[i])&(c[i-m]>c[i])&(cm5[i]<0.2)        #如果cm5的m单位之前小于当前值，和c的m单位时间前大于当前，和cm5<0.2
                  b[i]=true
                  else
                  b[i]=false
                  end
              end
       return b
       end



"收盘价上穿均线   idx=x2idx(GS_cXma_n(c,5))"
function GS_cXma_n(c,n=5)
     return xcrros(c,ma((c*2+l+h)/4,n))
end
