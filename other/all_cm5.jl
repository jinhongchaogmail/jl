#!/usr/bin/env julia
######################################################
#  功能：把每一天出现 cm5买点的所有股票进行统计，
#       并画出来与大盘进行比较。
######################################################

@everywhere function onecm5(csv="SH600005.csv",n=10,u=3,num=1500)

        df=try read(tdxdata[csv])[end-num:end,:];
          catch; [] end
        if length(df)>num
 
	        t,o,h,l,c,v,m=df[:,1],df[:,3],df[:,4],df[:,5],df[:,6],df[:,7],df[:,8];
	        m5=ma((c*2+l+h)/4,n);

	       ix=try GS_m5_c_bl(m5,c,u);#选股公式
	                                   catch; [] end
	       return ix
       else
             return falses(num+1)

       end
end




@everywhere dname="1ddata"

@everywhere fid=h5open("data.h5","r")

@everywhere tdxdata=fid[dname]

@everywhere files=names(tdxdata)


num=1500
an=pmap(onecm5,Progress(length(files)),files)
an=hcat(an...)'
allidx=sum(an,1)'

dname="index"
fid=h5open("data.h5","r")
index=fid[dname]
files1=names(index)
c=read(index[files1[1]])[end-num:end,6]
t=read(index[files1[1]])[end-num:end,1]


fig=plt.figure(facecolor="white")
plt.title(files1[1])
plot(Dates.rata2datetime(t),allidx,"ro-",label="all c-m5")
plt.legend(loc="upper left")
plt.grid(true)

plt.twinx()
t1=read(tdxdata[files[1]])[end-num:end,1]
plot(Dates.rata2datetime(t1),c,"bo-",label="index")
plt.legend(loc="upper right")
plt.grid(true)



plt.show()



























