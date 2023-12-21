#!/usr/bin/env julia
######################################################
#显示一个股票的曲线
######################################################
using PyCall;
ts=pyimport("tushare");
pd=pyimport("pandas")
plt=pyimport("matplotlib.pyplot");
fm=pyimport("matplotlib.font_manager");
font = fm.FontProperties(fname="/usr/share/fonts/truetype/wqy/wqy-microhei.ttc", size=16);


"画出股票的走势图"	
function get_this(id="603239")
	df=ts.get_hist_data(id);
	x=pd.to_datetime(df.index);
	y=df.close;
	z=df.open;
	dict=ts.get_stock_basics().name.to_dict();

	fig=plt.figure(facecolor="green") ;#画布的色彩
	plt.title(dict[id], fontproperties=font); #标题
	plt.plot(x,y,label="Close");
	
	plt.gcf().autofmt_xdate() ; # 自动旋转日期标记
	plt.legend(loc="upper left") ;#显示label的位置
	plt.twinx() ;         #另一个叠加的坐标
	plt.grid(true) ;#加横线
	plt.show()
end

function plt_show(x=[],y=[],title="",label="")
		fig=plt.figure(facecolor="green") ;#画布的色彩
		plt.title(title, fontproperties=font); #标题
		plt.plot(x,y,label=label);
		plt.gcf().autofmt_xdate() ; # 自动旋转日期标记
		plt.legend(loc="upper left") ;#显示label的位置
		plt.twinx() ;         #另一个叠加的坐标
		plt.grid(true) ;#加横线
		plt.show()
end
             

while true
         print("请输入股票代码")
        id=readline()
        get_this(id)
end
