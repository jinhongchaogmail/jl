
include("init.jl")
using PyCall;
ts=pyimport("tushare");
pd=pyimport("pandas")
plt=pyimport("matplotlib.pyplot");
fm=pyimport("matplotlib.font_manager");
plt3d=pyimport("mpl_toolkits.mplot3d.axes3d")


font = fm.FontProperties(fname="/usr/share/fonts/truetype/wqy/wqy-microhei.ttc", size=12);

	


	#labels = [1,2,3,4,5,6,7,8,9,10,"a","b","cc","d","e","f","g","h","*%","20"]
	data=maintest_1()
	a=[Int(i) for i in (data[5:7,:])]
	x=keys(a[1,:])[end]

	xlabels=[string(a[1,i],"-",a[2,i],"-",a[3,i])  for i in keys(a[1,:])]

        fig = plt.figure(figsize=(9,8))
	ax = fig.add_subplot(111)
	ax.xaxis.set_major_formatter(plt.FuncFormatter(format_x))
	ax.xaxis.set_major_locator(plt.MaxNLocator(integer=true))

	plt.title("公式效果分析", fontproperties=font); #标题
	plt.xlabel("x轴",fontproperties=font)
	plt.ylabel("Y轴",fontproperties=font)

	plt.plot(data[1,:],label="chishu")
	plt.plot(data[6,:],label="beichi")
	plt.plot(data[5,:],label="junxian")
	plt.twinx()
	plt.plot(data[2:4,:]',label="ab")

	plt.legend(loc="upper left",prop =font) ;#显示label的位置


	plt.gcf().autofmt_xdate() ; # 自动旋转xlabel

	
	
	plt.grid(true) ;#加横线
plt.show()


