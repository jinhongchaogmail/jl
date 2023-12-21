
include("init.jl")
plt=pyimport("matplotlib.pyplot");
fm=pyimport("matplotlib.font_manager");
plt3d=pyimport("mpl_toolkits.mplot3d.axes3d")
days=[10:1:90;];us=[2:2:10;];ns=[2:2:20;]









font = fm.FontProperties(fname="/usr/share/fonts/truetype/wqy/wqy-microhei.ttc", size=12);
	#labels = [1,2,3,4,5,6,7,8,9,10,"a","b","cc","d","e","f","g","h","*%","20"]
data1=maintest_1("a002356")

	

	a=[Int(i) for i in (data1[5:7,:])]
	

xlabels=[string(a[1,i],"day")  for i in keys(a[1,:])][1:81]
ylabels=[string(a[2,i],"-",a[3,i],"day")  for i in keys(a[1,:])][1:50]
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
ax.yaxis.set_major_formatter(plt.FuncFormatter(format_x))
ax.yaxis.set_major_locator(plt.MaxNLocator(integer=true))

ax.xaxis.set_major_formatter(plt.FuncFormatter(format_x))
ax.xaxis.set_major_locator(plt.MaxNLocator(integer=true))

	plt.title("公式效果分析", fontproperties=font); #标题
	plt.xlabel("X=平仓周期",fontproperties=font)
	plt.ylabel("Y=均线＋背驰",fontproperties=font)

h=50  #坐标所需
w=81
a2=Array([1:h;])
a1=Array([1:w;])
c1=fill(0,(w,h))
c2=fill(0,(h,w))
x=broadcast(+,a1,c1)
y=broadcast(+,a2,c2)'




data=data1[:,2:end]

d2=reshape(data[2,:],w,h)
d3=reshape(data[3,:],w,h)#胜率
d4=reshape(data[4,:],w,h)
d5=reshape(data[5,:],w,h)
d6=reshape(data[6,:],w,h)
d7=reshape(data[7,:],w,h)
d8=reshape(data[8,:],w,h)
d9=reshape(data[9,:],w,h)#命中率

z2=c1.+d2
z3=c1.+d3
z4=c1.+d4
z5=c1.+d5
z6=c1.+d6
z7=c1.+d7
z8=c1.+d8
z9=c1.+d9



X=x;
Y=y;
Z=z4;
#ax.plot_surface(X, Y, Z, rstride=8, cstride=8, alpha=0.7)
cset = ax.contour(X, Y, Z, zdir="z", offset=1, cmap=plt.cm.coolwarm)
cset = ax.contour(X, Y, Z, zdir="x", offset=0, cmap=plt.cm.coolwarm)
cset = ax.contour(X, Y, Z, zdir="y", offset=60, cmap=plt.cm.coolwarm)

#ax.plot_wireframe(x,y,z2, rstride=1, cstride=1,color="blue",alpha=0.3)#
#ax.plot_wireframe(x,y,z3.+50, rstride=1, cstride=1,color="red",alpha=0.3)#胜率3
ax.plot_wireframe(x,y,(z4.*20).+40, rstride=1, cstride=1,color="black",alpha=0.3)#总收益
#ax.plot_wireframe(x,y,z5, rstride=1, cstride=1,color="green",alpha=0.3)#平仓周期
#ax.plot_wireframe(x,y,z6./10, rstride=1, cstride=1,color="green",alpha=0.3)#均线周期
#ax.plot_wireframe(x,y,z7, rstride=1, cstride=1,color="yellow",alpha=0.3)#背驰时长
#ax.plot_wireframe(x,y,(z9.*10000).+20, rstride=1, cstride=1,color="orange",alpha=0.3)#命中率

#for angle in range(1, stop=360)
#           ax.view_init(30, angle)
 #          plt.draw()
   #        plt.pause(.001)
     #      end
plt.show()







