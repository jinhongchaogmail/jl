#!/usr/bin/env julia

using Statistics,DataFrames, JLD2, Plots,Flux,MLDataUtils
using Base.Threads
 """
        prep_data(arr, w_size, l=[5, 6, 9])
    
    这个函数从日线数据中提取训练和验证数据。它首先对数据进行归一化处理，
    然后将数据转换为三维数组，其中第一维是特征数，第二维是时间步长，第三维是样本数。
    
    # 参数
    - `arr`: 输入的日线数据，应为一个二维数组。
    - `w_size`: 时间窗口的大小。
    - `l`: 一个列表，指定要从 `arr` 中提取的列。
    
    # 返回
    返回一个三维数组，其中包含了处理后的数据。
 """
 function prep_data(arr,w_size,l=[5, 6, 9])
    
        #读取数组并进行逆序选取相关列切片
          arr = copy(arr[end:-1:1,l]');#(3,n)
        # 计算均值和标准差
         mu = mean(arr, dims=2)
         std_dev = std(arr, dims=2)    
        # 归一化处理
         arr = (arr .- mu) ./ std_dev
        #将数组转换成三维数组(特征数，时间步长，样本数)
         data_x = Array{Float32,3}(undef, size(arr,1),w_size,size(arr, 2)-w_size+1)
        for j in 1:size(arr, 2)-w_size+1
            data_x[:,:,j] = arr[ :,j:j+w_size-1] |> Matrix{Float32}
        end
        #0trade_date,2open,3high,4low,5close,6change,7pct_chg,8volume,9amount
        train_x=data_x[:,1:end-1,:]
        train_y=data_x[5,end,:]
        train_y=reshape(train_y,1,1,size(train_x,3))    

        return   splitobs(shuffleobs((train_x, train_y)), at = 0.8)
   
 end 

f = jldopen("DB.jld2", "r")
    ks = keys(f)
        if !isdefined(Main, :i)
            i = 5
        end
w_size = 13
for i in 1:30#eachindex(ks)[1:end-1]        
    #trade_date,open,high,low,close,change,pct_chg,volume,amount
    (train_set, val_set) = prep_data(f[ks[i]], w_size, [2, 3, 4, 5,6, 7, 8])


    #定义模型

    begin
        bsz = 1
        epochs = 100
        m = 14
        model = Chain(
            LSTM(7, m),
            Dense(m, 1),
            last)
        # 定义损失函数和优化器
        loss(x, y) = Flux.mse(model(x), y)
        opt = Descent(0.1)

        # 定义空数组存放损失函数的值
        t_l = Float32[]
        v_l = Float32[]
    end

    # 使用 DataLoader 处理数据，并开启异步加载
    train_data = Flux.Data.DataLoader((copy(train_set[1]), copy(train_set[2])), batchsize=bsz)
    val_data = Flux.Data.DataLoader((copy(val_set[1]), copy(val_set[2])), batchsize=bsz)

    params = Flux.params(model)
    @time for i in 1:epochs
        Flux.reset!(model)
        Flux.train!(loss, params, train_data, opt)
        if i % 10 == 0
            # 计算训练集的平均损失
            train_loss = mean(loss(x, y) for (x, y) in train_data)
            append!(t_l, train_loss)

            # 计算验证集的平均损失
            val_loss = mean(loss(x, y) for (x, y) in val_data)
            append!(v_l, val_loss)

            # @info "Epoch $i. 训练集 loss: $(t_l[end]). 验证集 loss: $(v_l[end])"
        end

    end

        begin # 绘制损失函数随时间的变化趋势图

            ad1 = i
            ad2 = w_size
            ad = string(ad1, "_", ad2)

            ptl = "t_l_$ad1"
            pvl = "v_l_$ad1"

            p = plot(xlabel="epochs", ylabel="Loss", title="train Loss over epochs $ad", size=(800, 600), ylims=(0.5, 1.5))
            plot!(p, t_l, labels=[ptl])
            plot!(p, v_l, labels=[pvl])
            losspng = "png/l_p W_$ad.png"

            savefig(p, losspng)


        end




    

end
close(f)
 
 
 
 
 
 
 
