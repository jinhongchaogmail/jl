#!/usr/bin/env julia

using Statistics,DataFrames, JLD2, Plots,Flux,MLDataUtils,BSON,CUDA
using Base.Threads
 """
    prep_data(feature=4,w_size=10)
    从DB.jld2上按特征提取训练数据。默认上涨4个点之前10日数据。
    # 参数   
    -`feature`: 一个整数，指定要使用的特征数。
    - `w_size`: 时间窗口的大小。  
    # 返回
    包含训练张量的x列表,和包含对比y的列表。
 """
 function find_features(arr, feature, w_size)
    indices = []
    i = 1
    while i <= length(arr[6, :])
        if arr[6, i] > feature
            push!(indices, i)
            i += w_size  # 跳过w_size个数据
        else
            i += 1
        end
    end
    return indices
end
function prep_data(feature=9.5,w_size=10)
     f = jldopen("DB.jld2", "r")
    ks = keys(f)
    X = Array{Array{Float32,2},1}()  
    Y = Array{Array{Float32,2},1}()  
    for i in 1:1000#eachindex(ks)[1:end-1]   
    arr=f[ks[i]]
    #读取数组并进行逆序
    arr = copy(arr[end:-1:1,2:9])'|> Matrix{Float32};#(8,n)
    # 第六列的值大于feature的行的索引
    indices = find_features(arr, feature, w_size)
    # 计算均值和标准差
    mu = mean(arr, dims=2)        
    std_dev = std(arr, dims=2)            
    # 归一化处理       
    arr = (arr .- mu) ./ std_dev    #(8,n)    
           
    for i in indices
        if i > w_size
            d = arr[:,i-w_size:i-1]#(8,w_size时间窗）
            x= d[:,1:end-1]#(8,w_size-1)#取前
            y = reshape(d[6,2:end],1,w_size-1)#(1,w_size)＃取后并重塑
            push!(X,x)
            push!(Y,y)
           
        end
    end
    end 
    @show length(X)
    # 初始化一个空的三维数组
    X3 = Array{Float32,3}(undef, size(X[1])..., length(X))
    for i in 1:length(X)
        X3[:,:,i] = X[i]
    end
    Y3 = Array{Float32,3}(undef, size(Y[1])..., length(Y))
    for i in 1:length(Y)
        Y3[:,:,i] = Y[i]
    end
    close(f)
    return   splitobs(shuffleobs((X3, Y3)), at = 0.8)
end 


feature=9;w_size=20
   (train_set, val_set) = prep_data(feature,w_size)


    hidden_sizes = [5, 10, 20]
    learning_rates = [0.01, 0.001, 0.0001]
    batch_sizes = [1, 10, 100]
    epoch_nums = [100, 200, 300]  
   global best_val_loss = Inf
   global best_params = nothing
for m in hidden_sizes, η in learning_rates, bsz in batch_sizes, epochs in epoch_nums

   @show m, η, bsz, epochs  

    #bsz = 1
    # 使用 DataLoader 处理数据，并开启异步加载
    train_data = Flux.Data.DataLoader((copy(train_set[1])|> gpu, copy(train_set[2])|> gpu), batchsize=bsz)|> gpu
    val_data = Flux.Data.DataLoader((copy(val_set[1])|> gpu, copy(val_set[2])|> gpu), batchsize=bsz)|> gpu
   

    begin#定义模型
    #epochs = 100

    # 如果模型文件存在，就加载模型参数
    #if isfile("model.bson")
       # BSON.@load "model.bson" model
       #else
           m = 32
        model = Chain(
            LSTM(8, m),
            LSTM(m, m),
            LSTM(m, m),
            Dense(m, 1),
            )|> gpu
   # end
        # 定义损失函数和优化器
        loss(x, y) = Flux.mse(model(x), y)|> gpu
        opt = Descent(η)|> gpu
        #opt = ADAM(0.01, (0.8, 0.999))  # 设置学习率为0.01，β为(0.8, 0.999)
        #opt = AdaGrad(η)

        # 定义空数组存放损失函数的值
        t_l = Float32[]
        v_l = Float32[]
        params = Flux.params(model)
    end
    
   @time for i in 1:epochs
        Flux.reset!(model)
        Flux.train!(loss, params, train_data, opt)
        if i % 10 == 0
            # 并行计算训练集和验证集的平均损失
            train_loss_future = Threads.@spawn mean(loss(x, y) for (x, y) in train_data)
            val_loss_future = Threads.@spawn mean(loss(x, y) for (x, y) in val_data)
            println(Threads.nthreads())  
            train_loss = fetch(train_loss_future)
            append!(t_l, train_loss)
    
            val_loss = fetch(val_loss_future)
            append!(v_l, val_loss)
    
            @info "Epoch $i. 训练集 loss: $(t_l[end]). 验证集 loss: $(v_l[end])"  
            # 如果验证集的损失函数值小于之前的最小值，就保存模型参数
                if val_loss < best_val_loss
                    global best_val_loss = val_loss
                    global best_params = (m, η, bsz, epochs)
            end
        end

    end
    println("Best parameters: ", best_params)
    # 保存模型参数
    BSON.@save("model.bson",model)


    

    begin # 绘制损失函数随时间的变化趋势图

            ad1 = bsz
            ad2 = w_size
            ad = string(ad1, "_", ad2)

            ptl = "t_l_$ad1"
            pvl = "v_l_$ad1"

            p = plot(xlabel="epochs", ylabel="Loss", title="train Loss over epochs $ad", size=(800, 600))
            plot!(p, t_l, labels=[ptl])
            plot!(p, v_l, labels=[pvl])
            losspng = "png/l_p W_$ad.png"

            savefig(p, losspng)


    end

end 

 
 
 
 
 
 
 
