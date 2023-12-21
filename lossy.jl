#!/usr/bin/env julia
using Distributed
addprocs(9 - nprocs()) 
using DataFrames, JLD2, Plots
@everywhere using Statistics,SharedArrays

@everywhere function train_model(train_x, train_y, val_x, val_y, model, loss, opt, train_losses, val_losses)
    for i in 1:e
	    #这里修改了之后就出现了问题.请帮我查一下是什么原因
	    
	    for j in 1:size(train_x,1)

	    #Flux.reset!(model)
     		Flux.train!(loss, Flux.params(model), (train_x[j, :, :], train_y[j]) , opt)
            end
	#训练集loss差值的平均值

        train_loss = [loss(train_x[i, :, :], train_y[i]) for i in 1:size(train_x, 1)]
        train_loss = mean(train_loss)
        append!(train_losses, train_loss)

        #验证集loss差值的平均值
        val_loss = [loss(val_x[i, :, :], val_y[i]) for i in 1:size(val_x, 1)]
        val_loss = mean(val_loss)
        append!(val_losses, val_loss)

        if i % 10 == 0
            @info "Epoch $i. 训练集 loss: $(train_losses[end]). 验证集 loss: $(val_losses[end])"
        end
    end
end


 function preprocess_data(window_size)
    f = jldopen("DB.jld2", "r")
    train_x, train_y, val_x, val_y = fill([], 4)
    ks = keys(f)
    for i in 1:3#eachindex(ks)[1:end-1]
        arr = @view f[ks[3]][end:-1:1,:]
        data_x = Vector{Matrix{Float32}}(undef, size(arr, 1)-window_size+1)
        for j in 1:size(data_x, 1)
            data_x[j] = arr[j:j+window_size-1, [5, 6, 9]] |> Matrix{Float32}
        end
        #将每个窗口内的数据归一化处理
        indices = eachindex(data_x)
        mu = [mean(data_x[i], dims=1) for i in indices]
        std_dev = [std(data_x[i], dims=1) for i in indices]
        data_x = [(data_x[i] .- mu[i]) ./ std_dev[i] for i in indices]
        data_y = [data_x[i][end, 2] for i in indices]
        data_x = [data_x[i][1:end-1, :] for i in indices]
        l = size(data_x, 1)
        k = Int(ceil(size(data_x, 1) * 0.8))
        #分配训练集和验证集
        t_x = Vector{Matrix{Float32}}(undef, k)
        t_y = Vector{Float32}(undef, k)
        v_x = Vector{Matrix{Float32}}(undef, l-k)
        v_y = Vector{Float32}(undef, l-k)
        for j in 1:k
            t_x[j] = permutedims(data_x[j])
            t_y[j] = data_y[j]
        end
        for j in k+1:l
            v_x[j-k] = permutedims(data_x[j])
            v_y[j-k] = data_y[j]
        end
        #将输入数组转换成三维数组(样本数，特征数，时间步长)
        t_x = cat(t_x..., dims=3)
        v_x = cat(v_x..., dims=3)
        #当i=1时，初始化训练集和验证集
        if i == 1
            train_x = t_x
            val_x = v_x
            train_y = t_y
            val_y = v_y
        #当i>1时，将每次的训练集和验证集添加到数组中   
        else
            train_x = cat(train_x, t_x, dims=3)
            val_x = cat(val_x, v_x, dims=3)
            append!(train_y, t_y)
            append!(val_y, v_y)
        end
    end
    train_x = permutedims(train_x, (3, 1, 2))
    val_x = permutedims(val_x, (3, 1, 2))
    close(f)
    return train_x ,train_y,val_x ,val_y 
end



for window_size in 6:30
train_x0 ,train_y0,val_x0 ,val_y0=preprocess_data(window_size)


#根#据训练数据的型状定义模型


global train_x1=SharedArray(train_x0)
global train_y1=SharedArray(train_y0)
global val_x1=SharedArray(val_x0) 
global val_y1=SharedArray(val_y0)
@everywhere begin
m=3
using Flux
	train_x=fetch(@spawnat 1 train_x1)
	train_y=fetch(@spawnat 1 train_y1)
	val_x=fetch(@spawnat 1 val_x1) 
	val_y=fetch(@spawnat 1 val_y1)
	e = 100
	model = Chain(
        LSTM(size(train_x, 2), m),
        Dense(m, 1),
    )
#定义损失函数和优化器
    loss(x, y) = Flux.mse(model(x), y)
    opt = Descent(0.1)

    #定义空数组存放损失函数的值
    train_losses = Float32[]
    val_losses = Float32[]

   #定义多进程数量
    num_processes = 9

    #定义每个进程的训练集数据量
    batch_train_x_size = div(size(train_x, 1), num_processes)
    #定义每个进程的验证集数据量
    batch_val_x_size = div(size(val_x, 1), num_processes)

    #定义多进程数组
    train_losses_multi = SharedArray{Float32}(e,num_processes)
    val_losses_multi = SharedArray{Float32}(e,num_processes)
end
#多进程训练模型
@sync @distributed for i in 1:num_processes#定义多进程
    #定义每个进程的训练集数据量
    start_idx = (i - 1) * batch_train_x_size + 1
    end_idx = min(i * batch_train_x_size, size(train_x, 1))
    train_x_batch = train_x[start_idx:end_idx, :, :]
    train_y_batch = train_y[start_idx:end_idx, :]

    #定义每个进程的验证集数据量
    start_idx = (i - 1) * batch_val_x_size + 1
    end_idx = min(i * batch_val_x_size, size(val_x, 1))
    val_x_batch = val_x[start_idx:end_idx, :, :]
    val_y_batch = val_y[start_idx:end_idx, :]
    #定义损失函数数组
    train_losses_batch = Float32[]
    val_losses_batch = Float32[]
    #训练模型
    train_model(train_x_batch, train_y_batch, val_x_batch, val_y_batch, model, loss, opt, train_losses_batch, val_losses_batch)   
    train_losses_multi[:,i] = train_losses_batch
    val_losses_multi[:,i] = val_losses_batch
end

#合并多进程的损失函数数组
train_losses = mean(train_losses_multi, dims=2)
val_losses = mean(val_losses_multi, dims=2)



#绘制损失函数随时间的变化趋势图
p = plot(train_losses, xlabel="Time", ylabel="Loss", labels=["train_losses"], title="train Loss over time $window_size",size = (800,600))
plot!(p, val_losses, labels=["val_losses"])
losspng="loss_plot $window_size .png"
#检查文件是否存在
if ispath(losspng)
    #如果存在，删除文件
    rm(losspng)
    #打印删除成功的信息
    println("文件已删除")
else
    #如果不存在，打印文件不存在的信息
    println("文件不存在")
end

#保存图片
savefig(p,losspng)


end
run(`python3 sendmail.py`)
