
using CSV
using Flux

# 读取股票数据
df = CSV.read("data.csv")

# 定义模型
function model(x, n, b)
    ma = (x[:, 1] .* 2 .+ x[:, 2] .+ x[:, 3]) ./ 4 |> MA(n)
    buy_signal = (x[:, 4:end] .> ma) .& (x[:, 4:end] .< ma .* (1 + b / 100))
    sell_signal = ((1:size(buy_signal, 2)) .> n .+ findfirst(sum(buy_signal, dims=2) .> 0) .- 1 .+ b) .| (size(buy_signal, 2) .== findfirst(sum(buy_signal, dims=2) .> 0))
    sell_signal = min.(sell_signal, size(buy_signal, 2))
    return buy_signal, sell_signal
end

# 定义损失函数
function loss(params, x, y)
    buy_signal_pred, sell_signal_pred = model(x, params[1], params[2])
    returns_pred = calc_returns(df, buy_signal_pred, sell_signal_pred)
    total_return_pred = prod(1 + returns_pred) - 1
    return -total_return_pred
end

# 计算收益率
function calc_returns(df, buy_signal, sell_signal)
    returns = zeros(size(buy_signal, 1), size(buy_signal, 2))
    pos = zeros(size(buy_signal, 1))
    for i = 1:size(buy_signal, 2)
        pos[buy_signal[:, i]] .= pos[buy_signal[:, i]] .+ 1
        sel = pos .> 0 .& i .== sell_signal .+ pos .- 1
        returns[sel, i] = df[sel, :c] ./ df[findfirst(sel) .+ pos[sel] .- 1, :c] .- 1
        pos[sel] .= pos[sel] .- 1
    end
    return returns
end

# 训练模型
best_n = 0
best_b = 0
best_return = -Inf
for n = 2:30
    for b = 2:50
        params = [n, b]
        gs = Flux.Optimise.GridSearch(range(0.1, 1.0, length=10), range(0.1, 1.0, length=10))
        result = Flux.Optimise.optimize(() -> loss(params, df[:, [:c, :l, :h, Symbol.(1:50)]]), gs, verbose=false)
        if -result.minimum > best_return
            best_n = n
            best_b = b
            best_return = -result.minimum
        end
    end
end

# 输出最佳参数和收益率
println("Best n: ", best_n)
println("Best b: ", best_b)
println("Best total return: ", best_return)


