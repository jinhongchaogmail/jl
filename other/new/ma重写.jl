#以下是一个简单的 Julia 代码示例，展示如何使用您提供的回测方案进行股票回测：

using CSV
using DataFrames

# 读取股票数据

df = CSV.read("data.csv",DataFrame)

# 计算移动平均线
function calc_ma(data::Vector{Float64}, n::Int64)
    ma = zeros(length(data) - n + 1)
    for i = n:length(data)
        ma[i - n + 1] = mean(data[(i - n + 1):i])
    end
    return ma
end

# 计算买卖信号
function calc_signals(df::DataFrame, n::Int64, b::Int64)
    ma = calc_ma((df[:c] .* 2 .+ df[:l] .+ df[:h]) ./ 4, n)
    buy_signal = (df[:c][n:end] > ma) & (df[:c][n:end] .< ma .* (1 + b / 100))
    sell_signal = (1:length(buy_signal)) .> n .+ findfirst(buy_signal) .- 1 .+ b
    sell_signal = min.(sell_signal, length(buy_signal))
    return buy_signal, sell_signal
end

# 计算收益率
function calc_returns(df::DataFrame, buy_signal::Vector{Bool}, sell_signal::Vector{Int64})
    returns = zeros(length(buy_signal))
    pos = 0
    for i = 1:length(buy_signal)
        if buy_signal[i]
            pos += 1
        end
        if (pos > 0) && (i == sell_signal[pos])
            returns[i] = df[:c][i] / df[:c][findfirst(buy_signal) + pos - 1] - 1
            pos -= 1
        end
    end
    return returns
end

# 回测
best_n = 0
best_b = 0
best_return = -Inf
for n = 2:30
    for b = 2:50
        buy_signal, sell_signal = calc_signals(df, n, b)
        returns = calc_returns(df, buy_signal, sell_signal)
        total_return = prod(1 + returns) - 1
        if total_return > best_return
            best_n = n
            best_b = b
            best_return = total_return
        end
    end
end

# 输出最佳参数和收益率
println("Best n: ", best_n)
println("Best b: ", best_b)
println("Best total return: ", best_return)
#需要注意的是，以上代码只是一个简单的示例，可能无法满足您的所有需求。如果您需要更复杂的功能，可以进一步修改和优化代码。

