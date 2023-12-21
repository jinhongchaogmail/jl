using CSV, DataFrames, Plots,Statistics

# 计算移动平均
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
    ma = calc_ma((df.close .* 2 .+ df.low .+ df.high) ./ 4, n)
    buy_signal = (df.close[n:end] > ma) .& (df.close[n:end] .< ma .* (1 + b / 100))
    if any(buy_signal)
        sell_signal = (1:length(buy_signal)) .> n .+ findfirst(buy_signal) .- 1 .+ b
        sell_signal = min.(sell_signal, length(buy_signal))
    else
        sell_signal = zeros(Int64, length(buy_signal))
    end
    return collect(buy_signal), sell_signal
end
function calc_returns(df::DataFrame, buy_signal::Vector{Bool}, sell_signal::Vector{Int64})
    returns = zeros(length(buy_signal))
    pos = 0
    for i = 1:length(buy_signal)
        if buy_signal[i]
            pos += 1
        end
        if (pos > 0) && (i == sell_signal[pos])
            returns[i] = df.close[i] / df.close[findfirst(buy_signal) + pos - 1] - 1
            pos -= 1
        end
    end
    return returns
end

# 画出收益率曲线
function plot_returns(returns::DataFrame, stock_code::String)
    plot(returns[:sell_date], cumprod(1 .+ returns[:return]), title="Returns of Stock $(stock_code)", xlabel="Date", ylabel="Cumulative Returns", legend=false)
end

# 主程序
function main(stock_code::String, n::Int64, b::Int64)
    df = CSV.read("$(stock_code).csv",DataFrame)
    buy_signal, sell_signal = calc_signals(df, n, b)
    returns = calc_returns(df, buy_signal, sell_signal)
    plot_returns(returns, stock_code)
end
main("data",2,5)
