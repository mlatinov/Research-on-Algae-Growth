
#### Module for Diffrent Ploting functions ####
module Ploting_helpers

using StatsPlots
using Plots
using Statistics

# Histogram of Y and log(Y)
export plot_y_vs_log_y
function plot_y_vs_log_y(;algae_data)

    # Function ploting theme 
    theme(:rose_pine)

    # Plot histogram of Y 
    hist_y = @df algae_data histogram(
        :Population,
        title  = "Population in a native Scale",
        xlabel = "Population native",
        ylabel = "Count"
    )
    # Plot histogram of Log(y)
    hist_log_y = @df algae_data histogram(
        :log_population,
        title  = "Log Scaled Population",
        xlabel = "Log(Population)",
        ylabel = "Count" 
    )
    # Combine the plots side by side
    combined = plot(hist_y, hist_log_y, layout = (1,2))
    return combined
end

# Histogram of features distributions 
export plot_distributions

function plot_distributions(; algae_data, x)
    theme(:rose_pine)

    col = x isa Symbol ? x : Symbol(x)

    hist_plot = histogram(
        algae_data[!, col],
        title = "Histogram of $col",
        xlabel = string(col),
        ylabel = "Counts",
        size = (400, 400)
    )
    return hist_plot
end

#### Function to plot Combined Scatter Plots ####
export plot_scatters

function plot_scatters(;algae_data)
    combined_scatter = cornerplot(
        Matrix(algae_data),
        title = "Combined Algae Cornerplot",
        size  = (1400, 1400),
        compact = true
    )
    return combined_scatter
end

#### Function to plot a heatmap with correlation matrix x#### 
export corr_heatmap

function corr_heatmap(; algae_data)

    # Create a simple correlation matrix
    cols = names(algae_data, Real)
    corr_mat = cor(Matrix(algae_data[:, cols]))

    # Plot it 
    corr_heatmap = heatmap(
        cols,
        cols,
        corr_mat,
        title = "Algae Correlation Heatmap",
        c = :coolwarm
    )
    return corr_heatmap
end

#### Function to plot a single Scatterplot ####
export single_scatter

function single_scatter(; algae_data, x, y)

    x_symbol = x isa Symbol ? x : Symbol(x)
    y_symbol = y isa Symbol ? y : Symbol(y)

    single_scatter = scatter(
        algae_data[!, x_symbol],
        algae_data[!, y_symbol]
    )
    return single_scatter
end


end
