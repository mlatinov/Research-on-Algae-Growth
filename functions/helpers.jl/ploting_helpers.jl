
#### Module for Diffrent Ploting functions ####
module Ploting_helpers

using StatsPlots
using Plots

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
function plot_distributions(;algae_data)

end
end
