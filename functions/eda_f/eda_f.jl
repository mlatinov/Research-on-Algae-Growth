
# Modules for EDA analysis 
module EDA
# Include custom Modules ##
include("../helpers.jl/general_helpers.jl")
include("../helpers.jl/ploting_helpers.jl")
using .General_helpers
using .Ploting_helpers

#### Function to Combine summary Statiscs and plots ####
export eda_f
function eda_f(;algae_data)
    # Summary Statics 
    algae_summary = General_helpers.summarize_algae(algae_data = algae_data);
    # Plot Population Vs Log(Population)
    population_log_plot = Ploting_helpers.plot_y_vs_log_y(algae_data = algae_summary[:log_population])

    # Plot Histograms of features 
    hist_light   = Ploting_helpers.plot_distributions(algae_data = algae_data, x = "Light")
    hist_nitrate = Ploting_helpers.plot_distributions(algae_data = algae_data, x = "Nitrate")
    hist_iron    = Ploting_helpers.plot_distributions(algae_data = algae_data, x = "Iron")
    hist_ph  = Ploting_helpers.plot_distributions(algae_data = algae_data, x = "pH")
    hist_co2 = Ploting_helpers.plot_distributions(algae_data = algae_data, x = "Temperature")
    hist_phosphate   = Ploting_helpers.plot_distributions(algae_data = algae_data, x = "Phosphate")
    hist_temperature = Ploting_helpers.plot_distributions(algae_data = algae_data, x = "Temperature")

    # Plot Scatter Plots 
    combined_scatter = Ploting_helpers.plot_scatters(algae_data = algae_data)

    # Plot Correlation Matrix 
    correlation_heatmap = Ploting_helpers.corr_heatmap(algae_data = algae_data)

    # Plot Single Scatterplots 
    scatter_light_x_population = Ploting_helpers.single_scatter(algae_data = algae_data, x = "Light", y = "Population")

    # Combine all histograms in a histogram dictionary 
    features_histograms = Dict(
        :histogram_light   => hist_light,
        :histogram_nitrate => hist_nitrate,
        :histogram_iron    => hist_iron,
        :histogram_phosphate => hist_phosphate,
        :histogram_ph        => hist_ph,
        :histogram_co2       => hist_co2,
        :histogram_temperature => hist_temperature
    )
    # Combine the results in Dict
    eda_results = Dict(
        :summary_stat         => algae_summary,
        :log_population_plot  => population_log_plot,
        :features_histograms  => features_histograms,
        :combined_scatter     => combined_scatter,
        :correlation_heatmap  => correlation_heatmap,
        :scatter_light_x_population => scatter_light_x_population
    )
    return eda_results
end

end 
