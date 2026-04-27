
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
    histogram_features = Ploting_helpers.plot_distributions(algae_data = algae_data)

    # Combine the results in Dict
    eda_results = Dict(
        :summary_stat            => algae_summary,
        :log_population_plot     => population_log_plot,
        :features_distributions  => histogram_features
    )
    return eda_results
end


end 
