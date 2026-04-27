
module General_helpers

#### Libraries ####
using DataFrames, DataFramesMeta, CSV
using DataFrames
using CSV
using Random
import StatsBase: sample
using Statistics

# Function to load the data in the main script and take a sample from it to speed up the development
export load_algae

function load_algae(;url, sample_size)
    # Laod the data 
    algae_data = CSV.read(url, DataFrame)
    # Take a sample 
    algae_data_sample =  sample(1:nrow(algae_data), sample_size, replace = false) 

    return algae_data[algae_data_sample,:]

end

# Function to Summarize the the algae sample data and return Dict for ploting 
export summarize_algae

function summarize_algae(;algae_data)

    # Zero Count and Near-Zero Proportion for Population
    nzc_population_algae = @chain algae_data begin
        @transform(
            nzc_population = ifelse.(:Population .< 50 , "near_zero" , "normal")  
        )
        @groupby(:nzc_population)
        @combine(
            median_light     = median(:Light),
            median_nitrate   = median(:Nitrate),
            median_phosphate = median(:Phosphate),
            median_iron      = median(:Iron),
            median_temp      = median(:Temperature),
            median_ph        = median(:pH),
            median_co2       = median(:CO2),
            count            = length(:nzc_population)
        )
    end
    # Corrlation matrix 
    cols = names(algae_data, Real)
    correlation_matrix = cor(Matrix(algae_data[:, cols]))
    correlation_matrix_df = DataFrame(correlation_matrix, cols)

    # Combine the results and return a Dict
    combined_results = Dict(
        :near_zero_population_summary => nzc_population_algae,
        :correlation_matirix          => correlation_matrix_df
    )

    return combined_results
end

end

