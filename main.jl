
# Libraries ##
using Dagger
using DataFrames, DataFramesMeta
using Chain

# Include custom Modules ##
include("functions/helpers.jl/general_helpers.jl")
using .General_helpers


# Pipeline Tasks

# Load the data 
algae_data = Dagger.@spawn(General_helpers.load_algae(url = "data/algeas.csv", sample_size = 1000))

# Summary Statics 
algae_summary = Dagger.@spawn(General_helpers.summarize_algae(algae_data = algae_data)) 



# Run the Pipeline 
algae_data_fetch = fetch(algae_summary)

algae_data_fetch[:correlation_matirix]




