
# Libraries ##
using Dagger

# Include custom Modules ##
include("functions/helpers.jl/general_helpers.jl")
include("functions/eda_f/eda_f.jl")
using .General_helpers
using .EDA


# Pipeline Tasks

# Load the data 
algae_data = Dagger.@spawn(General_helpers.load_algae(url = "data/algeas.csv", sample_size = 1000));


# Exploratory Analysis 
algae_eda = Dagger.@spawn(EDA.eda_f(algae_data = algae_data));
 


# Run the Pipeline 
data = fetch(algae_data)

eda = fetch(algae_eda)
