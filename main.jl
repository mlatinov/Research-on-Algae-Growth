
# Libraries ##
using Dagger

# Include custom Modules ##
include("functions/helpers.jl/general_helpers.jl")
include("functions/eda_f/eda_f.jl")
include("functions/modeling_f/dgps.jl")
using .General_helpers
using .EDA
using .DGPs


# Pipeline Tasks

# Load the data 
algae_data = Dagger.@spawn(General_helpers.load_algae(url = "data/algeas.csv", sample_size = 1000));


# Exploratory Analysis 
algae_eda = Dagger.@spawn(EDA.eda_f(algae_data = algae_data));
 
# DGP 1 for the Family of GLMs Baseline Quadratic Regression (Light Only)
dgp_1 = Dagger.@spawn(DGPs.dgp_quadratic_baseline(n = 1000));

# DGP 2 for the Falily of GLMs  Full Additive GLM (Quadratic Light + Six Linear Predictors)
dgp_2 = Dagger.@spawn(DGPs.dgp_full_quadratic(n = 1000));

# DGP 3 for the Family of GLMs Eilers Peeters PI Curve + Linear Secondary Predictors
dgp_3 = Dagger.@spawn(DGPs.dgp_eilers_peeters(n = 1000));

# DGP 4 for the Family of SEMs Two-Factor Reflective CFA with Quadratic Light Structural Equation
dgp_4 = Dagger.@spawn(DGPs.dgp_sema(n = 1000));

# DGP 5 for the Family of SEMs  Two-Factor Reflective CFA with Eilers–Peeters Light Structural Equation
dgp_5 = Dagger.@spawn(DGPs.dgp_semb(n =  1000));

# Run the Pipeline 
data = fetch(algae_data)
dgp_1_data = fetch(dgp_1)
dgp_2_data = fetch(dgp_2)
dgp_3_data = fetch(dgp_3)
dgp_4_data = fetch(dgp_4)
dgp_5_data = fetch(dgp_5)

eda = fetch(algae_eda)
using DataFrames
using DataVoyager
describe(data)
describe(dgp_1_data)
describe(dgp_2_data)
describe(dgp_3_data)
describe(dgp_4_data)
describe(dgp_5_data)

v1 = Voyager(dgp_5_data)