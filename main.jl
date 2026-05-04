# Libraries 
using Dagger
using DataFrames

# Include custom Modules ##
include("functions/helpers.jl/general_helpers.jl")
include("functions/eda_f/eda_f.jl")
include("functions/modeling_f/dgps.jl")
include("functions/modeling_f/bglm_models.jl")
include("functions/modeling_f/bsem_models.jl")
include("functions/modeling_f/diagnostics.jl")
using .General_helpers
using .EDA
using .DGPs
using .BGLM
using .BSEM
using .Diagnostics

# Pipeline Tasks

# Load the data 
algae_data = Dagger.@spawn(General_helpers.load_algae("data/algeas.csv", 3000));


# Exploratory Analysis 
algae_eda = Dagger.@spawn(EDA.eda_f(algae_data));

# Data Generative Processs ==========================================================================================

# DGP 1 for the Family of GLMs Baseline Quadratic Regression (Light Only)
sim_dgp_1 = Dagger.@spawn(DGPs.dgp_quadratic_baseline(1000));

# DGP 2 for the Falily of GLMs  Full Additive GLM (Quadratic Light + Six Linear Predictors)
sim_dgp_2 = Dagger.@spawn(DGPs.dgp_full_quadratic(1000));

# DGP 3 for the Family of GLMs Eilers Peeters PI Curve + Linear Secondary Predictors
sim_dgp_3 = Dagger.@spawn(DGPs.dgp_eilers_peeters(2000));

# DGP 4 for the Family of SEMs Two-Factor Reflective CFA with Quadratic Light Structural Equation
sim_dgp_4 = Dagger.@spawn(DGPs.dgp_sema(1000));

# DGP 5 for the Family of SEMs  Two-Factor Reflective CFA with Eilers–Peeters Light Structural Equation
sim_dgp_5 = Dagger.@spawn(DGPs.dgp_semb(1000));

# Model Specifications ==============================================================================================

# BM-1: Baseline Quadratic Regression (Light Only) Test in simulation against DGP 1 for Parametric Recovery 
bqrl_sim = Dagger.@spawn(BGLM.bayes_quadratic_regreesion_light(sim_dgp_1, 1000));

# BM-2: Bayesian Full Additive GLM (Quadratic Light + Six Linear Predictors) for Parametric Recovery 
bfag_sim = Dagger.@spawn(BGLM.bayes_full_quadratic_regreesion(sim_dgp_2, 1000));

# BM-3: Eilers Peeters PI Curve + Linear Secondary Predictors for Parametric Recovery 
bepl_sim = Dagger.@spawn(BGLM.bepl_full(sim_dgp_3, 2000));

# BM-4: SEM-A: Two-Factor Reflective CFA with Quadratic Light Structural Equation for Parametric Recovery 
bsema_sim = Dagger.@spawn(BSEM.sema_tfr_cfa_qlse(sim_dgp_4, 2000));

# BM-5: SEM-B: Two-Factor Reflective CFA with Eilers–Peeters Light Structural Equation for Parametric Recovery 
bsemb_sim = Dagger.@spawn(BSEM.semb_tfr_cfa_ep_light(sim_dgp_5, 1000));

# Run the Model with the Actual Data ==================================================================================

# BM-1: Baseline Quadratic Regression (Light Only) Real Data 
bqrl_real = Dagger.@spawn(BGLM.bayes_quadratic_regreesion_light(algae_data, 1000, false));

# BM-2: Bayesian Full Additive GLM (Quadratic Light + Six Linear Predictors) Real data 
bfag_real = Dagger.@spawn(BGLM.bayes_full_quadratic_regreesion(algae_data, 1000, false));

# BM-3: Eilers Peeters PI Curve + Linear Secondary Predictors for Parametric Recovery 
bepl_real = Dagger.@spawn(BGLM.bepl_full(algae_data, 1000, false));

# Convergent Diagnostics ==============================================================================================
bqrl_real_conv = Dagger.@spawn(Diagnostics.conv_diagnostics(bqrl_real, [:β₀,:β₁,:β₂,:σ])); # Fine 
bfag_real_conv = Dagger.@spawn(Diagnostics.conv_diagnostics(bfag_real, [:β₀,:β₁,:β₂,:σ])); # Fine 
bepl_real_conv = Dagger.@spawn(Diagnostics.conv_diagnostics(bepl_real, [:I_star,:Pm,:α₀,:σ])); # Fuck off



check_model = fetch(bepl_sim_fixed)
describe(check_model)
truth = fetch(sim_dgp_3)
a = truth[:ground_truth]


example = fetch(bepl_real_conv)
conv_example = fetch(bepl_real_conv)
conv_example[:autocor]

# Posterior Predictive Checks =========================================================================================



