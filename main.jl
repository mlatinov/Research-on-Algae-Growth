# Libraries 
using Dagger
using DataFrames
using ArviZ

# Include custom Modules ##
include("functions/helpers.jl/general_helpers.jl")
include("functions/eda_f/eda_f.jl")
include("functions/modeling_f/dgps.jl")
include("functions/modeling_f/bglm_models.jl")
include("functions/modeling_f/bsem_models.jl")
include("functions/modeling_f/diagnostics.jl")
include("functions/modeling_f/Inference.jl")
using .General_helpers
using .EDA
using .DGPs
using .BGLM
using .BSEM
using .Diagnostics
using .Inference

# Pipeline Tasks

# Load the data 
algae_data = Dagger.@spawn(General_helpers.load_algae("data/algeas.csv", 5000));


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
sim_dgp_4 = Dagger.@spawn(DGPs.dgp_sema(500));

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
bsema_sim = Dagger.@spawn(BSEM.sema_tfr_cfa_qlse(sim_dgp_4, 2000)); # Not tested

# BM-5: SEM-B: Two-Factor Reflective CFA with Eilers–Peeters Light Structural Equation for Parametric Recovery 
bsemb_sim = Dagger.@spawn(BSEM.semb_tfr_cfa_ep_light(sim_dgp_5, 1000)); # Not tested Not Implemented only DGP exists 

# Run the Model with the Actual Data ==================================================================================

# BM-1: Baseline Quadratic Regression (Light Only) Real Data 
bqrl_real = Dagger.@spawn(BGLM.bayes_quadratic_regreesion_light(algae_data, 2000, false));

# BM-2: Bayesian Full Additive GLM (Quadratic Light + Six Linear Predictors) Real data 
bfag_real = Dagger.@spawn(BGLM.bayes_full_quadratic_regreesion(algae_data, 2000, false));

# BM-3: Eilers Peeters PI Curve + Linear Secondary Predictors for Parametric Recovery 
bepl_real = Dagger.@spawn(BGLM.bepl_full(algae_data, 2000, false));

# Convergent Diagnostics ==============================================================================================
bqrl_real_conv = Dagger.@spawn(Diagnostics.conv_diagnostics(bqrl_real, [:β₀,:β₁,:β₂,:σ]));                 # Fine 
bfag_real_conv = Dagger.@spawn(Diagnostics.conv_diagnostics(bfag_real, [:β₀,:β₁,:β₂,:σ]));                 # Fine 
bepl_real_conv = Dagger.@spawn(Diagnostics.conv_diagnostics(bepl_real, [:log_I_star,:log_Pm,:log_α₀,:σ])); # Fine 

# Posterior Predictive checks 
ppc_bm1 = Dagger.@spawn Inference.bm1_ppc(bqrl_real, algae_data);
ppc_bm2 = Dagger.@spawn Inference.bm2_ppc(bfag_real, algae_data);
ppc_bm3 = Dagger.@spawn Inference.bm3_ppc(bepl_real, algae_data);

# Fetch 
ppc_bm1_f = fetch(ppc_bm1);
ppc_bm2_f = fetch(ppc_bm2);
ppc_bm3_f = fetch(ppc_bm3);

# Wrap into ArviZ idata objects
idata_bm1 = Dagger.@spawn Inference.build_idata(bqrl_real, ppc_bm1_f[1], ppc_bm1_f[2], algae_data);
idata_bm2 = Dagger.@spawn Inference.build_idata(bfag_real,ppc_bm2_f[1], ppc_bm2_f[2], algae_data);
idata_bm3 = Dagger.@spawn Inference.build_idata(bepl_real,ppc_bm3_f[1], ppc_bm3_f[2], algae_data);

# Fetch all the data 
idata1 = fetch(idata_bm1);
idata2 = fetch(idata_bm2);
idata3 = fetch(idata_bm3);

# Plot PPC 
plots_ppc_idata_bm1 = Dagger.@spawn Inference.plot_ppchecks(idata1);
plots_ppc_idata_bm2 = Dagger.@spawn Inference.plot_ppchecks(idata2);
plots_ppc_idata_bm3 = Dagger.@spawn Inference.plot_ppchecks(idata3);

# Fetch 
plots_ppc_idata1 = fetch(idata_bm1);
plots_ppc_idata2 = fetch(idata_bm2);
plots_ppc_idata3 = fetch(idata_bm3);

plots_ppc_idata1[:sample_stats]