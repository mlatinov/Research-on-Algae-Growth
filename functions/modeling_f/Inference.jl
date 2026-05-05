module Inference

using Turing, MCMCChains, ArviZ, ParetoSmooth
using Distributions, Statistics, DataFrames

export bm1_ppc, bm2_ppc, bm3_ppc, build_idata, loo_table, plot_ppchecks

# Internal helper: extract a parameter as an (iters, chains) matrix

_param(chain, name) = Array(chain[:, name, :])   # shape (iters, chains)

# Tobit forward simulation + pointwise log-likelihood
function _tobit_ppc(μ_fn, σ_mat::Matrix, Y_obs::Vector, n_chains::Int, iters::Int)
    n = length(Y_obs)
    Y_ppc   = Array{Float64}(undef, n_chains, iters, n)
    log_lik = Array{Float64}(undef, n_chains, iters, n)

    @inbounds for c in 1:n_chains, d in 1:iters
        μ = μ_fn(d, c)                         # length-n vector
        σ = σ_mat[d, c]                        # scalar

        # Forward simulation under the Tobit DGP: max(0, Normal)
        z = μ .+ σ .* randn(n)
        Y_ppc[c, d, :] .= max.(0.0, z)

        # Pointwise log-likelihood at observed Y
        for i in 1:n
            log_lik[c, d, i] = Y_obs[i] == 0.0 ?
                logcdf(Normal(μ[i], σ), 0.0) :
                logpdf(Normal(μ[i], σ), Y_obs[i])
        end
    end
    return Y_ppc, log_lik
end

# =====================================================================
# BM-1: Quadratic in Light only
function bm1_ppc(chain, df::DataFrame)
    I = df.Light;  Y = df.Population
    β₀ = _param(chain, :β₀);  β₁ = _param(chain, :β₁)
    β₂ = _param(chain, :β₂);  σ  = _param(chain, :σ)
    iters, n_chains = size(β₀)

    μ_fn = (d, c) -> β₀[d, c] .+ β₁[d, c] .* I .+ β₂[d, c] .* I.^2
    return _tobit_ppc(μ_fn, σ, Y, n_chains, iters)
end

# =====================================================================
# BM-2: Quadratic in Light + six linear predictors
function bm2_ppc(chain, df::DataFrame)
    I=df.Light; N=df.Nitrate; Fe=df.Iron; P=df.Phosphate
    T=df.Temperature; H=df.pH; C=df.CO2; Y=df.Population

    β₀=_param(chain,:β₀); β₁=_param(chain,:β₁); β₂=_param(chain,:β₂)
    β₃=_param(chain,:β₃); β₄=_param(chain,:β₄); β₅=_param(chain,:β₅)
    β₆=_param(chain,:β₆); β₇=_param(chain,:β₇); β₈=_param(chain,:β₈)
    σ =_param(chain,:σ)
    iters, n_chains = size(β₀)

    μ_fn = (d, c) -> β₀[d,c] .+ β₁[d,c].*I .+ β₂[d,c].*I.^2 .+
                     β₃[d,c].*N .+ β₄[d,c].*Fe .+ β₅[d,c].*P .+
                     β₆[d,c].*T .+ β₇[d,c].*H  .+ β₈[d,c].*C
    return _tobit_ppc(μ_fn, σ, Y, n_chains, iters)
end

# =====================================================================
# BM-3: Eilers–Peeters PI curve + linear secondary predictors
function bm3_ppc(chain, df::DataFrame)
    I=df.Light; N=df.Nitrate; Fe=df.Iron; P=df.Phosphate
    T=df.Temperature; H=df.pH; C=df.CO2; Y=df.Population
    Ī = mean(I)

    Is=_param(chain,:log_I_star); α0=_param(chain,:log_α₀); Pm=_param(chain,:log_Pm)
    δ₀=_param(chain,:δ₀); δ₁=_param(chain,:δ₁); δ₂=_param(chain,:δ₂)
    δ₃=_param(chain,:δ₃); δ₄=_param(chain,:δ₄); δ₅=_param(chain,:δ₅)
    δ₆=_param(chain,:δ₆); δ₇=_param(chain,:δ₇); σ=_param(chain,:σ)
    iters, n_chains = size(Is)

    μ_fn = function(d, c)
        γₑ = 1.0 / α0[d,c]
        α  = 1.0 / (α0[d,c] * Is[d,c]^2)
        βₑ = 1.0 / Pm[d,c] - 2.0 * sqrt(α * γₑ)

        fPI       = I ./ (α .* I.^2 .+ βₑ .* I .+ γₑ)
        fPI_peak  = Is[d,c] / (α * Is[d,c]^2 + βₑ * Is[d,c] + γₑ)
        fPIn      = fPI ./ fPI_peak
        fPI_mean  = Ī / (α * Ī^2 + βₑ * Ī + γₑ)
        fPIn_c    = fPIn .- (fPI_mean / Pm[d,c])

        return δ₀[d,c] .+ δ₁[d,c] .* fPIn_c .+
               δ₂[d,c].*N .+ δ₃[d,c].*Fe .+ δ₄[d,c].*P .+
               δ₅[d,c].*T .+ δ₆[d,c].*H  .+ δ₇[d,c].*C
    end
    return _tobit_ppc(μ_fn, σ, Y, n_chains, iters)
end

# =====================================================================
# Build the ArviZ InferenceData object that carries:

function build_idata(chain, Y_ppc, log_lik, Y_obs)
    Y_obs_p = Y_obs[!,:Population]
    return from_mcmcchains(
        chain;
        posterior_predictive = (Y = Y_ppc,),
        log_likelihood       = (Y = log_lik,),
        observed_data        = (Y = Y_obs_p,),
        coords               = Dict("obs" => 1:length(Y_obs_p)),
        dims                 = Dict("Y" => ["obs"]),
        library              = "Turing"
    )
end

# Tiny convenience: LOO from the idata
loo_table(idata) = ArviZ.loo(idata)

function plot_ppchecks(idata)
    # 1) Density overlay — does the marginal Y distribution match?
    density = plot_ppc(idata; data_pairs=Dict("Y"=>"Y"), num_pp_samples=100)
    
    # 2) Cumulative — best view for left skew and tail behaviour
    cumulative = plot_ppc(idata; kind="cumulative")
    
    # 3) Statistic-based — does the model reproduce specific features?
    stats = plot_ppc(idata; kind="stats", stats=["mean", "median", "std"])

    return Dict(
        :density => density,
        :cumulative => cumulative,
        :stats => stats
    )
end 

end 