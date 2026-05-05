
#### Functions for Data Generative Processes 
module DGPs

using Distributions
using Statistics
using Distributions
using DataFrames

# GLM-1: Baseline Quadratic Regression (Light Only)
export dgp_quadratic_baseline

function dgp_quadratic_baseline(
    n = 1000,
    Ιᵢ_lower = 40,   
    Ιᵢ_upper = 2000,
    μβ₀  = 2500,     
    σβ₀ = 1500,     
    μβ₁  = 0,
    σβ₁ = 10,
    μβ₂  = 0,
    σβ₂ = 0.01,
    σμ = 0,
    σσ = 500
    )

    # Light Intensity for observation i 
    Ιᵢ = rand(Uniform(Ιᵢ_lower, Ιᵢ_upper), n);    # Ιᵢ ~ U(lower , upper)
 
    # Paramters 
    β₀  = rand(Normal(μβ₀, σβ₀));                      # β₀  ~  Ν(μ, σ) 
    β₁  = rand(truncated(Normal(μβ₁, σβ₁), 0, Inf));   # β₁  ~  Ν+(μ, σ)
    β₂  = rand(truncated(Normal(μβ₂, σβ₂), -Inf, 0));  # β₂  ~  Ν-(μ, σ)
    σ  = rand(truncated(Normal(σμ, σσ),0, Inf));       # σ²   ~  N(μ, σ)

    # Compute the conditional mean.
    μᵢ = β₀ .+ β₁ .* Ιᵢ .+ β₂ .* Ιᵢ.^2;

    # Draw observed population
    Yᵢ = max.(0.0, μᵢ .+ σ .* randn(n));

    # Combine into a Dataframe
    sim = DataFrame(Light = Ιᵢ, Population = Yᵢ);

    # Return the Global Paramters to check for parametric recovery 
    global_paramters = Dict(
        :β₀ => β₀,
        :β₁ => β₁,
        :β₂ => β₂,
        :σ => σ
    )
    return Dict(
        :sim_data => sim,
        :ground_truth => global_paramters
    )

end 

# GLM-2: Full Additive GLM (Quadratic Light + Six Linear Predictors)
export dgp_full_quadratic

function dgp_full_quadratic(
    n        = 1000,
    Ιᵢ_lower = 40.0,
    Ιᵢ_upper = 2000.0,
    μβ₀  = 2500.0,  σβ₀  = 1500.0,
    μβ₁  = 0.0,     σβ₁  = 10.0,   
    μβ₂  = 0.0,     σβ₂  = 0.01,
    σσ   = 500.0,
    μβⱼ  = 0.0,     σβⱼ  = 50.0    
    )
 
    # Observation-level covariates 
    Ιᵢ  = rand(Uniform(Ιᵢ_lower, Ιᵢ_upper), n)
    Feᵢ = rand(Uniform(0.01,  0.20), n)
    Tᵢ  = rand(Uniform(10.00, 30.0), n)
    Cᵢ  = rand(Uniform(2.00,  10.0), n)
    Nᵢ  = rand(Uniform(1.00,   8.0), n)
    Pᵢ  = rand(Uniform(0.01,  0.20), n)
    Hᵢ  = rand(Uniform(7.00,   8.0), n)
 
    # Model parameters
    β₀ = rand(Normal(μβ₀, σβ₀))
    β₁ = rand(truncated(Normal(μβ₁, σβ₁), 0.0, Inf))
    β₂ = rand(truncated(Normal(μβ₂, σβ₂), -Inf, 0.0))
    β₃ = rand(Normal(μβⱼ, σβⱼ))   # Nitrate effect
    β₄ = rand(Normal(μβⱼ, σβⱼ))   # Iron effect
    β₅ = rand(Normal(μβⱼ, σβⱼ))   # Phosphate effect
    β₆ = rand(Normal(μβⱼ, σβⱼ))   # Temperature effect
    β₇ = rand(Normal(μβⱼ, σβⱼ))   # pH effect
    β₈ = rand(Normal(μβⱼ, σβⱼ))   # CO₂ effect
    σ  = rand(truncated(Normal(0.0, σσ), 0.0, Inf))
 
    # Conditional mean 
    μᵢ = β₀ .+
        β₁ .* Ιᵢ .+ β₂ .* Ιᵢ.^2 .+
        β₃ .* Nᵢ .+ β₄ .* Feᵢ .+ β₅ .* Pᵢ .+
        β₆ .* Tᵢ .+ β₇ .* Hᵢ .+ β₈ .* Cᵢ
 
    # Observed population 
    Yᵢ = max.(0.0, μᵢ .+ σ .* randn(n))

    # Combine in a dataframe 
    sim = DataFrame(
        Population  = Yᵢ,
        Temperature = Tᵢ,
        Phosphate   = Pᵢ,
        Light       = Ιᵢ,
        Nitrate     = Nᵢ,
        Iron        = Feᵢ,
        pH          = Hᵢ,
        CO2         = Cᵢ
    )

    # Combine the model parameters as a ground truth
    global_parameters = Dict(
        :β₀ =>  β₀,
        :β₁ =>  β₁,
        :β₂ =>  β₂,
        :β₃ =>  β₃,
        :β₄ =>  β₄,
        :β₅ =>  β₅,
        :β₆ =>  β₆,
        :β₇ =>  β₇,
        :β₈ =>  β₈,
        :σ  =>  σ
    )
    return Dict(
        :sim_data => sim,
        :ground_truth => global_parameters
    )

end
# GLM-3: Eilers Peeters PI Curve + Linear Secondary Predictors
#=
Mechanistic Photosynthesis–Irradiance GLM  
The Eilers Peeters replaces the empirical quadratic with a mechanistically grounded photosynthesis–irradiance (PI) curve, 
=#

export dgp_eilers_peeters

function dgp_eilers_peeters(
    n        = 1000,
    Ιᵢ_lower = 40.0,
    Ιᵢ_upper = 2000.0;
    # Biological EP parameters 
    μI_star = 1000.0, σI_star = 300.0,
    I_star_lo = 100.0, I_star_hi = 2500.0,
    σα₀ = 0.05,
    σPm = 0.1,
    # Structural parameters 
    σδ₀  = 500.0,
    σδ₁  = 500.0,
    σδⱼ  = 50.0,
    σσ   = 500.0
    )

    # Environmental covariates
    Ιᵢ  = rand(Uniform(Ιᵢ_lower, Ιᵢ_upper), n)
    Nᵢ  = rand(Uniform(1.00,  8.00), n)
    Feᵢ = rand(Uniform(0.01,  0.20), n)
    Pᵢ  = rand(Uniform(0.01,  0.20), n)
    Tᵢ  = rand(Uniform(10.0, 30.0),  n)
    Hᵢ  = rand(Uniform(7.00,  8.00), n)
    Cᵢ  = rand(Uniform(2.00, 10.0),  n)

    # Draw biological EP parameters 
    I_star = rand(truncated(Normal(μI_star, σI_star), I_star_lo, I_star_hi))
    α₀     = rand(truncated(Normal(0.0, σα₀), 0.0, Inf))
    Pm     = rand(truncated(Normal(0.0, σPm),  0.0, Inf))

    # Recover the raw EP parameters 
    γₑ = 1.0 / α₀
    α  = γₑ / I_star^2
    βₑ = 1.0/Pm - 2.0 * sqrt(α * γₑ)

    # if βₑ ≤ 0 the curve is not unimodal 
    attempts = 0
    while βₑ <= 0.0
        attempts += 1
        attempts > 1000 && error("Could not draw valid EP parameters after 1000 attempts. " *
                                  "Check that σPm is small enough (Pm < α₀·I_star/2 required).")
        I_star = rand(truncated(Normal(μI_star, σI_star), I_star_lo, I_star_hi))
        α₀     = rand(truncated(Normal(0.0, σα₀), 0.0, Inf))
        Pm     = rand(truncated(Normal(0.0, σPm),  0.0, Inf))
        γₑ = 1.0 / α₀
        α  = γₑ / I_star^2
        βₑ = 1.0/Pm - 2.0 * sqrt(α * γₑ)
    end

    # Normalised PI curve 
    fPI      = Ιᵢ ./ (α .* Ιᵢ.^2 .+ βₑ .* Ιᵢ .+ γₑ)
    fPI_peak = I_star / (α * I_star^2 + βₑ * I_star + γₑ)
    fPIn     = fPI ./ fPI_peak
    fPIn_s   = (fPIn .- mean(fPIn)) ./ std(fPIn)

    # Standardise all secondary predictors too 
    Nᵢ_s  = (Nᵢ  .- mean(Nᵢ))  ./ std(Nᵢ)
    Feᵢ_s = (Feᵢ .- mean(Feᵢ)) ./ std(Feᵢ)
    Pᵢ_s  = (Pᵢ  .- mean(Pᵢ))  ./ std(Pᵢ)
    Tᵢ_s  = (Tᵢ  .- mean(Tᵢ))  ./ std(Tᵢ)
    Hᵢ_s  = (Hᵢ  .- mean(Hᵢ))  ./ std(Hᵢ)
    Cᵢ_s  = (Cᵢ  .- mean(Cᵢ))  ./ std(Cᵢ)

    # Structural parameters
    σ  = rand(truncated(Normal(0.0, σσ), 0.0, Inf))
    δ₀ = rand(Normal(0.0, σδ₀))
    δ₁ = rand(truncated(Normal(500.0, 500.0), max(200.0, 2*σ), Inf))
    δ₂ = rand(Normal(0.0, σδⱼ))   # Nitrate
    δ₃ = rand(Normal(0.0, σδⱼ))   # Iron
    δ₄ = rand(Normal(0.0, σδⱼ))   # Phosphate
    δ₅ = rand(Normal(0.0, σδⱼ))   # Temperature
    δ₆ = rand(Normal(0.0, σδⱼ))   # pH
    δ₇ = rand(Normal(0.0, σδⱼ))   # CO₂

    # Conditional mean 
    μᵢ = δ₀ .+ δ₁ .* fPIn_s .+
         δ₂ .* Nᵢ_s .+ δ₃ .* Feᵢ_s .+ δ₄ .* Pᵢ_s .+
         δ₅ .* Tᵢ_s .+ δ₆ .* Hᵢ_s .+ δ₇ .* Cᵢ_s

    # Observed population 
    Yᵢ = max.(0.0, μᵢ .+ σ .* randn(n))

    # Ground truth
    param_recovery = Dict{Symbol, Float64}(
        :I_star => I_star, :α₀ => α₀,   :Pm  => Pm,
        :α      => α,      :βₑ => βₑ,   :γₑ  => γₑ,
        :δ₀     => δ₀,     :δ₁ => δ₁,   :δ₂  => δ₂,
        :δ₃     => δ₃,     :δ₄ => δ₄,   :δ₅  => δ₅,
        :δ₆     => δ₆,     :δ₇ => δ₇,   :σ   => σ,
    )

    sim_data = DataFrame(
        Population  = Yᵢ,
        Light       = Ιᵢ,
        Nitrate     = Nᵢ,
        Iron        = Feᵢ,
        Phosphate   = Pᵢ,
        Temperature = Tᵢ,
        pH          = Hᵢ,
        CO2         = Cᵢ
    )

    return Dict{Symbol, Any}(
        :sim_data     => sim_data,
        :ground_truth => param_recovery
    )
end

# SEM A 
#=
SEM-A: Two-Factor Reflective CFA with Quadratic Light Structural Equation
Group the six secondary predictors into two latent constructs (ηN for
nutrients; ηC for physicochemical quality) via a reflective CFA measurement model.
The structural equation follows GLM-1 for the light effect, extended by the two
latent path coefficients.
=#
export dgp_sema

function dgp_sema(
    n        = 1000,
    # Structural model
    μβ₀  = 2500.0,  σβ₀  = 1500.0,
    μβ₁  = 0.0,     σβ₁  = 10.0,
    μβ₂  = 0.0,     σβ₂  = 0.01,
    Ι_lower = 40.0,  Ι_upper = 2000.0,
    μγₙ  = 0.0,     σγₙ  = 50.0,
    μγᵪ  = 0.0,     σγᵪ  = 50.0,
    σσ   = 500.0,
    # Indicator intercepts 
    μₙ   = 4.505,   μₑ   = 0.105,  μₚ   = 0.105,
    μₜ   = 20.0,    μₕ   = 7.5,    μᵪ   = 6.0,
    # Latent variable variance priors 
    σψₙ  = 3.0,     
    σψᵪ  = 8.0,     
    # Loading priors  
    σλ₂  = 0.05,    # Fe loading
    σλ₃  = 0.05,    # P loading
    σλ₅  = 0.10,    # pH loading
    σλ₆  = 0.40,    # CO₂ loading
    # Measurement error std priors 
    σθ₁  = 1.5,     # Nitrate residual std-dev prior
    σθ₂  = 0.03,    # Iron residual
    σθ₃  = 0.03,    # Phosphate residual
    σθ₄  = 3.0,     # Temperature residual
    σθ₅  = 0.30,    # pH residual
    σθ₆  = 1.5      # CO₂ residual
    )
 
    # Structural parameters
    β₀ = rand(Normal(μβ₀, σβ₀))
    β₁ = rand(truncated(Normal(μβ₁, σβ₁), 0.0, Inf))
    β₂ = rand(truncated(Normal(μβ₂, σβ₂), -Inf, 0.0))
    γₙ = rand(Normal(μγₙ, σγₙ))
    γᵪ = rand(Normal(μγᵪ, σγᵪ))
    σ  = rand(truncated(Normal(0.0, σσ), 0.0, Inf))
 
    # Latent variable variances
    ψₙ = rand(truncated(Normal(0.0, σψₙ), 0.0, Inf))   # Nutrient factor variance
    ψᵪ = rand(truncated(Normal(0.0, σψᵪ), 0.0, Inf))   # Physico factor variance
 
    # Factor loadings
    λ₁ = 1.0                         # fixed reference (Nitrate)
    λ₂ = rand(Normal(0.0, σλ₂))      # Iron on Nutrient factor
    λ₃ = rand(Normal(0.0, σλ₃))      # Phosphate on Nutrient factor
    λ₄ = 1.0                         # fixed reference (Temperature)
    λ₅ = rand(Normal(0.0, σλ₅))      # pH on Physico factor
    λ₆ = rand(Normal(0.0, σλ₆))      # CO₂ on Physico factor
 
    # Measurement error std-devs
    θ₁ = rand(truncated(Normal(0.0, σθ₁), 0.0, Inf))
    θ₂ = rand(truncated(Normal(0.0, σθ₂), 0.0, Inf))
    θ₃ = rand(truncated(Normal(0.0, σθ₃), 0.0, Inf))
    θ₄ = rand(truncated(Normal(0.0, σθ₄), 0.0, Inf))
    θ₅ = rand(truncated(Normal(0.0, σθ₅), 0.0, Inf))
    θ₆ = rand(truncated(Normal(0.0, σθ₆), 0.0, Inf))
 
    # Observation-level draws 
    Ιᵢ  = rand(Uniform(Ι_lower, Ι_upper), n)
    ηₙᵢ = rand(Normal(0.0, ψₙ), n)    # Latent nutrient quality, per obs
    ηᵪᵢ = rand(Normal(0.0, ψᵪ), n)    # Latent physico quality, per obs
 
    # Nutrient measurement residuals 
    δ₁ᵢ = rand(Normal(0.0, θ₁), n)
    δ₂ᵢ = rand(Normal(0.0, θ₂), n)
    δ₃ᵢ = rand(Normal(0.0, θ₃), n)
 
    # Physico measurement residuals
    δ₄ᵢ = rand(Normal(0.0, θ₄), n)
    δ₅ᵢ = rand(Normal(0.0, θ₅), n)
    δ₆ᵢ = rand(Normal(0.0, θ₆), n)
 
    # Measurement model Nutrient block 
    Nᵢ  = μₙ .+ λ₁ .* ηₙᵢ .+ δ₁ᵢ
    Feᵢ = μₑ .+ λ₂ .* ηₙᵢ .+ δ₂ᵢ
    Pᵢ  = μₚ .+ λ₃ .* ηₙᵢ .+ δ₃ᵢ
 
    # Measurement model Physicochemical block 
    Tᵢ  = μₜ .+ λ₄ .* ηᵪᵢ .+ δ₄ᵢ
    Hᵢ  = μₕ .+ λ₅ .* ηᵪᵢ .+ δ₅ᵢ
    Cᵢ  = μᵪ .+ λ₆ .* ηᵪᵢ .+ δ₆ᵢ

    # Structural model 
    Yᵢ = β₀ .+ β₁ .* Ιᵢ .+ β₂ .* Ιᵢ.^2 .+
        γₙ .* ηₙᵢ .+ γᵪ .* ηᵪᵢ .+
        σ .* randn(n)

    Yᵢ = max.(0.0, Yᵢ)

    # Collect the model paramters for recovery 
    param_recovery = Dict{Symbol, Any}(
        # Structural equation
        :β₀ => β₀,  :β₁ => β₁,  :β₂ => β₂,
        :γₙ => γₙ,  :γᵪ => γᵪ,  :σ  => σ,
        
        # Latent variable scales
        :ψₙ => ψₙ,  :ψᵪ => ψᵪ,
        
        # Free loadings
        :λ₂ => λ₂,  :λ₃ => λ₃,
        :λ₅ => λ₅,  :λ₆ => λ₆,
        
        # Measurement error SDs
        :θ₁ => θ₁,  :θ₂ => θ₂,  :θ₃ => θ₃,
        :θ₄ => θ₄,  :θ₅ => θ₅,  :θ₆ => θ₆,
    )
    # Store latent vectors separately for diagnostic plots
    latent_truth = Dict{Symbol, Vector{Float64}}(
        :eta_N => ηₙᵢ,
        :eta_C => ηᵪᵢ,
    )

    # Retrun the simulated data 
    sim_data = DataFrame(
        Population  = Yᵢ,
        Light       = Ιᵢ,
        Nitrate     = Nᵢ,
        Iron        = Feᵢ,
        Phosphate   = Pᵢ,
        Temperature = Tᵢ,
        pH          = Hᵢ,
        CO2         = Cᵢ
    )
    
    return Dict(
        :sim_data      => sim_data,
        :ground_truth  => param_recovery,
        :latent_truth  => latent_truth
    ) 
end

#=
SEM-B: Two-Factor Reflective CFA with Eilers–Peeters Light Structural Equation
It combines the mechanistic Eilers–Peeters light response (from GLM-3) with the two-factor latent variable structure (from SEM-A).
=#
export dgp_semb

function dgp_semb(
    n = 1000;
    # EP biological parameterisation 
    μI_star  = 1000.0, σI_star  = 300.0,
    I_star_lo = 100.0, I_star_hi = 2500.0,
    σα₀      = 0.05,
    σPm      = 0.1,
    # Structural parameters
    σδ₀ = 500.0, σδ₁ = 2000.0,
    σγₙ = 50.0,  σγᵪ = 50.0,
    σσ  = 500.0,
    # Latent variable SDs
    σψₙ = 3.0,  σψᵪ = 8.0,
    # Free factor loadings 
    σλ₂ = 0.05, σλ₃ = 0.05,
    σλ₅ = 0.10, σλ₆ = 0.40,
    # Measurement error SDs 
    σθ₁ = 1.5,  σθ₂ = 0.03, σθ₃ = 0.03,
    σθ₄ = 3.0,  σθ₅ = 0.30, σθ₆ = 1.5,
    # Indicator intercepts
    μₙ = 4.505, μFe = 0.105, μP = 0.105,
    μT = 20.0,  μH  = 7.5,   μC = 6.0,
    # Light range 
    Ι_lower = 40.0, Ι_upper = 2000.0
)

    # Draw biological EP parameters 
    I_star = rand(truncated(Normal(μI_star, σI_star), I_star_lo, I_star_hi))
    α₀     = rand(truncated(Normal(0.0, σα₀), 0.0, Inf))
    Pm     = rand(truncated(Normal(0.0, σPm),  0.0, Inf))

    # Recover raw EP parameters analytically 
    γₑ = 1.0 / α₀
    α  = γₑ / I_star^2
    βₑ = 1.0/Pm - 2.0 * sqrt(α * γₑ)

    # Reject draws where βₑ ≤ 0 
    attempts = 0
    while βₑ <= 0.0
        attempts += 1
        attempts > 1000 && error("Could not draw valid EP parameters after 1000 attempts.")
        I_star = rand(truncated(Normal(μI_star, σI_star), I_star_lo, I_star_hi))
        α₀     = rand(truncated(Normal(0.0, σα₀), 0.0, Inf))
        Pm     = rand(truncated(Normal(0.0, σPm),  0.0, Inf))
        γₑ = 1.0 / α₀
        α  = γₑ / I_star^2
        βₑ = 1.0/Pm - 2.0 * sqrt(α * γₑ)
    end

    # Structural parameters 
    δ₀ = rand(Normal(0.0,  σδ₀))
    δ₁ = rand(truncated(Normal(0.0, σδ₁), 0.0, Inf))
    γₙ = rand(Normal(0.0,  σγₙ))
    γᵪ = rand(Normal(0.0,  σγᵪ))
    σ  = rand(truncated(Normal(0.0, σσ),  0.0, Inf))

    # Latent variable SDs 
    ψₙ = rand(truncated(Normal(0.0, σψₙ), 0.0, Inf))
    ψᵪ = rand(truncated(Normal(0.0, σψᵪ), 0.0, Inf))

    # Factor loadings 
    λ₂ = rand(Normal(0.0, σλ₂))
    λ₃ = rand(Normal(0.0, σλ₃))
    λ₅ = rand(Normal(0.0, σλ₅))
    λ₆ = rand(Normal(0.0, σλ₆))

    #  Measurement error SDs 
    θ₁ = rand(truncated(Normal(0.0, σθ₁), 0.0, Inf))
    θ₂ = rand(truncated(Normal(0.0, σθ₂), 0.0, Inf))
    θ₃ = rand(truncated(Normal(0.0, σθ₃), 0.0, Inf))
    θ₄ = rand(truncated(Normal(0.0, σθ₄), 0.0, Inf))
    θ₅ = rand(truncated(Normal(0.0, σθ₅), 0.0, Inf))
    θ₆ = rand(truncated(Normal(0.0, σθ₆), 0.0, Inf))

    # Observation-level draws 
    Ιᵢ  = rand(Uniform(Ι_lower, Ι_upper), n)
    ηₙᵢ = rand(Normal(0.0, ψₙ), n)
    ηᵪᵢ = rand(Normal(0.0, ψᵪ), n)

    # Measurement model: Nutrient block 
    Nᵢ  = μₙ  .+ 1.0 .* ηₙᵢ .+ rand(Normal(0.0, θ₁), n)
    Feᵢ = μFe .+ λ₂  .* ηₙᵢ .+ rand(Normal(0.0, θ₂), n)
    Pᵢ  = μP  .+ λ₃  .* ηₙᵢ .+ rand(Normal(0.0, θ₃), n)

    # Measurement model: Physicochemical block 
    Tᵢ  = μT .+ 1.0 .* ηᵪᵢ .+ rand(Normal(0.0, θ₄), n)
    Hᵢ  = μH .+ λ₅  .* ηᵪᵢ .+ rand(Normal(0.0, θ₅), n)
    Cᵢ  = μC .+ λ₆  .* ηᵪᵢ .+ rand(Normal(0.0, θ₆), n)

    # Normalised PI curve
    fPI      = Ιᵢ ./ (α .* Ιᵢ.^2 .+ βₑ .* Ιᵢ .+ γₑ)
    fPI_peak = I_star / (α * I_star^2 + βₑ * I_star + γₑ)  
    fPIn     = fPI ./ fPI_peak

    # Centred PI curve breaks δ₀–δ₁ funnel kinda.. i hope it does 
    Ī        = mean(Ιᵢ)
    fPI_mean = Ī / (α * Ī^2 + βₑ * Ī + γₑ)
    fPIn_c   = fPIn .- (fPI_mean / fPI_peak)

    # Structural equation 
    μᵢ = δ₀ .+ δ₁ .* fPIn_c .+ γₙ .* ηₙᵢ .+ γᵪ .* ηᵪᵢ
    Yᵢ = max.(0.0, μᵢ .+ σ .* randn(n))

    # Ground truth
    param_recovery = Dict{Symbol, Any}(
        # EP biological parameters 
        :I_star => I_star, :α₀ => α₀, :Pm => Pm,
        # EP raw parameters 
        :α => α, :βₑ => βₑ, :γₑ => γₑ,
        # Structural equation
        :δ₀ => δ₀, :δ₁ => δ₁,
        :γₙ => γₙ, :γᵪ => γᵪ, :σ => σ,
        # Latent variable scales
        :ψₙ => ψₙ, :ψᵪ => ψᵪ,
        # Free loadings 
        :λ₂ => λ₂, :λ₃ => λ₃,
        :λ₅ => λ₅, :λ₆ => λ₆,
        # Measurement error SDs
        :θ₁ => θ₁, :θ₂ => θ₂, :θ₃ => θ₃,
        :θ₄ => θ₄, :θ₅ => θ₅, :θ₆ => θ₆,
    )
    sim_data = DataFrame(
        Population  = Yᵢ,
        Light       = Ιᵢ,
        Nitrate     = Nᵢ,
        Iron        = Feᵢ,
        Phosphate   = Pᵢ,
        Temperature = Tᵢ,
        pH          = Hᵢ,
        CO2         = Cᵢ
    )

    return Dict{Symbol, Any}(
        :sim_data     => sim_data,
        :ground_truth => param_recovery
    )
end

end