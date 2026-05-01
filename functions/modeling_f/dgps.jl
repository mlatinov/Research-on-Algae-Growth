
#### Functions for Data Generative Processes 
module DGPs

using Distributions
using Statistics
using Distributions
using DataFrames

# GLM-1: Baseline Quadratic Regression (Light Only)
export dgp_quadratic_baseline

function dgp_quadratic_baseline(;
    n = 1000,
    Ιᵢ_lower = 40,   
    Ιᵢ_upper = 2000,
    μβ₀  = 2500,     
    σ²β₀ = 1500,     
    μβ₁  = 0,
    σ²β₁ = 10,
    μβ₂  = 0,
    σ²β₂ = 0.01,
    σ²μ = 0,
    σ²σ² = 500
    )

    # Light Intensity for observation i 
    Ιᵢ = rand(Uniform(Ιᵢ_lower, Ιᵢ_upper), n);    # Ιᵢ ~ U(lower , upper)
 
    # Paramters 
    β₀  = rand(Normal(μβ₀, σ²β₀));                      # β₀  ~  Ν(μ, σ) 
    β₁  = rand(truncated(Normal(μβ₁, σ²β₁), 0, Inf));   # β₁  ~  Ν+(μ, σ)
    β₂  = rand(truncated(Normal(μβ₂, σ²β₂), -Inf, 0));  # β₂  ~  Ν-(μ, σ)
    σ²  = rand(truncated(Normal(σ²μ, σ²σ²),0, Inf));    # σ²   ~  N(μ, σ)

    # Compute the conditional mean.
    μᵢ = β₀ .+ β₁ .* Ιᵢ .+ β₂ .* Ιᵢ.^2;

    # Draw observed population
    Yᵢ = max.(0.0, μᵢ .+ σ² .* randn(n));

    # Combine into a Dataframe
    sim = DataFrame(Light = Ιᵢ, Population = Yᵢ);

    return sim
end 

# GLM-2: Full Additive GLM (Quadratic Light + Six Linear Predictors)
export dgp_full_quadratic

function dgp_full_quadratic(;
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
 
    return DataFrame(
        Population  = Yᵢ,
        Temperature = Tᵢ,
        Phosphate   = Pᵢ,
        Light       = Ιᵢ,
        Nitrate     = Nᵢ,
        Iron        = Feᵢ,
        pH          = Hᵢ,
        CO2         = Cᵢ
    )
end
# GLM-3: Eilers Peeters PI Curve + Linear Secondary Predictors
#=
Mechanistic Photosynthesis–Irradiance GLM  
The Eilers Peeters replaces the empirical quadratic with a mechanistically grounded photosynthesis–irradiance (PI) curve, 
=#

export dgp_eilers_peeters
 
function dgp_eilers_peeters(;
    n        = 1000,
    Ιᵢ_lower = 40.0,
    Ιᵢ_upper = 2000.0,
    μα   = 0.0,  σα   = 1e-4,   #  photoinhibition curvature
    μβₑ  = 0.0,  σβₑ  = 0.1,    #  light-saturation term
    μγₑ  = 0.0,  σγₑ  = 100.0,  
    μδ₀  = 0.0,  σδ₀  = 500.0,  # intercept 
    μδ₁  = 0.0,  σδ₁  = 2000.0, 
    μδⱼ  = 0.0,  σδⱼ  = 50.0,   # secondary predictor coefficients
    σσ   = 500.0
    )
 
    # Observation  covariates 
    Ιᵢ  = rand(Uniform(Ιᵢ_lower, Ιᵢ_upper), n)
    Feᵢ = rand(Uniform(0.01,  0.20), n)
    Tᵢ  = rand(Uniform(10.00, 30.0), n)
    Cᵢ  = rand(Uniform(2.00,  10.0), n)
    Nᵢ  = rand(Uniform(1.00,   8.0), n)
    Pᵢ  = rand(Uniform(0.01,  0.20), n)
    Hᵢ  = rand(Uniform(7.00,   8.0), n)
 
    # EP curve parameters
    α  = rand(truncated(Normal(μα,  σα),  0.0, Inf))
    βₑ = rand(truncated(Normal(μβₑ, σβₑ), 0.0, Inf))
    γₑ = rand(truncated(Normal(μγₑ, σγₑ), 0.0, Inf))
 
    # EP curve evaluated 
    PIᵢ = Ιᵢ ./ (α .* Ιᵢ.^2 .+ βₑ .* Ιᵢ .+ γₑ)
 
    # Structural parameters:
    δ₀ = rand(Normal(μδ₀, σδ₀))
    δ₁ = rand(truncated(Normal(μδ₁, σδ₁), 0.0, Inf)) 
    δ₂ = rand(Normal(μδⱼ, σδⱼ))
    δ₃ = rand(Normal(μδⱼ, σδⱼ))
    δ₄ = rand(Normal(μδⱼ, σδⱼ))
    δ₅ = rand(Normal(μδⱼ, σδⱼ))
    δ₆ = rand(Normal(μδⱼ, σδⱼ))
    δ₇ = rand(Normal(μδⱼ, σδⱼ))
    σ  = rand(truncated(Normal(0.0, σσ), 0.0, Inf))
 
    # Conditional mean 
    μᵢ = δ₀ .+ δ₁ .* PIᵢ .+
        δ₂ .* Nᵢ .+ δ₃ .* Feᵢ .+ δ₄ .* Pᵢ .+
        δ₅ .* Tᵢ .+ δ₆ .* Hᵢ .+ δ₇ .* Cᵢ
 
    # Observed population 
    Yᵢ = max.(0.0, μᵢ .+ σ .* randn(n))
 
    return DataFrame(
        Population  = Yᵢ,
        Temperature = Tᵢ,
        Phosphate   = Pᵢ,
        Light       = Ιᵢ,
        Nitrate     = Nᵢ,
        Iron        = Feᵢ,
        pH          = Hᵢ,
        CO2         = Cᵢ
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

function dgp_sema(;
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
 
    return DataFrame(
        Population  = Yᵢ,
        Temperature = Tᵢ,
        Phosphate   = Pᵢ,
        Light       = Ιᵢ,
        Nitrate     = Nᵢ,
        Iron        = Feᵢ,
        pH          = Hᵢ,
        CO2         = Cᵢ
    )
end

#=
SEM-B: Two-Factor Reflective CFA with Eilers–Peeters Light Structural Equation
It combines the mechanistic Eilers–Peeters light response (from GLM-3) with the two-factor latent variable structure (from SEM-A).
=#
export dgp_semb

function dgp_semb(;
    n  = 1000,
    # EP curve parameters
    μα   = 0.0,  σα   = 1e-4,
    μβₑ  = 0.0,  σβₑ  = 0.1,
    μγₑ  = 0.0,  σγₑ  = 100.0,  
    # Structural parameters
    μδ₀  = 0.0,  σδ₀  = 500.0,
    μδ₁  = 0.0,  σδ₁  = 2000.0, 
    μγₙ  = 0.0,  σγₙ  = 50.0,
    μγᵪ  = 0.0,  σγᵪ  = 50.0,
    Ι_lower = 40.0,  Ι_upper = 2000.0,
    σσ   = 500.0,
    # Indicator intercepts
    μₙ   = 4.505,   μₑ   = 0.105,  μₚ   = 0.105,
    μₜ   = 20.0,    μₕ   = 7.5,    μᵪ   = 6.0,
    # Latent variable variance priors  
    σψₙ  = 3.0,
    σψᵪ  = 8.0,
    # Loading priors  
    σλ₂  = 0.05,
    σλ₃  = 0.05,
    σλ₅  = 0.10,
    σλ₆  = 0.40,
    # Measurement error std priors
    σθ₁  = 1.5,
    σθ₂  = 0.03,
    σθ₃  = 0.03,
    σθ₄  = 3.0,
    σθ₅  = 0.30,
    σθ₆  = 1.5
    )
 
    # EP curve parameters
    α  = rand(truncated(Normal(μα,  σα),  0.0, Inf))
    βₑ = rand(truncated(Normal(μβₑ, σβₑ), 0.0, Inf))
    γₑ = rand(truncated(Normal(μγₑ, σγₑ), 0.0, Inf))
 
    # Structural parameters
    δ₀ = rand(Normal(μδ₀, σδ₀))
    δ₁ = rand(truncated(Normal(μδ₁, σδ₁), 0.0, Inf))
    γₙ = rand(Normal(μγₙ, σγₙ))
    γᵪ = rand(Normal(μγᵪ, σγᵪ))
    σ  = rand(truncated(Normal(0.0, σσ), 0.0, Inf))
 
    # Latent variances
    ψₙ = rand(truncated(Normal(0.0, σψₙ), 0.0, Inf))
    ψᵪ = rand(truncated(Normal(0.0, σψᵪ), 0.0, Inf))
 
    # Loadings
    λ₁ = 1.0
    λ₂ = rand(Normal(0.0, σλ₂))
    λ₃ = rand(Normal(0.0, σλ₃))
    λ₄ = 1.0
    λ₅ = rand(Normal(0.0, σλ₅))
    λ₆ = rand(Normal(0.0, σλ₆))
 
    # Measurement error std-devs
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
 
    # Measurement residuals 
    δ₁ᵢ = rand(Normal(0.0, θ₁), n)
    δ₂ᵢ = rand(Normal(0.0, θ₂), n)
    δ₃ᵢ = rand(Normal(0.0, θ₃), n)
    δ₄ᵢ = rand(Normal(0.0, θ₄), n)
    δ₅ᵢ = rand(Normal(0.0, θ₅), n)
    δ₆ᵢ = rand(Normal(0.0, θ₆), n)
 
    # Measurement model: Nutrient block 
    Nᵢ  = μₙ .+ λ₁ .* ηₙᵢ .+ δ₁ᵢ
    Feᵢ = μₑ .+ λ₂ .* ηₙᵢ .+ δ₂ᵢ
    Pᵢ  = μₚ .+ λ₃ .* ηₙᵢ .+ δ₃ᵢ
 
    # Measurement model: Physicochemical block 
    Tᵢ  = μₜ .+ λ₄ .* ηᵪᵢ .+ δ₄ᵢ
    Hᵢ  = μₕ .+ λ₅ .* ηᵪᵢ .+ δ₅ᵢ
    Cᵢ  = μᵪ .+ λ₆ .* ηᵪᵢ .+ δ₆ᵢ
 
    # EP curve 
    PIᵢ = Ιᵢ ./ (α .* Ιᵢ.^2 .+ βₑ .* Ιᵢ .+ γₑ)
 
    # Structural model 
    Yᵢ = δ₀ .+ δ₁ .* PIᵢ .+
         γₙ .* ηₙᵢ .+ γᵪ .* ηᵪᵢ .+
         σ .* randn(n)   
    Yᵢ = max.(0.0, Yᵢ)
 
    return DataFrame(
        Population  = Yᵢ,
        Temperature = Tᵢ,
        Phosphate   = Pᵢ,
        Light       = Ιᵢ,
        Nitrate     = Nᵢ,
        Iron        = Feᵢ,
        pH          = Hᵢ,
        CO2         = Cᵢ
    )
end

end