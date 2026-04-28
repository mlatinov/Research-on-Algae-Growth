
#### Functions for Data Generative Processes 
module DGPs

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
    σ²β₁ = 20,
    μβ₂  = 0,
    σ²β₂ = 0.01,
    σ²μ = 0,
    σ²σ² = 500
    )

    # Light Intensity for observation i 
    Ιᵢ = rand(Uniform(Ιᵢ_lower, Ιᵢ_upper), n);    # Ιᵢ ~ U(lower , upper)
 
    # Paramters 
    β₀  = rand(Normal(μβ₀, σ²β₀), n);                      # β₀  ~  Ν(μ, σ) 
    β₁  = rand(truncated(Normal(μβ₁, σ²β₁), 0, Inf), n);   # β₁  ~  Ν+(μ, σ)
    β₂  = rand(truncated(Normal(μβ₂, σ²β₂), -Inf, 0), n);  # β₂  ~  Ν-(μ, σ)
    σ²  = rand(truncated(Normal(σ²μ, σ²σ²),0, Inf), n);    # σ²   ~  N(μ, σ)

    # Compute the conditional mean.
    μᵢ = β₀ .+ β₁ .* Ιᵢ .+ β₂ .* Ιᵢ.^2;

    # Draw observed population
    Yᵢ = max.(0.0, μᵢ .+ rand(Normal(0, σ²),n));

    # Combine into a Dataframe
    sim = DataFrame(Light = Ιᵢ, Population = Yᵢ);

    return sim
end 

# GLM-2: Full Additive GLM (Quadratic Light + Six Linear Predictors)
export dgp_full_quadratic

function dgp_full_quadratic(;
    n = 1000,
    Ιᵢ_lower = 40,   
    Ιᵢ_upper = 2000,
    μβ₀  = 2500,     
    σ²β₀ = 1500,     
    μβ₁  = 0,
    σ²β₁ = 20,
    μβ₂  = 0,
    σ²β₂ = 0.01,
    σ²μ  = 0,
    σ²σ² = 500,
    μβⱼ  = 0,
    σ²βⱼ = 50
    )

    # Sample predictors 
    Ιᵢ  = rand(Uniform(Ιᵢ_lower, Ιᵢ_upper), n);
    Feᵢ = rand(Uniform(0.01, 0.20), n);
    Tᵢ  = rand(Uniform(10.00, 30), n);
    Cᵢ  = rand(Uniform(2.00, 10), n);
    Nᵢ  = rand(Uniform(1.00, 8), n);
    Pᵢ  = rand(Uniform(0.01, 0.20), n);
    Hᵢ  = rand(Uniform(7.00, 8.00),n);

    # Paramters 
    β₀ = rand(Normal(μβ₀, σ²β₀), n)
    β₁ = rand(truncated(Normal(μβ₁, σ²β₁), 0, Inf), n)
    β₂ = rand(truncated(Normal(μβ₂, σ²β₂), -Inf, 0), n)
    β₃ = rand(Uniform(μβⱼ, σ²βⱼ), n)
    β₄ = rand(Uniform(μβⱼ, σ²βⱼ), n)
    β₅ = rand(Uniform(μβⱼ, σ²βⱼ), n)
    β₆ = rand(Uniform(μβⱼ, σ²βⱼ), n)
    β₇ = rand(Uniform(μβⱼ, σ²βⱼ), n)
    β₈ = rand(Uniform(μβⱼ, σ²βⱼ), n)
    σ² = rand(truncated(Normal(σ²μ, σ²σ²), 0, Inf), n)

    # Compute the conditional mean
    μᵢ = β₀ .+
      β₁ .* Ιᵢ .+
      β₂ .* Ιᵢ.^2 .+
      β₃ .* Nᵢ .+
      β₄ .* Feᵢ .+
      β₅ .* Pᵢ .+
      β₆ .* Tᵢ .+
      β₇ .* Hᵢ .+
      β₈ .* Cᵢ
    
    # Draw observed population
    Yᵢ = max(0.00, μᵢ .+ rand(Normal(0, σ²), n))

    # Combine everythin into dataframe 
    sim = DataFrame(
        Population  = Yᵢ,
        Temperature = Tᵢ,
        Phosphate   = Pᵢ,
        Light   = Ιᵢ,
        Nitrate = Nᵢ,
        Iron    = Feᵢ,
        pH  = Hᵢ,
        CO2 = Cᵢ
    )
    return sim
end

# GLM-3: Eilers Peeters PI Curve + Linear Secondary Predictors
#=
Mechanistic Photosynthesis–Irradiance GLM  
The Eilers Peeters replaces the empirical quadratic with a mechanistically grounded photosynthesis–irradiance (PI) curve, 
=#

export dgp_eilers_peeters

function dgp_eilers_peeters(;
    n = 1000,
    Ιᵢ_lower = 40,   
    Ιᵢ_upper = 2000,
    μα  = 0, σ²α  = 10^-4,
    μβₑ = 0, σ²βₑ = 0.1,
    μγₑ = 0, σ²γₑ = 500, 
    μδ₀  = 0, σ²δ₀ = 500,     
    μδ₁  = 0, σ²δ₁ = 10^12,
    μγⱼ  = 0, σ²γⱼ = 50,
    σ²μ  = 0, σ²σ² = 500,
    )

    # Sample predictors 
    Ιᵢ  = rand(Uniform(Ιᵢ_lower, Ιᵢ_upper), n);
    Feᵢ = rand(Uniform(0.01, 0.20), n);
    Tᵢ  = rand(Uniform(10.00, 30), n);
    Cᵢ  = rand(Uniform(2.00, 10), n);
    Nᵢ  = rand(Uniform(1.00, 8), n);
    Pᵢ  = rand(Uniform(0.01, 0.20), n);
    Hᵢ  = rand(Uniform(7.00, 8.00),n);

    # Parameters
    α  = rand(truncated(Normal(μα, σ²α), 0, Inf), n);
    βₑ = rand(truncated(Normal(μβₑ, σ²βₑ), 0, Inf), n);
    γₑ = rand(truncated(Normal(μγₑ, σ²γₑ), 0, Inf), n);

    # Eilers–Peeters PI function
    fₑ(I, α, βₑ, γₑ) = I ./ (α .* I.^2 .+ βₑ .* I .+ γₑ);
    PIᵢ = fₑ.(Ιᵢ, α, βₑ, γₑ);

    δ₀ = rand(Normal(μδ₀, σ²δ₀), n);
    δ₁ = rand(truncated(Normal(μδ₁, σ²δ₁), 0, Inf), n);
    δ₂ = rand(Normal(μγⱼ, σ²γⱼ), n)
    δ₃ = rand(Normal(μγⱼ, σ²γⱼ), n)
    δ₄ = rand(Normal(μγⱼ, σ²γⱼ), n)
    δ₅ = rand(Normal(μγⱼ, σ²γⱼ), n)
    δ₆ = rand(Normal(μγⱼ, σ²γⱼ), n)
    δ₇ = rand(Normal(μγⱼ, σ²γⱼ), n)
    σ² = rand(truncated(Normal(σ²μ, σ²σ²),0 ,Inf), n)

    # Compute the conditional mean
    μᵢ = δ₀ .+ δ₁ .* PIᵢ .+ 
        δ₂ .* Nᵢ .+ 
        δ₃ .* Feᵢ.+ 
        δ₄ .* Pᵢ .+ 
        δ₅ .* Tᵢ .+ 
        δ₆ .* Hᵢ .+ 
        δ₇ .* Cᵢ

    # Draw observed population
    Yᵢ = max(0.00, μᵢ .+ rand(Normal(0, σ²), n))

    # Combine everythin into dataframe 
    sim = DataFrame(
        Population  = Yᵢ,
        Temperature = Tᵢ,
        Phosphate   = Pᵢ,
        Light   = Ιᵢ,
        Nitrate = Nᵢ,
        Iron    = Feᵢ,
        pH  = Hᵢ,
        CO2 = Cᵢ
    )
    return sim
end

end