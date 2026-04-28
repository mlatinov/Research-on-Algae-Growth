
#### Functions for Data Generative Processes 
module DGPs

using Statistics
using Distributions
using DataFrames

# GLM-1: Baseline Quadratic Regression (Light Only)
export dgp_quadratic_baseline

function dgp_quadratic_baseline(;
    n = 1000,
    ɭᵢ_lower = 40,   
    ɭᵢ_upper = 2000,
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
    ɭᵢ = rand(Uniform(ɭᵢ_lower, ɭᵢ_upper), n);    # ɭᵢ ~ U(lower , upper)
    
    # Paramters 
    β₀  = rand(Normal(μβ₀, σ²β₀));                         # β₀  ~  Ν(μ, σ) 
    β₁  = rand(truncated(Normal(μβ₁, σ²β₁), 0, Inf), n);   # β₁  ~  Ν+(μ, σ)
    β₂  = rand(truncated(Normal(μβ₂, σ²β₂), -Inf, 0), n);  # β₂  ~  Ν-(μ, σ)
    σ²  = rand(truncated(Normal(σ²μ, σ²σ²),0, Inf), n);    # σ²   ~  N(μ, σ)

    # Compute the conditional mean.
    μᵢ = β₀ .+ β₁ .* ɭᵢ .+ β₂ .* ɭᵢ.^2;

    # Draw observed population
    Yᵢ = max.(0.0, μᵢ .+ rand(Normal(0, σ²)),n);

    # Combine into a Dataframe
    sim = DataFrame(Light = ɭᵢ, Population = Yᵢ);

    return sim
end 

end