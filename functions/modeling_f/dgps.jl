
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

# SEM A 
#=
SEM-A: Two-Factor Reflective CFA with Quadratic Light Structural Equation
Group the six secondary predictors into two latent constructs (ηN for
nutrients; ηC for physicochemical quality) via a reflective CFA measurement model.
The structural equation follows GLM-1 for the light effect, extended by the two
latent path coefficients.
=#
export dgp_sema

function dgp_sema(;n)

    # Measurement model: Nutrient block 
    # Draw the latent variables 
    ψₙ = rand(truncated(Normal(0, 1), 0, Inf), 100);
    ηₙᵢ = rand.(Normal.(0, ψₙ), 100);

    # Means Paramters coming from sample of 1000 EDA
    μₙ = 4.5
    μₑ = 0.105
    μₚ = 0.105

    # Lambda 
    λ₁ = 1 # Fixed Reference 
    λ₂ = rand(Normal(0, 1), 100)
    λ₃ = rand(Normal(0, 1), 100)

    # Theta Paramters for delta error term
    θ₁ = rand(truncated(Normal(0, 1), 0, Inf),100);
    θ₂ = rand(truncated(Normal(0, 1), 0, Inf),100);
    θ₃ = rand(truncated(Normal(0, 1), 0, Inf),100);

    # Delta error terms
    δ₁ᵢ = rand.(Normal.(0, θ₁));
    δ₂ᵢ = rand.(Normal.(0, θ₂));
    δ₃ᵢ = rand.(Normal.(0, θ₃));

    # Each observed nutrient is a noisy linear function of ηN,i:
    Nᵢ  = μₙ .+ λ₁ .* ηₙᵢ .+ δ₁ᵢ 
    Feᵢ = μₑ .+ λ₂ .* ηₙᵢ .+ δ₂ᵢ
    Pᵢ  = μₚ .+ λ₃ .* ηₙᵢ .+ δ₃ᵢ

    # Measurement model: Physicochemical block
    # Draw the latent variables 
    ψᵪ = rand(truncated(Normal(0, 1), 0, Inf), 100);
    ηᵪᵢ = rand.(Normal.(0, ψᵪ),100);

    # Mean Parameters 
    μₜ = 20.00;
    μₕ = 7.5;
    μᵪ  =  6.0;

    # Lambda 
    λ₄ = 1 # Fixed Reference 
    λ₅ = rand(Normal(0, 1), 100)
    λ₆ = rand(Normal(0, 1), 100)

    # Theta Paramters for delta error term
    θ₄ = rand(truncated(Normal(0, 1), 0, Inf),100);
    θ₅ = rand(truncated(Normal(0, 1), 0, Inf),100);
    θ₆ = rand(truncated(Normal(0, 1), 0, Inf),100);

    # Delta error terms
    δ₄ᵢ = rand.(Normal.(0, θ₄));
    δ₅ᵢ = rand.(Normal.(0, θ₅));
    δ₆ᵢ = rand.(Normal.(0, θ₆));
    
    # Each observed nutrient is a noisy linear function of ηᵪᵢ:
    Tᵢ = μₜ .+ λ₄ .* ηᵪᵢ .+ δ₄ᵢ
    Hᵢ = μₕ .+ λ₅ .* ηᵪᵢ .+ δ₅ᵢ
    Cᵢ = μᵪ .+ λ₆ .* ηᵪᵢ .+ δ₆ᵢ

    # Structural model.
    β₀ = rand(Normal(2500 , 1500), 100);
    β₁ = rand(truncated(Normal(0, 10), 0, Inf),100)
    β₂ = rand(truncated(Normal(0, 0.001), -Inf, 0), 100)
    Ιᵢ = rand(Uniform(40, 2000), 100);
    γₙ = rand(Normal(0, 50), 100);
    γᵪ = rand(Normal(0, 50), 100);
    ζ = rand(Normal(0, 10), 100);

    # Final Eq
    Yᵢ = β₀ .+ β₁ .*Ιᵢ + β₂ .* Ιᵢ.^2 .+ γₙ .* ηₙᵢ .+ γᵪ .* ηᵪᵢ + ζ;

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

#=
SEM-B: Two-Factor Reflective CFA with Eilers–Peeters Light Structural Equation
It combines the mechanistic Eilers–Peeters light response (from GLM-3) with the two-factor latent variable structure (from SEM-A).
=#
export dgp_semb

function dgp_semb(;
    n,
    μα  = 0, σ²α  = 10^-4,
    μβₑ = 0, σ²βₑ = 0.1,
    μγₑ = 0, σ²γₑ = 500, 
    μδ₀  = 0, σ²δ₀ = 500,     
    μδ₁  = 0, σ²δ₁ = 10^12,
    )

    # Priors from GLM-3 (for α, βEP, γEP, δ0, δ1)
    α  = rand(truncated(Normal(μα, σ²α), 0, Inf), n);
    βₑ = rand(truncated(Normal(μβₑ, σ²βₑ), 0, Inf), n);
    γₑ = rand(truncated(Normal(μγₑ, σ²γₑ), 0, Inf), n);
    Ιᵢ = rand(Uniform(40, 2000), 100);

    # Eilers–Peeters PI function
    fₑ(I, α, βₑ, γₑ) = I ./ (α .* I.^2 .+ βₑ .* I .+ γₑ);
    PIᵢ = fₑ.(Ιᵢ, α, βₑ, γₑ);

    # Measurement model: Nutrient block 
    # Draw the latent variables 
    ψₙ = rand(truncated(Normal(0, 1), 0, Inf), 100);
    ηₙᵢ = rand.(Normal.(0, ψₙ), 100);

    # Means Paramters coming from sample of 1000 EDA
    μₙ = 4.5
    μₑ = 0.105
    μₚ = 0.105

    # Lambda 
    λ₁ = 1 # Fixed Reference 
    λ₂ = rand(Normal(0, 1), 100)
    λ₃ = rand(Normal(0, 1), 100)

    # Theta Paramters for delta error term
    θ₁ = rand(truncated(Normal(0, 1), 0, Inf),100);
    θ₂ = rand(truncated(Normal(0, 1), 0, Inf),100);
    θ₃ = rand(truncated(Normal(0, 1), 0, Inf),100);

    # Delta error terms
    δ₁ᵢ = rand.(Normal.(0, θ₁));
    δ₂ᵢ = rand.(Normal.(0, θ₂));
    δ₃ᵢ = rand.(Normal.(0, θ₃));

    # Each observed nutrient is a noisy linear function of ηN,i:
    Nᵢ  = μₙ .+ λ₁ .* ηₙᵢ .+ δ₁ᵢ 
    Feᵢ = μₑ .+ λ₂ .* ηₙᵢ .+ δ₂ᵢ
    Pᵢ  = μₚ .+ λ₃ .* ηₙᵢ .+ δ₃ᵢ

    # Measurement model: Physicochemical block
    # Draw the latent variables 
    ψᵪ = rand(truncated(Normal(0, 1), 0, Inf), 100);
    ηᵪᵢ = rand.(Normal.(0, ψᵪ),100);

    # Mean Parameters 
    μₜ = 20.00;
    μₕ = 7.5;
    μᵪ  =  6.0;

    # Lambda 
    λ₄ = 1 # Fixed Reference 
    λ₅ = rand(Normal(0, 1), 100)
    λ₆ = rand(Normal(0, 1), 100)

    # Theta Paramters for delta error term
    θ₄ = rand(truncated(Normal(0, 1), 0, Inf),100);
    θ₅ = rand(truncated(Normal(0, 1), 0, Inf),100);
    θ₆ = rand(truncated(Normal(0, 1), 0, Inf),100);

    # Delta error terms
    δ₄ᵢ = rand.(Normal.(0, θ₄));
    δ₅ᵢ = rand.(Normal.(0, θ₅));
    δ₆ᵢ = rand.(Normal.(0, θ₆));
    
    # Each observed nutrient is a noisy linear function of ηᵪᵢ:
    Tᵢ = μₜ .+ λ₄ .* ηᵪᵢ .+ δ₄ᵢ
    Hᵢ = μₕ .+ λ₅ .* ηᵪᵢ .+ δ₅ᵢ
    Cᵢ = μᵪ .+ λ₆ .* ηᵪᵢ .+ δ₆ᵢ

    # Structural model.
    δ₀ = rand(Normal(μδ₀, σ²δ₀), n);
    δ₁ = rand(truncated(Normal(μδ₁, σ²δ₁), 0, Inf), n);
    γₙ = rand(Normal(0, 50), 100);
    γᵪ = rand(Normal(0, 50), 100);
    ζ = rand(Normal(0, 10), 100);

    # Final Eq
    Yᵢ = δ₀ .+ δ₁ .* PIᵢ .+ γₙ .* ηₙᵢ .+ γᵪ .* ηᵪᵢ + ζ;

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