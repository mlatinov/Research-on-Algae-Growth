module BSEM

using Turing 
using DataFrames  

# BM-4: SEM-A: Two-Factor Reflective CFA with Quadratic Light Structural Equation for Parametric Recovery 
export sema_tfr_cfa_qlse

function sema_tfr_cfa_qlse(data, iters = 1000)

    # Get the simulated data from the data dict 
    sim_data = data[:sim_data]
    Nᵢ = sim_data[!, :Nitrate]
    Hᵢ = sim_data[!, :pH]
    Pᵢ = sim_data[!, :Phosphate]
    Tᵢ = sim_data[!, :Temperature]   
    Cᵢ = sim_data[!, :CO2]
    Feᵢ = sim_data[!,:Iron]
    n = nrow(sim_data)
    Ī = mean(sim_data[!, :Light])

    # Standardise all secondary predictors 
    Nᵢ_s  = (Nᵢ  .- mean(Nᵢ))  ./ std(Nᵢ)
    Feᵢ_s = (Feᵢ .- mean(Feᵢ)) ./ std(Feᵢ)
    Pᵢ_s  = (Pᵢ  .- mean(Pᵢ))  ./ std(Pᵢ)
    Tᵢ_s  = (Tᵢ  .- mean(Tᵢ))  ./ std(Tᵢ)
    Hᵢ_s  = (Hᵢ  .- mean(Hᵢ))  ./ std(Hᵢ)
    Cᵢ_s  = (Cᵢ  .- mean(Cᵢ))  ./ std(Cᵢ)
    
    # Specify the model 
    @model function sema(Ιᵢ, Nᵢ, Feᵢ, Pᵢ, Tᵢ, Hᵢ, Cᵢ, Y, Ī)

        # Indicator Intercepts 
        μₙ  = 4.505   # mean Nitrate
        μFe = 0.105   # mean Iron
        μP  = 0.105   # mean Phosphate
        μT  = 20.0    # mean Temperature
        μH  = 7.5     # mean pH
        μC  = 6.0     # mean CO₂

        # Vatent Variables standard deviations
        ψₙ ~ truncated(Normal(1.0, 0.5), 0.2, 3.0)  # Nutrient factor variance
        ψᵪ ~ truncated(Normal(1.0, 0.5), 0.2, 3.0)   # Physico factor variance

        # Factor Loading 
        λ₂ ~ Normal(0.7, 0.3)   # Iron on nutrient factor
        λ₃ ~ Normal(0.7, 0.3)   # Phosphate on nutrient factor
        λ₅ ~ Normal(0.7, 0.3)   # pH on physicochemical factor
        λ₆ ~ Normal(0.7, 0.3)   # CO₂ on physicochemical factor

        # Measurement error standard deviations 
        θ₁ ~ truncated(Normal(0.7, 0.3), 0.05, 2.0) # Nitrate residual SD
        θ₂ ~ truncated(Normal(0.7, 0.3), 0.05, 2.0)  # Iron residual SD
        θ₃ ~ truncated(Normal(0.7, 0.3), 0.05, 2.0) # Phosphate residual SD
        θ₄ ~ truncated(Normal(0.7, 0.3), 0.05, 2.0) # Temperature residual SD
        θ₅ ~ truncated(Normal(0.7, 0.3), 0.05, 2.0) # pH residual SD
        θ₆ ~ truncated(Normal(0.7, 0.3), 0.05, 2.0) # CO₂ residual SD

        # Latent variables
        ηₙ ~ filldist(Normal(0.0, ψₙ), n)
        ηᵪ ~ filldist(Normal(0.0, ψᵪ), n)
        
        # Measurement model: nutrient block
        Turing.@addlogprob! sum(logpdf.(Normal.(μₙ  .+ 1.0 .* ηₙ, θ₁), Nᵢ_s))
        Turing.@addlogprob! sum(logpdf.(Normal.(μFe .+ λ₂ .* ηₙ, θ₂), Feᵢ_s))
        Turing.@addlogprob! sum(logpdf.(Normal.(μP  .+ λ₃ .* ηₙ, θ₃), Pᵢ_s))

        # Measurement model: physicochemical block 
        Turing.@addlogprob! sum(logpdf.(Normal.(μT  .+ 1.0 .* ηᵪ, θ₄), Tᵢ_s))
        Turing.@addlogprob! sum(logpdf.(Normal.(μH  .+ λ₅ .* ηᵪ, θ₅), Hᵢ_s))
        Turing.@addlogprob! sum(logpdf.(Normal.(μC  .+ λ₆ .* ηᵪ, θ₆), Cᵢ_s))
        
        # Structural Priors 
        β₀ ~ Normal(0.0, 1.0)
        β₁ ~ Normal(0.0, 0.5)
        β₂ ~ Normal(0.0, 0.2)
        γₙ ~ Normal(0.5, 0.3)
        γᵪ ~ Normal(0.5, 0.3)
        σ ~ truncated(Normal(500, 200), 50, 2000)

        # Structural Equation 
        μᵢ = β₀ .+ β₁ .* (Ιᵢ .- Ī) .+ β₂ .* (Ιᵢ .- Ī).^2 .+ γₙ .* ηₙ .+ γᵪ .* ηᵪ

        # Vectorized Tobit Likehood (Hope is faster)
        log_lik_censored = logcdf.(Normal.(μᵢ, σ), 0.0)
        log_lik_observed = logpdf.(Normal.(μᵢ, σ), Y)
        Turing.@addlogprob! sum(ifelse.(Y .<= 0.0, log_lik_censored, log_lik_observed))
    end

    # Specify the model with data 
    model_sema = sema(sim_data[!, :Light], Nᵢ_s,Feᵢ_s,Pᵢ_s,Tᵢ_s,Hᵢ_s,Cᵢ_s,sim_data[!, :Population],Ī)

    # Sample
    chain = sample(model_sema, NUTS(0.90), MCMCSerial(), iters, 4)
    return chain
end


end
