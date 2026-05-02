module BSEM

using Turing 
using DataFrames

# BM-4: SEM-A: Two-Factor Reflective CFA with Quadratic Light Structural Equation for Parametric Recovery 
export sema_tfr_cfa_qlse

function sema_tfr_cfa_qlse(data, iters = 1000)

    # Get the simulated data from the data dict 
    sim_data = data[:sim_data]
    n = nrow(sim_data)
    Ī = mean(sim_data[!, :Light])

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
        ψₙ ~ truncated(Normal(0.0, 3.0), 0.0, Inf)  # Nutrient factor variance
        ψᵪ ~ truncated(Normal(0.0, 8.0), 0.0, Inf)  # Physico factor variance

        # Factor Loading 
        λ₂ ~ Normal(0.0, 1.0)   # Iron on nutrient factor
        λ₃ ~ Normal(0.0, 1.0)   # Phosphate on nutrient factor
        λ₅ ~ Normal(0.0, 1.0)   # pH on physicochemical factor
        λ₆ ~ Normal(0.0, 1.0)   # CO₂ on physicochemical factor

        # Measurement error standard deviations 
        θ₁ ~ truncated(Normal(0.0, 1.0), 0.0, Inf)  # Nitrate residual SD
        θ₂ ~ truncated(Normal(0.0, 1.0), 0.0, Inf)  # Iron residual SD
        θ₃ ~ truncated(Normal(0.0, 1.0), 0.0, Inf)  # Phosphate residual SD
        θ₄ ~ truncated(Normal(0.0, 1.0), 0.0, Inf)  # Temperature residual SD
        θ₅ ~ truncated(Normal(0.0, 1.0), 0.0, Inf)  # pH residual SD
        θ₆ ~ truncated(Normal(0.0, 1.0), 0.0, Inf)  # CO₂ residual SD

        # Latent variables
        ηₙ ~ filldist(Normal(0.0, ψₙ), n)
        ηᵪ ~ filldist(Normal(0.0, ψᵪ), n)

        # Measurement model: nutrient block
        Turing.@addlogprob! sum(logpdf.(Normal.(μₙ  .+ 1.0 .* ηₙ, θ₁), Nᵢ))
        Turing.@addlogprob! sum(logpdf.(Normal.(μFe .+ λ₂ .* ηₙ, θ₂), Feᵢ))
        Turing.@addlogprob! sum(logpdf.(Normal.(μP  .+ λ₃ .* ηₙ, θ₃), Pᵢ))

        # Measurement model: physicochemical block 
        Turing.@addlogprob! sum(logpdf.(Normal.(μT  .+ 1.0 .* ηᵪ, θ₄), Tᵢ))
        Turing.@addlogprob! sum(logpdf.(Normal.(μH  .+ λ₅ .* ηᵪ, θ₅), Hᵢ))
        Turing.@addlogprob! sum(logpdf.(Normal.(μC  .+ λ₆ .* ηᵪ, θ₆), Cᵢ))

        # Structural Priors 
        β₀ ~ Normal(2500, 1500)
        β₁ ~ truncated(Normal(0, 10), 0.0, Inf)
        β₂ ~ truncated(Normal(0, 0.01), -Inf, 0.0)
        γₙ ~ Normal(0, 50)
        γᵪ ~ Normal(0, 50)
        σ  ~ truncated(Normal(0.0, 500), 0.0, Inf)

        # Structural Equation 
        μᵢ = β₀ .+ β₁ .* (Ιᵢ .- Ī) .+ β₂ .* (Ιᵢ .- Ī).^2 .+ γₙ .* ηₙ .+ γᵪ .* ηᵪ

        # Vectorized Tobit Likehood (Hope is faster)
        log_lik_censored = logcdf.(Normal.(μᵢ, σ), 0.0)
        log_lik_observed = logpdf.(Normal.(μᵢ, σ), Y)
        Turing.@addlogprob! sum(ifelse.(Y .<= 0.0, log_lik_censored, log_lik_observed))
    end

    # Specify the model with data 
        model_sema = sema(
        sim_data[!, :Light],
        sim_data[!, :Nitrate],
        sim_data[!, :Iron],
        sim_data[!, :Phosphate],
        sim_data[!, :Temperature],
        sim_data[!, :pH],
        sim_data[!, :CO2],
        sim_data[!, :Population],
        Ī
    )

    # Sample
    chain = sample(model_sema, NUTS(0.65), MCMCSerial(), iters, 4)
    return chain
end

# SEM-B: Two-Factor Reflective CFA with Eilers–Peeters Light Structural Equation
export semb_tfr_cfa_ep_light

function semb_tfr_cfa_ep_light()


    
end 
end




