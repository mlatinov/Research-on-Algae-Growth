module BGLM

using Turing

# BM-1: Bayesian Quadratic Regression (Light Only)
export bayes_quadratic_regreesion_light

function bayes_quadratic_regreesion_light(model_data, iters = 1000, sim = true)

    # Check if we are running real data or simulation
    if sim 
        # Get the simulation data from the DGP
        model_data = model_data[:sim_data]
    end 

    # Model Specification 
    @model function bqrl(I, Y)

        # Priors 
        β₀ ~ Normal(2500, 1500)
        β₁ ~ truncated(Normal(0, 10), 0, Inf)
        β₂ ~ truncated(Normal(0, 0.01), -Inf, 0)
        σ  ~ truncated(Normal(0, 500), 0, Inf) 

        # Conditional Mean 
        μᵢ = β₀ .+ β₁ .* I .+ β₂ .* I.^2
         
        # Tobit likelihood
        for i in 1:length(Y)
                
            if Y[i] == 0.0
            # Observation was censored: the true value was ≤ 0
                Turing.@addlogprob! logcdf(Normal(μᵢ[i], σ), 0.0)
            else
                # Observation is positive: standard Normal log-density
                Turing.@addlogprob! logpdf(Normal(μᵢ[i], σ), Y[i])
            end
        end
    end

    # Specify the model with the data 
    bqrl_model = bqrl(model_data.Light, model_data.Population)

    # Sampler NUTS
    chain = sample(bqrl_model, NUTS(0.65), MCMCSerial(), iters, 4)

    return chain

end

# BM-2: Bayesian Full Additive GLM (Quadratic Light + Six Linear Predictors)
export bayes_full_quadratic_regreesion

function bayes_full_quadratic_regreesion(model_data, iters = 1000, sim = false)

    # Check if we are running real data or simulation
    if sim 
        # Get the simulation data from the DGP
        model_data = model_data[:sim_data]
    end 

    # Model Specification 
    @model function bfqr(Ιᵢ,Nᵢ,Feᵢ,Pᵢ,Tᵢ,Hᵢ,Cᵢ,Y)
        # Priors 
        β₀ ~ Normal(2500.0, 1500.0)
        β₁ ~ truncated(Normal(0.0, 10.0), 0.0, Inf)
        β₂ ~ truncated(Normal(0, 0.01),-Inf, 0)
        β₃ ~ Normal(0.0, 50.0) # Nitrate effect
        β₄ ~ Normal(0.0, 50.0) # Iron effect
        β₅ ~ Normal(0.0, 50.0) # Phosphate effect
        β₆ ~ Normal(0.0, 50.0) # Temperature effect
        β₇ ~ Normal(0.0, 50.0) # pH effect
        β₈ ~ Normal(0.0, 50.0) # CO₂ effect
        σ ~ truncated(Normal(0, 500), 0.0, Inf)

        # Conditional mean 
        μᵢ = β₀ .+
            β₁ .* Ιᵢ .+ β₂ .* Ιᵢ.^2 .+
            β₃ .* Nᵢ .+ β₄ .* Feᵢ .+ β₅ .* Pᵢ .+
            β₆ .* Tᵢ .+ β₇ .* Hᵢ .+ β₈ .* Cᵢ

        # Tobit likelihood
        for i in 1:length(Y)
            if Y[i] == 0
                Turing.@addlogprob! logcdf(Normal(μᵢ[i], σ), 0.0)
            else 
                Turing.@addlogprob! logpdf(Normal(μᵢ[i], σ), Y[i])
            end
        end
    end

    # Specify the model with the data 
    bfqr_model = bfqr(
        model_data.Light,
        model_data.Nitrate,
        model_data.Iron,
        model_data.Phosphate,
        model_data.Temperature,
        model_data.pH,
        model_data.CO2,
        model_data.Population
    )

    # Sample 
    chain = sample(bfqr_model, NUTS(0.65), MCMCSerial(), iters, 4)
    return chain

end

# BM-3: Eilers Peeters PI Curve + Linear Secondary Predictors
export bepl_full

function bepl_full(data, iters = 1000)

    # Get the simulation data from the DGP
    sim_data = data[:sim_data]

    # Pre-compute the mean of Light outside the model 
    Ī = mean(sim_data[!, :Light])

    @model function bepl(Ιᵢ, Nᵢ, Feᵢ, Pᵢ, Tᵢ, Hᵢ, Cᵢ, Y, Ī)

        # Biological EP parameters 
        I_star ~ truncated(Normal(1000.0, 300.0), 100.0, 2500.0)
        α₀     ~ truncated(Normal(0.0, 0.05),     0.0,   Inf)

        # Recover raw EP parameters analytically 
        γₑ = 1.0 / α₀
        α  = γₑ / I_star^2
        βₑ = 1.0/Pm - 2.0 * sqrt(α * γₑ)

        # Guard unimodality: βₑ ≤ 0 means the curve has no finite peak.
        if βₑ <= 0.0
            Turing.@addlogprob! -Inf
            return
        end

        # Normalised PI curve 
        fPI_peak = I_star / (α * I_star^2 + βₑ * I_star + γₑ) 
        fPIn = fPI ./ fPI_peak    

        # Center the PI curve 
        fPI_mean  = Ī  / (α * Ī^2  + βₑ * Ī  + γₑ)
        fPIn_c    = fPIn .- (fPI_mean / Pm)
        
        # Structural priors 
        δ₀ ~ Normal(3000.0, 500.0)
        δ₁ ~ truncated(Normal(0.0, 2000.0), 0.0, Inf)
        δ₂ ~ Normal(0.0, 50.0)   # Nitrate
        δ₃ ~ Normal(0.0, 50.0)   # Iron
        δ₄ ~ Normal(0.0, 50.0)   # Phosphate
        δ₅ ~ Normal(0.0, 50.0)   # Temperature
        δ₆ ~ Normal(0.0, 50.0)   # pH
        δ₇ ~ Normal(0.0, 50.0)   # CO₂
        σ  ~ truncated(Normal(0.0, 500.0), 0.0, Inf)

        # Conditional mean 
        μᵢ = δ₀ .+ δ₁ .* fPIn_c .+
             δ₂ .* Nᵢ  .+ δ₃ .* Feᵢ .+ δ₄ .* Pᵢ .+
             δ₅ .* Tᵢ  .+ δ₆ .* Hᵢ  .+ δ₇ .* Cᵢ

        # Tobit likelihood
        log_lik_censored  = logcdf.(Normal.(μᵢ, σ), 0.0)
        log_lik_observed  = logpdf.(Normal.(μᵢ, σ), Y)
        Turing.@addlogprob! sum(ifelse.(Y .<= 0.0, log_lik_censored, log_lik_observed))
    end

    model_bepl = bepl(
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

    chain = sample(model_bepl, NUTS(0.65), MCMCSerial(), iters, 4)
    return chain
end


















end
