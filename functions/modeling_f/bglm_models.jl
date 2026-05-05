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

function bayes_full_quadratic_regreesion(model_data, iters = 1000, sim = true)

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
    chain = sample(bfqr_model, NUTS(0.65), MCMCSerial(), iters, 4, progress = true)
    return chain

end

# BM-3: Eilers Peeters PI Curve + Linear Secondary Predictors
export bepl_full

function bepl_full(model_data, iters = 1000, sim = true)

    # Check if we are running real data or simulation
    if sim 
        # Get the simulation data from the DGP
        model_data = model_data[:sim_data]
    end 

    # Pre-compute the mean of Light outside the model 
    Ī = mean(model_data[!, :Light])

    @model function bepl(Ιᵢ, Nᵢ, Feᵢ, Pᵢ, Tᵢ, Hᵢ, Cᵢ, Y, Ī)

        # Biological EP parameters 
        log_I_star ~ Normal(log(1000.0), 0.3)   # log(1000) ≈ 6.9
        log_α₀     ~ Normal(log(0.05), 0.5)
        log_Pm     ~ Normal(log(0.1),    1.0)

        # Back-transform
        I_star = exp(log_I_star)
        α₀     = exp(log_α₀)
        Pm     = exp(log_Pm)

        # Recover raw EP parameters analytically 
        γₑ = 1.0 / α₀
        α  = γₑ / I_star^2
        βₑ = 1.0 / Pm - 2.0 * sqrt(α * γₑ)

        # Soft constraint only — no return, no hard wall
        Turing.@addlogprob! -exp(-10.0 * βₑ)

        # Standardise PI curve 
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

        # Structural priors 
        δ₀ ~ Normal(0.0, 500.0)  
        δ₁ ~ truncated(Normal(0.0, 1000.0), 0.0, Inf)
        δ₂ ~ Normal(0.0, 50.0)   # Nitrate
        δ₃ ~ Normal(0.0, 50.0)   # Iron
        δ₄ ~ Normal(0.0, 50.0)   # Phosphate
        δ₅ ~ Normal(0.0, 50.0)   # Temperature
        δ₆ ~ Normal(0.0, 50.0)   # pH
        δ₇ ~ Normal(0.0, 50.0)   # CO₂
        σ  ~ truncated(Normal(0.0, 500.0), 0.0, Inf)

        # Conditional mean 
        μᵢ = δ₀ .+ δ₁ .* fPIn_s .+
             δ₂ .* Nᵢ_s  .+ δ₃ .* Feᵢ_s .+ δ₄ .* Pᵢ_s .+
             δ₅ .* Tᵢ_s  .+ δ₆ .* Hᵢ_s  .+ δ₇ .* Cᵢ_s

        # Tobit likelihood
        log_lik_censored  = logcdf.(Normal.(μᵢ, σ), 0.0)
        log_lik_observed  = logpdf.(Normal.(μᵢ, σ), Y)
        Turing.@addlogprob! sum(ifelse.(Y .<= 0.0, log_lik_censored, log_lik_observed))

        return (I_star = I_star, α₀ = α₀, Pm = Pm)
    end

    model_bepl = bepl(
        model_data[!, :Light],
        model_data[!, :Nitrate],
        model_data[!, :Iron],
        model_data[!, :Phosphate],
        model_data[!, :Temperature],
        model_data[!, :pH],
        model_data[!, :CO2],
        model_data[!, :Population],
        Ī
    )

    chain = sample(model_bepl, NUTS(0.9), MCMCSerial(), iters, 4)
    return chain
end

# Fuck this nonsense if this doesnt work i give up on this 
export bepl_full_fixed

function bepl_full_fixed(model_data, iters)

    # Model specification with fixed PI paramters 
    @model function bepl_fixed_ep(Ιᵢ, Nᵢ, Feᵢ, Pᵢ, Tᵢ, Hᵢ, Cᵢ, Y, I_star_true, α₀_true, Pm_true)

    # EP parameters fixed 
    I_star = I_star_true
    α₀     = α₀_true
    Pm     = Pm_true

    γₑ = 1.0 / α₀
    α  = γₑ / I_star^2
    βₑ = 1.0 / Pm - 2.0 * sqrt(α * γₑ)

    fPI      = Ιᵢ ./ (α .* Ιᵢ.^2 .+ βₑ .* Ιᵢ .+ γₑ)
    fPI_peak = I_star / (α * I_star^2 + βₑ * I_star + γₑ)
    fPIn     = fPI ./ fPI_peak

    δ₀ ~ Normal(0.0,    500.0)
    δ₁ ~ truncated(Normal(0.0, 1000.0), 0.0, Inf)
    δ₂ ~ Normal(0.0, 50.0)
    δ₃ ~ Normal(0.0, 50.0)
    δ₄ ~ Normal(0.0, 50.0)
    δ₅ ~ Normal(0.0, 50.0)
    δ₆ ~ Normal(0.0, 50.0)
    δ₇ ~ Normal(0.0, 50.0)
    σ  ~ truncated(Normal(0.0, 500.0), 0.0, Inf)

    μᵢ = δ₀ .+ δ₁ .* fPIn .+
         δ₂ .* Nᵢ .+ δ₃ .* Feᵢ .+ δ₄ .* Pᵢ .+
         δ₅ .* Tᵢ .+ δ₆ .* Hᵢ  .+ δ₇ .* Cᵢ

    log_lik_censored = logcdf.(Normal.(μᵢ, σ), 0.0)
    log_lik_observed = logpdf.(Normal.(μᵢ, σ), Y)
    Turing.@addlogprob! sum(ifelse.(Y .<= 0.0, log_lik_censored, log_lik_observed))
    end

    gt = model_data[:ground_truth]
    data = model_data[:sim_data]
    model_fixed = bepl_fixed_ep(
        data[!, :Light],
        data[!, :Nitrate],
        data[!, :Iron],
        data[!, :Phosphate],
        data[!, :Temperature],
        data[!, :pH],
        data[!, :CO2],
        data[!, :Population],
        gt[:I_star],    # ← true I_star the DGP used
        gt[:α₀],        # ← true α₀ the DGP used
        gt[:Pm]         # ← true Pm the DGP used
    )
    chain_fixed = sample(model_fixed, NUTS(0.9), MCMCSerial(), iters, 4)
    return chain_fixed
end





















end
