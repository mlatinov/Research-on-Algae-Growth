module BGLM

using Turing

# BM-1: Bayesian Quadratic Regression (Light Only)
export bayes_quadratic_regreesion_light

function bayes_quadratic_regreesion_light(data, iters = 1000)

    # Get the simulation data from the DGP
    sim_data = data[:sim_data]

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
    bqrl_model = bqrl(sim_data.Light, sim_data.Population)

    # Sampler NUTS
    chain = sample(bqrl_model, NUTS(0.65), MCMCSerial(), iters, 4)

    return chain

end

# BM-2: Bayesian Full Additive GLM (Quadratic Light + Six Linear Predictors)
export bayes_full_quadratic_regreesion

function bayes_full_quadratic_regreesion(data, iters = 1000)

    # Get the simulation data from the DGP
    sim_data = data[:sim_data]

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
        sim_data.Light,
        sim_data.Nitrate,
        sim_data.Iron,
        sim_data.Phosphate,
        sim_data.Temperature,
        sim_data.pH,
        sim_data.CO2,
        sim_data.Population
    )

    # Sample 
    chain = sample(bfqr_model, NUTS(0.65), MCMCSerial(), iters, 4)
    return chain

end






















end
