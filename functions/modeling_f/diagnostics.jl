module Diagnostics

using MCMCChains
using StatsPlots

# Convergent Diagnostics 
export conv_diagnostics

function conv_diagnostics(chain, parameters)

    # Subset the chain by parameters 
    chain_of_interest = chain[parameters]

    # Trace + density 
    trace = plot(chain_of_interest, size = (1200, 800))

    # R̂ and ESS  
    stats = summarystats(chain)

    # Autocorrelation 
    autocor_plot = plot(autocorplot(chain_of_interest), size = (1200, 800))

    # Corner plot joint posteriors 
    corner_plot = corner(chain_of_interest, size = (1400, 1400))

    return Dict(
        :trace    => trace,
        :autocor  => autocor_plot,
        :corner   => corner_plot,
        :stats    => stats        
    )
end






end