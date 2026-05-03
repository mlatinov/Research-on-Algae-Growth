module Diagnostics

using MCMCChains
using StatsPlots

# Convergent Diagnostics 
export conv_diagnostics

function conv_diagnostics(chain)

    # Trace + density 
    trace = plot(chain, size = (1200, 800))

    # R̂ and ESS  
    stats = summarystats(chain)

    # Autocorrelation 
    autocor_plot = plot(autocorplot(chain), size = (1200, 800))

    # Corner plot joint posteriors 
    corner_plot = corner(chain, size = (1400, 1400))

    return Dict(
        :trace    => trace,
        :autocor  => autocor_plot,
        :corner   => corner_plot,
        :stats    => stats        
    )
end






end