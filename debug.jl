using Turing

@model gdemo(x) = begin
    μ ~ Normal()
    σ ~ InverseGamma(2,3)
    x ~ Normal(μ, σ)
end

sample(gdemo(1.0), MH(), 1000)