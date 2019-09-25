# Import the relevant libraries.
using Turing
using Turing.Interface
using Distributions
using Random

# Import specific functions and types to use or overload.
import Distributions: VariateForm, ValueSupport, variate_form, value_support, Sampleable, insupport
import MCMCChains: Chains
import Turing: Model, Sampler
import Turing.Interface: step!, AbstractTransition, transition_type
import Turing.Inference: InferenceAlgorithm, AbstractSamplerState, Transition, parameters!, 
                         getspace, assume, observe
import Turing.RandomVariables: VarInfo, runmodel!, VarName, link!, invlink!

# Define an InferenceAlgorithm type.
struct MetropolisHastings{space} <: InferenceAlgorithm end
MetropolisHastings(space=Tuple{}()) = MetropolisHastings{space}()
getspace(::MetropolisHastings{space}) where {space} = space
getspace(::Type{<:MetropolisHastings{space}}) where {space} = space

# Tell the interface what transition type we would like to use. We can use the default
# Turing.Inference.Transition struct, and the Transition(spl) functions it 
# provides.
function transition_type(model::Model, spl::Sampler{<:MetropolisHastings})
    return typeof(Transition(spl))
end

# Define a function that makes a basic proposal.
function proposal(spl::Sampler{<:MetropolisHastings}, model::Model, t::Transition)
    d = MvNormal(parameters!(spl, t), 1)
    return Transition(model, spl, rand(d))
end

# Calculate the logpdf of one proposal given another proposal.
q(spl::Sampler{<:MetropolisHastings}, θ1::Real, θ2::Real) = logpdf(Normal(θ2, 1.0), θ2)
q(spl::Sampler{<:MetropolisHastings}, θ1::Vector{<:Real}, θ2::Vector{<:Real}) = logpdf(MvNormal(θ2, 1.0), θ2)
function q(spl::Sampler{<:MetropolisHastings}, t1::Transition, t2::Transition)
    return q(spl, parameters!(spl, t1), parameters!(spl, t2))
end

# Define the first step! function, which is called the 
# beginning of sampling. Return the initial parameter used
# to define the sampler.
function step!(
    rng::AbstractRNG,
    model::Model,
    spl::Sampler{<:MetropolisHastings},
    N::Integer;
    kwargs...
)
    return Transition(spl)
end

# Define the other step functions. Returns a Transition containing
# either a new proposal (if accepted) or the previous proposal 
# (if not accepted).
function step!(
    rng::AbstractRNG,
    model::Model,
    spl::Sampler{<:MetropolisHastings},
    ::Integer,
    θ_prev::T;
    kwargs...
) where {
    T <: Transition
}
    # Generate a new proposal.
    θ = proposal(spl, model, θ_prev)
    
    # Calculate the log acceptance probability.
    α = θ.lp - θ_prev.lp + q(spl, θ_prev, θ) - q(spl, θ, θ_prev)

    # Decide whether to return the previous θ or the new one.
    if log(rand()) < min(α, 0.0)
        return θ
    else
        return θ_prev
    end
end

function assume(
    spl::Sampler{<:MetropolisHastings},
    dist::Distribution,
    vn::VarName,
    vi::VarInfo
)
    # r = vi[vn] 
    # if insupport(dist, r)
    #     return r, logpdf(dist, r)
    # else
    #     return r, -Inf
    # end

    r = rand(dist)
    vi[vn] = [r]
    return r, logpdf(dist, r)

    # p = rand(Normal(r))
    # if insupport(dist, p)
    #     # println(r)
    #     # println(p)
    #     vi[vn] = [p]
    #     return p, logpdf(dist, p)
    # else
    #     return r, logpdf(dist, r)
    # end
end

# function assume(
#     spl::Sampler{<:MetropolisHastings},
#     dist::Vector{D},
#     vn::VarName,
#     vi::VarInfo
# ) where {D<:Distribution}
#     r = vi[vn]
#     p = rand(Normal(r))
#     if insupport(dist, r)
#         return r, sum(logpdf.(dist, r))
#     else
#         return r, -Inf
#     end
# end

function observe(spl::Sampler{<:MetropolisHastings}, d::Distribution, value::Any, vi::VarInfo)
    return observe(nothing, d, value, vi)  # accumulate pdf of likelihood
end

function observe( spl::Sampler{<:MetropolisHastings},
                  ds::Vector{D},
                  value::Any,
                  vi::VarInfo
                )  where D<:Distribution
    return observe(nothing, ds, value, vi) # accumulate pdf of likelihood
end

# Model declaration.
@model gdemo(xs) = begin
    μ ~ Normal(0, 1)
    σ ~ InverseGamma(2,3)
    for i in 1:length(xs)
        xs[i] ~ Normal(μ, σ)
    end
end

# Generate a set of data from the posterior we want to estimate.
data = rand(Normal(5,3), 10)

# Construct a DensityModel.
model = gdemo(data)

# Set up our sampler.
spl = MetropolisHastings()

# Sample from the posterior.
chain = sample(model, spl, 10000)