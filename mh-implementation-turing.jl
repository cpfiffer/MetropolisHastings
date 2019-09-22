# Import the relevant libraries.
using Turing
using Turing.Interface
using Distributions
using Random

# Import specific functions and types to use or overload.
import Distributions: VariateForm, ValueSupport, variate_form, value_support, Sampleable
import MCMCChains: Chains
import Turing: Model
import Turing.Interface: step!, AbstractSampler, AbstractTransition, transition_type
import Turing.Inference: InferenceAlgorithm, AbstractSamplerState, Transition
import Turing.RandomVariables: VarInfo, runmodel!

# Define an InferenceAlgorithm type.
struct MetropolisHastings{T} <: InferenceAlgorithm end

# Tell the interface what transition type we would like to use. We can use the default
# Turing.Inference.Transition struct, and the Transition(spl) functions it 
# provides.
function transition_type(model::Model, spl::Sampler{<:MetropolisHastings})
    return typeof(Transition(spl))
end

# Define a function that makes a basic proposal.
function proposal(spl::MetropolisHastings, model::Model)
    return Transition(model, spl, rand(MvNormal(spl.state.vi[spl], 1)))
end

# Calculate the logpdf of one proposal given another proposal.
q(spl::MetropolisHastings, θ1::Real, θ2::Real) = logpdf(Normal(θ2, 1.0), θ2)
q(spl::MetropolisHastings, θ1::Vector{<:Real}, θ2::Vector{<:Real}) = logpdf(MvNormal(θ2, 1.0), θ2)
q(spl::MetropolisHastings, t1::Transition, t2::Transition) = q(spl, t1.θ, t2.θ)
function q(spl::MetropolisHastings, t1::NamedTuple{names1}, t2::NamedTuple{names2})
    @assert names1 == names2
    
end

# Define the first step! function, which is called at the 
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
    model::M,
    spl::S,
    ::Integer,
    θ_prev::T;
    kwargs...
) where {
    M <: DensityModel,
    S <: MetropolisHastings,
    T <: Transition
}
    # Generate a new proposal.
    θ = proposal(spl, model)
    
    # Calculate the log acceptance probability.
    α = θ.lp - θ_prev.lp + q(spl, θ_prev, θ) - q(spl, θ, θ_prev)

    # Decide whether to return the previous θ or the new one.
    if log(rand()) < min(α, 0.0)
        return θ
    else
        return θ_prev
    end
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