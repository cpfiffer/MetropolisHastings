# Import the relevant libraries.
using Turing.Interface
using Distributions
using Random

# Import specific functions and types to use or overload.
import Distributions: VariateForm, ValueSupport, variate_form, value_support, Sampleable
import MCMCChains: Chains
import Turing.Interface: step!, AbstractSampler, AbstractTransition, transition_type

# Define a sampler type.
struct MetropolisHastings{T} <: AbstractSampler 
    init_θ :: T
end

# Define a model type. Stores the log density function and the data to 
# evaluate the log density on.
struct DensityModel{V<:VariateForm, S<:ValueSupport, T} <: Sampleable{V, S}
    π :: Function
    data :: T
end

# Default density constructor.
DensityModel(π::Function, data::T) where T = DensityModel{VariateForm, ValueSupport, T}(π, data)

# Create a very basic Transition type, only stores the 
# parameter draws and the log probability of the draw.
struct Transition{T} <: AbstractTransition
    θ :: T
    lp :: Float64
end

# Store the new draw and its log density.
Transition(model::DensityModel, θ::T) where T = Transition(θ, ℓπ(model, θ))

# Tell the interface what transition type we would like to use.
transition_type(model::DensityModel, spl::MetropolisHastings) = typeof(Transition(spl.init_θ, ℓπ(model, spl.init_θ)))

# Define a function that makes a basic proposal depending on a univariate
# parameterization or a multivariate parameterization.
proposal(spl::MetropolisHastings, model::DensityModel, θ::Real) = Transition(model, rand(Normal(θ, 1)))
proposal(spl::MetropolisHastings, model::DensityModel, θ::Vector{<:Real}) = Transition(model, rand(MvNormal(θ, 1)))
proposal(spl::MetropolisHastings, model::DensityModel, t::Transition) = proposal(spl, model, t.θ)

# Calculate the logpdf of one proposal given another proposal.
q(spl::MetropolisHastings, θ1::Real, θ2::Real) = logpdf(Normal(θ1, 1.0), θ2)
q(spl::MetropolisHastings, θ1::Vector{<:Real}, θ2::Vector{<:Real}) = logpdf(MvNormal(θ1, 1.0), θ2)
q(spl::MetropolisHastings, t1::Transition, t2::Transition) = q(spl, t1.θ, t2.θ)

# Calculate the density of the model given some parameterization.
ℓπ(model::DensityModel, θ::T) where T = model.π(model.data, θ)
ℓπ(model::DensityModel, t::Transition) = t.lp

# Define the first step! function, which is called at the 
# beginning of sampling. Return the initial parameter used
# to define the sampler.
function step!(
    rng::AbstractRNG,
    model::M,
    spl::S,
    N::Integer;
    kwargs...
) where {
    M <: DensityModel,
    S <: MetropolisHastings
}
    return Transition(model, spl.init_θ)
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
    θ = proposal(spl, model, θ_prev)
    
    # Calculate the log acceptance probability.
    α = ℓπ(model, θ) - ℓπ(model, θ_prev) + q(spl, θ_prev, θ) - q(spl, θ, θ_prev)

    # Decide whether to return the previous θ or the new one.
    if log(rand()) < min(α, 0.0)
        return θ
    else
        return θ_prev
    end
end

# A basic chains constructor that works with the Transition struct we defined.
function Chains(
    rng::AbstractRNG, 
    ℓ::DensityModel, 
    s::MetropolisHastings, 
    N::Integer, 
    ts::Vector{T}; 
    param_names=missing,
    kwargs...
) where {T <: Transition}
    # Turn all the transitions into a vector-of-vectors.
    vals = [vcat(t.θ, t.lp) for t in ts]

    # Check if we received any parameter names.
    if ismissing(param_names)
        param_names = ["Parameter $i" for i in 1:(length(first(vals))-1)]
    end

    # Add the log density field to the parameter names.
    push!(param_names, "lp")

    # Bundle everything up and return a Chains struct.
    return Chains(vals, param_names, (internals=["lp"],))
end

# Define the components of a basic model.
insupport(θ) = θ[2] >= 0
dist(θ) = Normal(θ[1], sqrt(θ[2]))
density(data, θ) = insupport(θ) ? sum(logpdf.(dist(θ), data)) : -Inf

# Generate a set of data from the posterior we want to estimate.
data = rand(Normal(5,3), 10)

# Construct a DensityModel.
model = DensityModel(density, data)

# Set up our sampler.
spl = MetropolisHastings([0.0, 0.0])

# Sample from the posterior.
chain = sample(model, spl, 10000; param_names=["μ", "σ"])