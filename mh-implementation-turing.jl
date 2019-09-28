# Import the relevant libraries.
using Turing
using Turing.Interface
using Distributions

# Import specific functions and types to use or overload.
import Distributions: VariateForm, ValueSupport, variate_form, value_support, Sampleable, insupport
import MCMCChains: Chains
import Turing: Model, Sampler, Selector
import Turing.Interface: step!, AbstractTransition, transition_type
import Turing.Inference: InferenceAlgorithm, AbstractSamplerState, Transition, parameters!, 
                         getspace, assume, parameter
import Turing.RandomVariables: VarInfo, VarName, variables

# Define an InferenceAlgorithm type.
struct MetropolisHastings{space} <: InferenceAlgorithm 
    proposal :: Function
end

# Default constructors.
MetropolisHastings(space=Tuple{}()) = MetropolisHastings{space}(x -> Normal(x, 1))
MetropolisHastings(f::Function) = MetropolisHastings{space}(f)

# These functions are required for your sampler to function with Turing,
# and they return the variables that a sampler has ownership of.
getspace(::MetropolisHastings{space}) where {space} = space
getspace(::Type{<:MetropolisHastings{space}}) where {space} = space

# Tell the interface what transition type we would like to use. We can use the default
# Turing.Inference.Transition struct, and the Transition(spl) functions it 
# provides.
function transition_type(model::Model, spl::Sampler{<:MetropolisHastings})
    return typeof(Transition(spl))
end

# Define a function that makes a basic proposal. This function runs the model and
# bundles the results up. In this case, the actual proposal occurs during
# the assume function.
function proposal(spl::Sampler{<:MetropolisHastings}, model::Model, t::Transition)
    return Transition(model, spl, parameters!(spl, t))
end

# Calculate the logpdf ratio of one proposal given another proposal.
function q(spl::Sampler{<:MetropolisHastings}, t1::Transition, t2::Transition)
    # Preallocate the ratio.
    ratio = 0.0

    # Iterate through each variable in the sampler.
    for vn in variables(spl)
        # Get the parameter from the Transition and the distribution 
        # associated with each variable.
        p1 = parameter(t1, vn)
        d1 = spl.alg.proposal(p1)

        p2 = parameter(t2, vn)
        d2 = spl.alg.proposal(p2)

        # Increment the log ratio.
        ratio += logpdf(d2, p1) - logpdf(d1, p2)
    end

    return ratio
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
    α = θ.lp - θ_prev.lp + q(spl, θ_prev, θ)

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
    # Retrieve the current parameter value.
    old_r = vi[vn]

    # Generate a proposal value.
    r = rand(spl.alg.proposal(vi[vn]))

    # Check if the proposal is in the distribution's support.
    if insupport(dist, r)
        # If the value is good, make sure to store it back in the VarInfo.
        vi[vn] = [r]
        return r, logpdf(dist, r)
    else
        # Otherwise return the previous value.
        return old_r, logpdf(dist, old_r)
    end
end

# Model declaration.
@model gdemo(xs) = begin
    σ ~ InverseGamma(2,3)
    μ ~ Normal(0, sqrt(σ))
    for i in 1:length(xs)
        xs[i] ~ Normal(μ, σ)
    end
end

# Generate a set of data from the posterior we want to estimate.
data = rand(Normal(5,3), 50)

# Construct a DensityModel.
model = gdemo(data)

# Set up our sampler.
spl = MetropolisHastings()

# Sample from the posterior.
chain = sample(model, spl, 100000)
