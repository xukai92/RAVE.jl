module RRVAE

using Reexport

@reexport using Knet, MLToolkit, Distributions

import AutoGrad

include("data.jl")
export get_data
include("plotting.jl")
export write_vae, make_Z_plot

function flatstr(args::Dict)
    TR_LIST = [:n_epochs, :batch_size, :lr, :beta1, :beta2]
    IGNORE_LIST = vcat([:continue, :store, :gpu_id, :check_every, :eval_every,
                        :bnp, :inf, :is_deep, :obs], TR_LIST)
    args_str_flat = flatten_dict(args; exclude=IGNORE_LIST, equal_sym="-")
    args_str_flat_tr = flatten_dict(args; include=TR_LIST, equal_sym="-")
    # Method
    args_str_flat_method = "$(args[:bnp])-$(args[:inf])-$(args[:is_deep] ? "deep" : "linear")-$(args[:obs])"
    return args_str_flat, args_str_flat_method, args_str_flat_tr
end

export flatstr

# IBP
include("ibpvae.jl")
export IBPDeepEncoder, IBPLinearEncoder, BernoulliIBPLinearDecoder, GaussianIBPLinearDecoder
include("mfibpvae.jl")
include("sibpvae.jl")
include("rrsibpvae.jl")

function IBPVAE(x_dim, args; sig_X=FT(0.1))
    N = args[:tr_sz]
    prior = IBP(FT(args[:alpha]))
    h_dim = args[:h_dim]
    k_max = args[:k_max]
    obs_type = Symbol(args[:obs])
    inf = Symbol(args[:inf])
    
    ibp_encoder = if inf == :mf
        MeanFieldSBC(h_dim, k_max)
    elseif inf == :s || inf == :rrs
        StructuredSBC(prior, h_dim, k_max)
    else
        throw("[IBPVAE] Unkown inference method: $(args[:inference])")
    end

    if args[:is_deep]
        encoder = IBPDeepEncoder(k_max, x_dim, h_dim, ibp_encoder)
        obs_layer = obs_type == :gauss ? GaussianLogVarDense(h_dim, x_dim) : BernoulliLogitDense(h_dim, x_dim)
        decoder = Chain(LazyDense(k_max, h_dim; f=relu), obs_layer)
    else
        encoder = IBPLinearEncoder(k_max, x_dim, h_dim, ibp_encoder)
        decoder = obs_type == :gauss ? GaussianIBPLinearDecoder(sig_X, k_max, x_dim) : BernoulliIBPLinearDecoder(k_max, x_dim)
    end

    return if inf == :mf
        MFIBPVAE(prior, N, k_max, x_dim, h_dim, encoder, decoder)
    elseif inf == :s
        SIBPVAE(prior, N, k_max, x_dim, h_dim, encoder, decoder)
    elseif inf == :rrs
        RRSIBPVAE(prior, N, k_max, x_dim, h_dim, encoder, decoder, Param(Rho(1)))
    end
end

export IBPVAE

# CRP
include("scrpvae.jl")
export CRPVAE, SCRPVAE, SCRPEncoder

end # module
