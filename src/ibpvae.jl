struct IBPEncoder <: StochasticLayer
    mlp
    ibp
    gauss
end

"""
Forward pass of the encoder that produces the variational distributions.
"""
function (enc::IBPEncoder)(x, d::Int...)
    h = enc.mlp(x)
    dist_kuma, dist_gumbel = enc.ibp(h, d...)
    dist_gauss = (enc.gauss == nothing) ? nothing : enc.gauss(h, d...)
    return dist_kuma, dist_gumbel, dist_gauss
end

function IBPLinearEncoder(k_max::Int, x_dim::Int, h_dim::Int, ibp_enc)
    mlp = Dense(x_dim, h_dim; f=relu)
    return IBPEncoder(mlp, ibp_enc, nothing)
end

function IBPDeepEncoder(k_max::Int, x_dim::Int, h_dim::Int, ibp_enc)
    mlp = Dense(x_dim, h_dim; f=relu)
    gauss = GaussianLogVarLazyDense(h_dim, k_max)
    return IBPEncoder(mlp, ibp_enc, gauss)
end

abstract type AbstractIBPVAE <: NeuralModel end

isdeep(vae::AbstractIBPVAE) = vae.encoder.gauss != nothing

"""
Sample pi in log space. This function also returns nu in log space as the FIRST parameter.
"""
function rand_logpi(dist_nu)
    lognu = logrand(dist_nu)
    logpi = break_logstick_ibp(lognu)
    return lognu, logpi
end

"""
Sample pi in logit space.
"""
function rand_logitpi(dist_nu; return_pi=false)
    logpi = rand_logpi(dist_nu)
    pi = exp.(logpi)
    logitpi = logit.(pi)
    
    return return_pi ? () : logitpi
end

"""
Make code.

For deep deocder, the code is the Hadamard product of A and Z.
For linear decoder, the code is just Z, as A is treated as the decoder parameter.
"""
function make_code(vae::AbstractIBPVAE, Z, dist_A)
    A = isdeep(vae) ? rand(dist_A) : nothing
    code = isdeep(vae) ? A .* Z : Z
    return code, A
end

"""
This function is used when computing ELBO.
"""
function compute_kl(vae::AbstractIBPVAE, dist_A, dist_nu, dist_Z, logpi, logitpi, logitZ; τ=FT(0.1))
    kl_A = isdeep(vae) ? sum(kldiv(dist_A, BatchNormal(zero(FT), one(FT)))) : 0
    kl_nu = sum(kldiv(dist_nu, BatchBeta(vae.prior.α, one(FT))))
    lq_Z = sum(logpdflogit(dist_Z, logitZ; τ=τ))
    lp_Z = sum(logpdflogit(BatchGumbelBernoulliLogit(logitpi), logitZ; τ=τ))
    kl_Z = lq_Z - lp_Z
    return kl_nu, kl_Z, kl_A
end


"""
This function is used when computing IWAE.
"""
function compute_logratio(vae::AbstractIBPVAE, dist_A, A, dist_nu, nu, pi, dist_Z, Z)
    logratioA = 0
    if isdeep(vae)
        lp_A = sum(logpdf(BatchNormal(zero(FT), one(FT)), A))
        lq_A = sum(logpdf(dist_A, A))
        logratioA = lp_A - lq_A
    end
    lp_nu = sum(logpdf(BatchBeta(vae.prior.α, one(FT)), nu))
    lq_nu = sum(logpdf(dist_nu, nu))
    lp_Z = sum(logpdf(BatchBernoulli(pi), Z)) 
    lq_Z = sum(logpdf(dist_Z, Z))
    return lp_nu - lq_nu, lp_Z - lq_Z, logratioA
end

"""
Linear Bernoulli decoder for IBP
"""
struct BernoulliIBPLinearDecoder <: StochasticLayer
    A
end

function BernoulliIBPLinearDecoder(k_max, x_dim)
    A = Knet.param(x_dim, k_max; atype=AT{FT,2})
    return BernoulliIBPLinearDecoder(A)
end

function (dec::BernoulliIBPLinearDecoder)(Z) 
    k_diff = size(Z, 1) - size(dec.A, 2)
    if k_diff > 0
        dec.A.value = hcat(dec.A.value, Knet.param(size(dec.A, 1), k_diff; atype=AT{FT,2}).value)
    end
    return BatchBernoulliLogit(dec.A[:,1:size(Z,1)] * Z)
end

"""
Linear Gaussian decoder for IBP
"""
struct GaussianIBPLinearDecoder <: StochasticLayer
    sig_X
    A
end

function GaussianIBPLinearDecoder(sig_X, k_max, x_dim)
    A = Knet.param(x_dim, k_max; atype=AT{FT,2})
    return GaussianIBPLinearDecoder(sig_X, A)
end

function (dec::GaussianIBPLinearDecoder)(Z)
    k_diff = size(Z, 1) - size(dec.A, 2)
    if k_diff > 0
        dec.A.value = hcat(dec.A.value, Knet.param(size(dec.A, 1), k_diff; atype=AT{FT,2}).value)
    end
    return BatchNormalLogVar(dec.A[:,1:size(Z,1)] * Z, log(dec.sig_X^2))
end

"""
Write IBP training stats to tbX.
"""
function write_ibp_train_stats!(is_deep, writer, iter, kl_A, kl_nu, kl_Z, lp_x, Z; k_list=nothing, plot_Z=false)
    if writer != nothing
        is_deep && writer."add_scalar"("batch/kl_A", value(kl_A), iter)
        writer."add_scalar"("batch/kl_nu", value(kl_nu), iter)
        writer."add_scalar"("batch/kl_Z", value(kl_Z), iter)
        writer."add_scalar"("batch/nll", -value(lp_x), iter)
        Z_bin = AT == Array ? Z .> 0.5 : Array(Z) .> 0.5
        num_acts = sum(Z_bin; dims=1)
        writer."add_histogram"("batch/num_acts", num_acts, iter)
        num_items = sum(Z_bin; dims=2)
        writer."add_histogram"("batch/num_items", num_items, iter)
        plot_Z && writer."add_image"("batch/Z", Z_bin[:,1:100], iter, dataformats="HW")
        k_list != nothing && writer."add_scalar"("batch/k_avg", mean(k_list), iter)
    end
end
