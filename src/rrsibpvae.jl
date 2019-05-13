mutable struct RRSIBPVAE <: AbstractIBPVAE
    prior
    N::Int
    k_max::Int
    x_dim::Int
    h_dim::Int
    encoder
    decoder
    rho
end

function Base.show(io::IO, vae::RRSIBPVAE)
    s = "RRSIBPVAE"
    s *= "\n  #params = $(numparams(vae))"
    print(io, s)
end

function (vae::RRSIBPVAE)(x, ::Val{true}; writer=nothing, epoch=nothing, iter=nothing, m::Int=10, τ=FT(0.1), kwargs...)
    # Roulette
    k_list = rand(value(vae.rho).lnpd, m)
    k_max = max(k_list...)
    if k_max > vae.k_max
        vae.k_max = k_max
    end

    # Compute weights for efficient multi-sample RRS
    rho = getρ(value(vae.rho), 1, k_max)
    one_minus_rho = 1 .- rho
    one_minus_rho_weighted = zeros(FT, k_max)
    # Multiple sample re-weighting
    for k in k_list
        one_minus_rho_weighted[1:k] .+= one_minus_rho[1:k]
    end
    one_minus_rho_weighted ./= m
    # NOTE: below takes use the independent property of KLs
    # .     cumsum due to the fact that R(i) includes all R(j<i)
    weight = AT(reverse(cumsum(reverse(one_minus_rho_weighted))))

    dist_nu, dist_gumbel, dist_A = vae.encoder(x, k_max)

    _, logpi = rand_logpi(dist_nu)
    logitpi = logit.(exp.(logpi))

    dist_Z = BatchGumbelBernoulliLogit(dist_gumbel.logitp .+ logitpi)
    logitZ = logitrand(dist_Z; τ=τ)

    L = zeros(FT, k_max)
    
    kl_A = if isdeep(vae)
        _kl_A = sum(kldiv(dist_A, BatchNormal(zero(FT), one(FT))); dims=2)
        L .+= vec(Array(value(_kl_A)))
        sum(_kl_A .* weight)
    else
        0
    end

    # Following Singh et al. (2017) use a multiplier of 1,000 for the KL term of nu to encourage adhering to the IBP prior.
    global_multiplier = size(x, 2) / vae.N * 1_000
    _kl_nu = sum(kldiv(dist_nu, BatchBeta(vae.prior.α, one(FT))); dims=2)
    _kl_nu = _kl_nu .* global_multiplier
    L .+= vec(Array(value(_kl_nu)))
    kl_nu = sum(_kl_nu .* weight)

    lq_Z = logpdflogit(dist_Z, logitZ; τ=τ)
    lp_Z = logpdflogit(BatchGumbelBernoulliLogit(logitpi), logitZ; τ=τ)
    _kl_Z = sum(lq_Z - lp_Z; dims=2)
    L .+= vec(Array(value(_kl_Z)))
    kl_Z = sum(_kl_Z .* weight)

    Z = sigm.(logitZ)
    code = isdeep(vae) ? rand(dist_A) .* Z : Z
    local lp_x
    for k = 1:k_max
        dist_x_k = vae.decoder(code[1:k,:])
        lp_x_k = sum(logpdf(dist_x_k, x))
        L[k] -= value(lp_x_k)
        if k == 1
            lp_x = lp_x_k * one_minus_rho_weighted[k]
        else
            lp_x += lp_x_k * one_minus_rho_weighted[k]
        end
    end
            
    L ./= size(x, 2)

    write_ibp_train_stats!(isdeep(vae), writer, iter, kl_A, kl_nu, kl_Z, lp_x, Z; k_list=k_list)
    
    # Update gradient for Rho
    dLdrho = MLToolkit.gradRR(i -> L[i], AutoGrad.value(vae.rho), k_list)
    updategrad!(AutoGrad.value(vae.rho), dLdrho)
            
    return (kl_nu + kl_Z + kl_A - lp_x) / size(x, 2)
end

function (vae::RRSIBPVAE)(x, ::Val{false}; m=10)
    k_list = rand(value(vae.rho).lnpd, 1_000)
    k = floor(Int, mean(k_list)) # mean
    return iwae_structured(vae, x, m, k)
end