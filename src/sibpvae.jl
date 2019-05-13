struct SIBPVAE <: AbstractIBPVAE
    prior
    N::Int
    k_max::Int
    x_dim::Int
    h_dim::Int
    encoder
    decoder
end

function Base.show(io::IO, vae::SIBPVAE)
    s = "SIBPVAE"
    s *= "\n  #params = $(numparams(vae))"
    print(io, s)
end

function (vae::SIBPVAE)(x, ::Val{true}; writer=nothing, iter=nothing, τ=FT(0.1), kwargs...)
    dist_nu, dist_gumbel, dist_A = vae.encoder(x, vae.k_max)

    _, logpi = rand_logpi(dist_nu)
    logitpi = logit.(exp.(logpi))

    dist_Z = BatchGumbelBernoulliLogit(dist_gumbel.logitp .+ logitpi)
    logitZ = logitrand(dist_Z; τ=τ)

    kl_nu, kl_Z, kl_A = compute_kl(vae, dist_A, dist_nu, dist_Z, logpi, logitpi, logitZ; τ=τ)
    # Following Singh et al. (2017) use a multiplier of 1,000 for the KL term of nu to encourage adhering to the IBP prior.
    global_multiplier = size(x, 2) / vae.N * 1_000
    kl_nu = kl_nu * global_multiplier

    Z = sigm.(logitZ)
    code, _ = make_code(vae, Z, dist_A)
    dist_x = vae.decoder(code)
    lp_x = sum(logpdf(dist_x, x))

    write_ibp_train_stats!(isdeep(vae), writer, iter, kl_A, kl_nu, kl_Z, lp_x, Z)

    return (kl_nu + kl_A + kl_Z - lp_x) / size(x, 2)
end

function iwae_structured(vae, x, m::Int, k_max::Int)
    dist_nu, dist_gumbel, dist_A = vae.encoder(x, k_max)

    local lw_list = []
    local x_mean_acc, Z_mean_acc
    for i = 1:m
        lognu, logpi = rand_logpi(dist_nu)
        pi = exp.(logpi); logitpi = logit.(pi)

        Z_mean = sigm.(dist_gumbel.logitp .+ logitpi)
        dist_Z = BatchBernoulli(Z_mean)
        Z = rand(dist_Z)

        code, A = make_code(vae, Z, dist_A)
        dist_x = vae.decoder(code)

        # IWAE
        lp_x = sum(logpdf(dist_x, x))
        
        nu = exp.(lognu)
        logrationu, logratioZ, logratioA = compute_logratio(vae, dist_A, A, dist_nu, nu, pi, dist_Z, Z)
        global_multiplier = size(x, 2) / vae.N
        lw = lp_x + logrationu * global_multiplier + logratioZ + logratioA
        push!(lw_list, lw)
        x_mean = mean(dist_x)
        if i == 1
            x_mean_acc = x_mean; Z_mean_acc = Z_mean
        else
            x_mean_acc += x_mean; Z_mean_acc += Z_mean
        end
    end
    lw = logsumexp(lw_list) - log(m)

    return lw / size(x, 2), x_mean_acc / m, BatchBernoulli(Z_mean_acc / m), dist_A
end

(vae::SIBPVAE)(x, ::Val{false}; m=10, k_max=vae.k_max) = iwae_structured(vae, x, m, k_max)
