struct MFIBPVAE <: AbstractIBPVAE
    prior
    N::Int
    k_max::Int
    x_dim::Int
    h_dim::Int
    encoder
    decoder
end

function Base.show(io::IO, vae::MFIBPVAE)
    s = "MFIBPVAE"
    s *= "\n  #params = $(numparams(vae))"
    print(io, s)
end

function (vae::MFIBPVAE)(x, ::Val{true}; writer=nothing, iter=nothing, τ=FT(0.1), kwargs...)
    dist_nu, dist_Z, dist_A = vae.encoder(x, vae.k_max)
    
    _, logpi = rand_logpi(dist_nu)
    logitpi = logit.(exp.(logpi))
    
    logitZ = logitrand(dist_Z; τ=τ)

    kl_nu, kl_Z, kl_A = compute_kl(vae, dist_A, dist_nu, dist_Z, logpi, logitpi, logitZ; τ=τ)

    Z = sigm.(logitZ)
    code, _ = make_code(vae, Z, dist_A)
    dist_x = vae.decoder(code)
    lp_x = sum(logpdf(dist_x, x))

    write_ibp_train_stats!(isdeep(vae), writer, iter, kl_A, kl_nu, kl_Z, lp_x, Z)
    
    return (kl_nu + kl_Z + kl_A - lp_x) / size(x, 2)
end

function (vae::MFIBPVAE)(x, ::Val{false}; m=10, k_max=vae.k_max)
    dist_nu, dist_gumbel, dist_A = vae.encoder(x, k_max)
    dist_Z = BatchBernoulli(sigm.(dist_gumbel.logitp))

    local lw_list = []
    local x_mean_acc
    for i = 1:m
        lognu, logpi = rand_logpi(dist_nu)
        pi = exp.(logpi)

        Z = rand(dist_Z)

        code, A = make_code(vae, Z, dist_A)
        dist_x = vae.decoder(code)

        # IWAE
        lp_x = sum(logpdf(dist_x, x))
        
        nu = exp.(lognu)
        logrationu, logratioZ, logratioA = compute_logratio(vae, dist_A, A, dist_nu, nu, pi, dist_Z, Z)
        lw = lp_x + logrationu + logratioZ + logratioA
        push!(lw_list, lw)
        x_mean = mean(dist_x)
        if i == 1
            x_mean_acc = x_mean
        else
            x_mean_acc += x_mean
        end
    end
    lw = logsumexp(lw_list) - log(m)

    return lw / size(x, 2), x_mean_acc / m, dist_Z, dist_A
end
