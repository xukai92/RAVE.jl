function write_vae(writer, vae, x, epoch; n_cols=7, n_rows=7)
    show_size = n_cols * n_rows

    p = plt.figure(figsize=(12,6))

    plt.subplot(1, 3, 1)
    ax = plot_grayimg(Array(x), n_cols, n_rows)
    ax."set_title"("data")

    x_renc_mean = vae(x, Val(false))[2]

    plt.subplot(1, 3, 2)
    ax = plot_grayimg(Array(x_renc_mean), n_cols, n_rows)
    ax."set_title"("reconstructions")

    Z = AT{FT}(rand(vae.prior, size(x, 2), vae.k_max)')
    code = isdeep(vae) ? AT(randn(FT, size(Z)...)) .* Z : Z
    dist_x = vae.decoder(code)
    x_gen_mean = mean(dist_x)

    plt.subplot(1, 3, 3)
    ax = plot_grayimg(Array(x_gen_mean), n_cols, n_rows)
    ax."set_title"("samples from prior")

    writer."add_figure"("d-r-s", p, epoch)

    if !isdeep(vae)
        p = plt.figure()
        ax = plot_grayimg(AT == Array ? vae.decoder.A : Array(vae.decoder.A))
        ax."set_title"("features")
        writer."add_figure"("features", p, epoch)
    end
end

"""
    make_Z_plot(Z::Matrix; n_show=100, subplotting=true)

Viusalise activations Z, which is assumed to be a N x K matrix.
When `subplotting` is `true`, it returns a single plot with 3 subplots.
Otherwise it returns a tuple of 3 individual plots.
"""
function make_Z_plot(Z::Matrix; n_show=50, k_mode_rr=nothing, subplotting=true)
    N, K = size(Z)

    p = plt.figure(figsize=(12,9))

    if subplotting
        plt.subplot(1, 2, 1)
        plot_actmat(Z[1:n_show,:])
    else
        p1 = plt.figure(figsize=(3,2))
        plot_actmat(Z[1:size(Z,2),:])
    end

    subplotting && plt.title("Feature activations")
    plt.xlabel("k-th feature")
    plt.ylabel("n-th customer")

    if subplotting
        plt.subplot(2, 2, 2)
    else
        p2 = plt.figure(figsize=(3,2))
    end

    act_per_data_num = vec(sum(Z; dims=2))

    plt.hist(act_per_data_num)
    plt.xlim([0, max(act_per_data_num...)])
    subplotting && plt.title("#activation per data")
    plt.xlabel("#feature")
    plt.ylabel("Frequency")

    if subplotting
        plt.subplot(2, 2, 4)
    else
        p3 = plt.figure(figsize=(3,2))
    end

    act_per_feature_sum = vec(sum(Z; dims=1))
    plt.bar(1:K, sort(act_per_feature_sum; rev=true))
    subplotting && plt.title("#activation per feature")
    !(k_mode_rr === nothing) && plt.axvline(x=k_mode_rr, color="r", linestyle="--")
    plt.xlabel("k-th feature")
    plt.ylabel("Fraction")

    if subplotting
        return p
    else
        return p1, p2, p3
    end
end
