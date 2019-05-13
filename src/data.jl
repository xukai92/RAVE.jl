using FileIO

"""
    get_data(args; is_save=true)

Get data for the specified `dataset`.

For synthetic data, it will be randomly generated and 
if `is_save` is `true` it will be saved in the first time and for the next time it will be loaded.
"""
function get_data(args; is_save=true)
    tr_sz = args[:tr_sz]
    te_sz = args[:te_sz]
    dataset = args[:dataset]
    local features = nothing
    if dataset == "synth" || dataset == "synth_large" || dataset == "ibp_gauss_gen" || dataset == "sbc_gauss_gen"
        data_file = "../data/$(args[:dataset]).jld2"
        # Load saved generated data
        if ispath(data_file)
           return load(data_file, "x_tr", "x_te", "features")
        end
        local x
        n = tr_sz + te_sz
        gauss_dim = 256
        k_max = 25
        if dataset == "ibp_gauss_gen"
            x, features = gen_synth(args[:alpha], gauss_dim, n)
        elseif dataset == "sbc_gauss_gen"
            x, features = gen_synth(args[:alpha], gauss_dim, n, k_max)
        else
            if dataset == "synth"
                features = get_features_griffiths2011indian()
            elseif dataset == "synth_large"
                features = get_features_large()
            end
            x = mix_features(features, n; alpha=args[:alpha])
        end
        x_tr, x_te = Matrix{FT}(x[:,1:tr_sz]), Matrix{FT}(x[:,tr_sz+1:end])
        # Save data
        if is_save
            data_folder = "../data"
            if !ispath(data_folder)
                mkpath(data_folder)
            end
            save(data_file, Dict("x_tr" => x_tr, "x_te" => x_te, "features" => features))
            @info "Synthetic data ($(args[:dataset])) is saved to $data_file"
        end
    elseif dataset == "mnist"
        x_tr, _, x_te, _ = load_mnist(:mnist, tr_sz, te_sz)
    elseif dataset == "fmnist"
        x_tr, _, x_te, _ = load_mnist(:fmnist, tr_sz, te_sz)
    else throw("[main] Unkown dataset: $dataset") end
    
    feature_size = features == nothing ? "nothing" : size(features)
    @info "Dataset size" size(x_tr) size(x_te) feature_size
    return x_tr, x_te, features
end
                
function gen_synth(alpha, D, N, K...)
    Z = rand(IBP(alpha), N, K...)
    K = size(Z, 2)
    A = randn(D, K)
    X = A * Z'
    return X, A
end

"""
    mix_features(features, N; alpha=nothing, clip01=true, noise_level=0.1)

Generate data by randomly mixing features. 
                
By default it uses each feature uniformly, but if `alpha` is provided, it generates the activations following a corresponding stick-breaking construction of IBP.
`features` is assumed to be in a `D` by `K` matrix, where `K` is the number of features and `D` is the dimension of each feature.
"""
function mix_features(features, N; alpha=nothing, clip01=true, noise_level=0.1)
    D, K = size(features)

    # Get activation matrix
    Z = if alpha != nothing
        # Mix via stick-breaking construction
        rand(IBP(alpha), N, K)
    else
        # Mix by uniformly activation
        z_list = []
        for n = 1:N
            z_raw = Vector{Int}(map(x->parse(Int, string(x)), collect(Base.bin(unsigned(rand(0:2^K-1)), 4, false))))
            z = zeros(Int, K)
            j = K
            for i = length(z_raw):-1:1
                z[j] = z_raw[i]
                j = j - 1
            end
            push!(z_list, z)
        end
        hcat(z_list...)'
    end
    @info "Maximum number of features" sum(sum(Z; dims=1) .> 0)
    
    # Mixing feature and activations
    X = features * Z'

    # Add Gaussian noise
    X = X + randn(D, N) * noise_level
                    
    # Clip to be between 0 and 1
    if clip01
        X[X .> 1.0] .= 1.0
        X[X .< 0.0] .= 0.0
    end
                    
    return X
end
