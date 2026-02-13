
function circGauss(N::Integer,μ::Vector{Float64},σ²::Float64)
    
    dimension = length(μ)

    #  Input Validation and Parameter Calculation
    if σ² < 0
        error("Variance (sigma^2) must be non-negative.")
    end

    # The standard deviation (sigma) is the scaling factor for N(0, 1) samples.
    std_dev = sqrt(σ²)

    # Generate Zero-Mean Samples (Standard Normal N(0, 1))
    W = randn(dimension, N)
    
    # Scale and Shift
    X_zero_mean = std_dev .* W
    X_matrix = X_zero_mean .+ μ

    X = Vector{Vector{Float64}}(undef, N)
    
    for n in 1:N
        # X_matrix[:, n] extracts the n-th sample vector (column)
        X[n] = X_matrix[:, n] 
    end

    return X::Vector{Vector{Float64}} 
end

