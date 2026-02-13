using Plots
using LinearAlgebra
using Statistics
using Random
using Images

function myKmeansBatch(𝐱ₙ::Vector{Vector{Float64}}, 𝛍ₖ::Vector{Vector{Float64}}; maxIter=50, verbose=true)
    N = length(𝐱ₙ) # Number of data points
    K = length(𝛍ₖ) # Number of clusters (determined by the input 𝛍ₖ size)
    D = length(𝐱ₙ[1]) # Dimension of data points

    # --- INITIALIZATION: Mutate input 𝛍ₖ to be a random subset of K data points ---
    if K > N
        error("K cannot be greater than the number of data points N.")
    end
    
    # Select K unique random indices from the data set 
    initial_indices = Random.shuffle(1:N)[1:K]
    
    # Mutate the input 𝛍ₖ vector in place to hold the initial centers
    for k in 1:K
        # Use deepcopy to ensure 𝛍ₖ[k] is not just a reference to 𝐱ₙ[initial_indices[k]]
        𝛍ₖ[k] = deepcopy(𝐱ₙ[initial_indices[k]])
    end

    iteration = 0
    # Initialize assignment matrix 'r' (will be updated and returned)
    r = fill(false, N, K) 

    if verbose
        println("\nStarting K-means with K=$K...")
    end

    # --- ITERATIVE LOOP (Algorithm 3, Line 2: while not converged) ---
    for outer iteration ∈ 1:maxIter
        if verbose
            println("Iteration: $(iteration)")
        end

        # 1. E-Step (Assignment: Line 3 of Algorithm 3)
        # Get the new assignments based on current centers
        _, r_new = compDistInd(𝐱ₙ, 𝛍ₖ)

        # Store old centers for convergence check
        μ_old = deepcopy(𝛍ₖ)

        # 2. M-Step (Update: Line 4 of Algorithm 3)
        μ_updated = false 
        
        # Iterate over each cluster center k
        for k in 1:K
            assigned_indices = findall(r_new[:, k])
            count_nk = length(assigned_indices) 
            
            if count_nk == 0
                # Stability Fix: Reinitialize the centroid (μk) to a new random data point
                if verbose
                    println("  -- Stability Warning: Cluster $k is empty. Reinitializing its location.")
                end
                
                new_idx = rand(1:N) 
                # Mutate the input 𝛍ₖ in place for the stability fix
                𝛍ₖ[k] = deepcopy(𝐱ₙ[new_idx]) 
                μ_updated = true
            else
                # Centroid Update: μk = (sum rnk * xn) / (sum rnk)
                sum_xn = sum(𝐱ₙ[i] for i in assigned_indices)
                μ_new_k = sum_xn ./ count_nk 
                
                # Check if the new center is different from the old center
                if norm(μ_new_k - 𝛍ₖ[k]) > 1e-9
                    # Mutate the input 𝛍ₖ in place
                    𝛍ₖ[k] = μ_new_k
                    μ_updated = true
                end
            end
        end

        # --- CONVERGENCE CHECK (Line 2: while not converged) ---
        if !μ_updated
            if verbose
                println("\nK-means converged after $(iteration) iterations.")
            end
            break
        end

    end # end while loop

    # Final assignment based on the last set of converged centers (to return 'r')
    _, r = compDistInd(𝐱ₙ, 𝛍ₖ)
    
    # Explicit return with required type annotation
    return 𝛍ₖ::Vector{Vector{Float64}}, r::Matrix{Bool}, iteration::Int64
end
