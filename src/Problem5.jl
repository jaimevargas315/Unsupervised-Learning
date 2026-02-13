using LinearAlgebra, Random, Statistics

function myKmeansOnline(𝐱ₙ::Vector{Vector{Float64}}, 𝛍ₖ::Vector{Vector{Float64}}, η::Float64; 
                        maxIter::Int=50, tol::Float64=1e-5, verbose::Bool=true)
    
    N = length(𝐱ₙ)
    K = length(𝛍ₖ)
    
    if N == 0 || K <= 0 || K > N
        error("Invalid input: N must be > 0, and 0 < K <= N.")
    end

    # 1. INITIALIZATION: Overwrite input 𝛍ₖ with K random unique data points (REQUIRED)
    initial_indices = Random.shuffle(1:N)[1:K]
    resize!(𝛍ₖ, K) # Ensure 𝛍ₖ has exactly K slots (though it should based on length check)
    for k in 1:K
        # Deepcopy the data vector into the centroid container
        𝛍ₖ[k] = deepcopy(𝐱ₙ[initial_indices[k]])
    end

    # Store centroids before the epoch begins for movement calculation
    𝛍ₖ_before_epoch = deepcopy(𝛍ₖ)
    
    epoch = 0
    
    for outer epoch ∈ 1:maxIter
        
        # 2. DATASET ORDER RANDOMIZATION: Shuffle data order before each epoch
        shuffled_indices = Random.shuffle(1:N)
        
        total_centroid_movement = 0.0
        
        # Store centers before this epoch for the convergence check
        𝛍ₖ_before_epoch_start = deepcopy(𝛍ₖ)

        # 3. ONLINE UPDATE LOOP (Iterate through all data points in random order)
        for n_idx in shuffled_indices
            x_n = 𝐱ₙ[n_idx]
            
            # Find the index I of the closest centroid (E-step)
            min_dist_sq = Inf
            I = -1
            
            for k in 1:K
                # Calculate squared Euclidean distance: ||xn - μk||²
                dist_sq = norm(x_n - 𝛍ₖ[k])^2
                if dist_sq < min_dist_sq
                    min_dist_sq = dist_sq
                    I = k
                end
            end
            
            # Update the closest centroid (Online M-step / SGD-like update)
            # 𝛍_I_new = 𝛍_I_old + η * (x_n - 𝛍_I_old)
            error_vector = x_n - 𝛍ₖ[I]
            𝛍ₖ[I] += η * error_vector # This performs the update in place
        end
        
        # 4. CONVERGENCE CHECK (after one full epoch)
        # Calculate the total distance moved by all centroids since the start of the epoch
        for k in 1:K
            movement = norm(𝛍ₖ[k] - 𝛍ₖ_before_epoch_start[k])
            total_centroid_movement += movement
        end
        
        if total_centroid_movement < tol
            if verbose
                println("Converged at Epoch $(epoch): Total movement ($total_centroid_movement) < Tolerance ($tol)")
            end
            break # Exit the while loop
        end
    
    end # End of for epoch loop
    
    if verbose
        print("The final centers: ")
        println(𝛍ₖ)
    end
    
    return 𝛍ₖ::Vector{Vector{Float64}}, epoch::Int64
end
