using LinearAlgebra
using Statistics

#FOR EACH DATA POINT IN 𝐱ₙ COMPUTE THE DISTANCE TO EACH CENTER IN 𝛍ₖ and indicate which center is the closest to each data point
function compDistInd(𝐱ₙ::Vector{Vector{Float64}},𝛍ₖ::Vector{Vector{Float64}})
    N = length(𝐱ₙ) # Number of data points (rows)
    K = length(𝛍ₖ) # Number of centers (columns)

    d = zeros(Float64, (N, K)) 
    r = fill(false, N, K)        #allocate memeory for r

      for n in 1:N
        # Temporary array to hold distances d_n1, d_n2, ..., d_nK for the current point
        dist_to_centers = zeros(Float64, K) 

        # 1. Compute all distances d_nk (Line 2 of Algorithm 2)
        for k in 1:K
            # Calculate the Euclidean distance (L2 norm) between x_n and μ_k
            distance = norm(𝐱ₙ[n] - 𝛍ₖ[k])
            d[n, k] = distance
            dist_to_centers[k] = distance
        end
        
        # 2. Find the closest center k* (Line 3 of Algorithm 2)
        # argmin returns the index (k*) that minimizes the value.
        closest_center_index = argmin(dist_to_centers)
        
        # 3. Set the indicator r_nk* to true
        r[n, closest_center_index] = true
    end

  return d::Matrix{Float64},r::Matrix{Bool}
end


