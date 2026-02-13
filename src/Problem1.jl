
function histo(data_points::Vector{Float64}, num_bins::Int)
    # 1. Determine Data Range
    data_min = minimum(data_points)
    data_max = maximum(data_points)
    
    # 2. Define Bin Centers (b_n)
    # Note: The `range` function is Julia's equivalent to `linspace`
    bin_centers = collect(range(data_min, stop=data_max, length=num_bins))
    
    # 3. Initialize Bin Counts (h_n)
    bin_counts = zeros(Int, num_bins) # Use Int for counts
    
    M = length(data_points)

    # 4. Double Loop for Nearest-Neighbor Assignment
    for point_index in 1:M # Outer loop over data points (m)
        current_point = data_points[point_index]
        
        # Temporary vector to store all distances (e_n)
        distance_vector = zeros(num_bins)
        
        # Inner loop to calculate distance to every center (n)
        for center_index in 1:num_bins
            distance_vector[center_index] = abs(bin_centers[center_index] - current_point)
        end
        
        # 5. Find Closest Bin (i)
        # `argmin` finds the index of the minimum value
        closest_center_index = argmin(distance_vector)
        
        # 6. Assign and Count (h_i <- h_i + 1)
        bin_counts[closest_center_index] += 1
    end
    
    # Return bin centers and counts
    return bin_counts, bin_centers 
end

