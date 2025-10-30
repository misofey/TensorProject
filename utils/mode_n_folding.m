function [X] = mode_n_folding(X, n, og_size)
    [unfolding_permutation, folding_permutation] = mode_n_permutations(n, og_size);
    permuted_shape = og_size(unfolding_permutation);
    a = size(X);
    permuted_shape(1) = a(1);
    X = reshape(X, permuted_shape);
    X = permute(X, folding_permutation);
end

function [unfolding_permutation, folding_permutation] = mode_n_permutations(n, og_size)
    order = length(og_size);
    unfolding_permutation = [n 1:n-1 n+1:order];
    folding_permutation = [2:n 1 n+1:order];
end