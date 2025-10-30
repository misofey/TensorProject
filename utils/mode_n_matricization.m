function X = mode_n_matricization(X,n)
% MODE_K_MATRICIZATION takes a tensor X as input and a mode n such
% that n<ndims(X) and returns the mode-n matricization of X.
% INPUTS tensor X, mode n.
% OUPUT mode-n matricization of X.
order = ndims(X);
dim_X = size(X);
unfolding_permutation = [n 1:n-1 n+1:order];

permuted_tensor = permute(X, unfolding_permutation);

output_size = [dim_X(n) prod(dim_X)/dim_X(n)];
X = reshape(permuted_tensor, output_size);
end