function outer = outer_product(varargin)
    % OUTER_PRODUCT takes a list of variable number of vectors varargin as input and
    % computes the outer product between them explicitly.
    % INPUT list of variable number of vectors varargin.
    % OUTPUT outer product tensor.
    if isempty(varargin)
        outer = [];
        return
    end
    lengths = [];
    for i=1:length(varargin)
        vector = varargin{i};
        lengths = [lengths length(vector)];
    end
    outer = ones(lengths);
    order = length(varargin);
    for i=1:order
        folding_permutation = [2:i 1 i+1:order];
        unfolding_permutation = [i 1:i-1 i+1:order];
        tensor_permutated = repmat(varargin{i}, [1 lengths(unfolding_permutation(2:end))]);
        tensor = permute(tensor_permutated, folding_permutation);
        outer = outer.*tensor;
    end
end


