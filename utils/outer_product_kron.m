function outer = outer_product_kron(varargin)
    % OUTER_PRODUCT_KRON takes a list of variable number of vectors varargin as input and
    % computes the outer product between them using the kron function which
    % is builtin in MATLAB.
    % INPUT list of variable number of vectors varargin.
    % OUTPUT outer product tensor.
    order = length(varargin);
    outer = varargin{order};
    lengths = [];
    for i=1:length(varargin)
        vector = varargin{i};
        lengths = [lengths length(vector)];
    end
    for i=order-1:-1:1 
        outer = kron(outer, varargin{i});
    end
    outer = reshape(outer, lengths);
end
