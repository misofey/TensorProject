function [] = tensor_reconstruction()
%TENSOR_RECONSTRUCTION Summary of this function goes here
%   Detailed explanation goes here
    [image, M, N] = load_image();
    n = ndim(image);
    addpath("utils/");
    [omega, omega_overline, removed_idx, kept_idx] = create_masks(0.1)
    x_0 = zeros(M, N, 3);
    x_0(kept_idx(:, 1), kept_idx(:, 2), :) = image(kept_idx(:, 1), kept_idx(:, 2), :);
    omega_1 = mode_n_matricization(omega, 1);
    omega_2 = mode_n_matricization(omega, 2);
    omega_3 = mode_n_matricization(omega, 3);
    X_1 = mode_n_matricization(x_0, 1);
    X_2 = mode_n_matricization(x_0, 2);
    X_3 = mode_n_matricization(x_0, 3);
    M = [X_1, X_2, X_3];
    alpha = ones(1, n) / n;
    beta = ones(1, n);
end

function [image, M, N] = load_image()
    tiffObj = Tiff("image1.tiff", 'r');
    image = imread(tiffObj);
    sz = size(image);
    [M, N] = sz(1:2);
end

function [omega, removed_idx, kept_idx] = create_masks(removed_percentage)
    omega = [];
    omega_overline = [];
    removed_idx = [];
    kept_idx = [];
    image_size = 500;
    pixel_count = image_size*image_size;
    no_removed = round(pixel_count*removed_percentage);
    removed_flattened = randperm(pixel_count, no_removed);  % idx to remove from the flattened array
    removed_idx = zeros(no_removed, 2,  "int64");
    removed_idx(:, 1) = idivide(int64(removed_flattened), int64(image_size))+1;  % width removed ones
    removed_idx(:, 2) = (mod(removed_flattened, image_size))+1;  % height removed ones
    omega = ones(500);
    omega(removed_idx(:, 1), removed_idx(:, 2)) = 0;
    omega = cat(3, omega, omega, omega);
    omega_overline = 1-omega;
    [rk, ck] = find(omega(:,:,1) == 1);
    kept_idx = int64([rk ck]);
end
    
function [X] = silrtc(M, X, alphas, betas, n, idx_removed, K)
    % idx removed are the pixels removed from the image
    for j = 1:K
        selection = @(m) m(idx_removed(:, 1), idx_removed(:, 2), :);
        selected = zeros(length(idx_removed, 1));  % for collecting the average of all versions of the optimizaiton variables
        for i = 1:n
            M(i) = shrinkage(mode_n_matricization(X, i), alphas(i)/betas(i));
            folded = mode_n_folding(M(i));
            selected = selected + selection(folded) * betas(i);
        end
        X(idx_removed(:, 1), idx_removed(:, 2), :) = selected / sum(betas);
    end
end

function halrtc(M, X, Y, alphas, rho, n, idx_removed, K)
    selection = @(m) m(idx_removed(:, 1), idx_removed(:, 2), :);
    X_things = zeros(length(idx_removed, 1));  % for collecting the average of all versions of the optimizaiton variables
    
    for j = 1:K
        for i = 1:n
            M(i) = mode_n_folding(shrinkage(mode_n_matricization(X) + mode_n_matricization(Y, i) / rho, alphas(i) / rho));
            X_things = X_things + M(i) - Y(i) / rho;
        end
        X(idx_removed(:, 1), idx_removed(:, 2), :) = selection(X_things) / n;
        for i = 1:n
            Y(i) = Y(i) - (M(i) - X) * rho;
        end
    end
end

function [D] = shrinkage(X, tau)
    %% rounds tau down to the lowest nu
    [U, S, V] = svd(X);
    S_tau = max(S - tau, 0);
    D = U * S * V';
end

function [T] = truncation(X, tau)
    [U, S, V] = svd(X);
    S_tau = max(S, tau);
    T = U * S_tau * V';
end
