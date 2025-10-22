function step2_matrix_completion_rgb3()
    rng(1);

    %% Load and normalize
    im_path = 'image1.tiff'; 
    Iraw = imread(im_path);
    if ndims(Iraw)==2
        Iraw = repmat(Iraw,[1 1 3]);
    end
    [M,N,C] = size(Iraw);
    if isa(Iraw,'uint16')
        I = double(Iraw)/65535;
    else
        I = im2double(Iraw);
    end

    %% Low-rank verification
    chans = {'R','G','B'};
    fprintf('Low-rank check (SVD per channel)\n');
    for c=1:3
        A = I(:,:,c);
        [~,S,~] = svd(A,'econ');
        s = diag(S); e = cumsum(s.^2)/sum(s.^2);
        k90 = find(e>=0.90,1);
        k95 = find(e>=0.95,1);
        k99 = find(e>=0.99,1);
        fprintf('Channel %s: k90=%d, k95=%d, k99=%d of %d\n',chans{c},k90,k95,k99,min(M,N));
    end

    %% Matrix completion parameters
    keep_list = [0.50, 0.75, 0.85];
    err = 1e-6; x0 = zeros(M*N,1);
    normfac = 1;
    insweep = 200;
    tol = 1e-4;
    decfac = 0.9;

    results = []; 
    keep_all = []; 
    Xhat_all = {}; 
    Iobs_all = {};

    %% Matrix completion loop
    for keep_ratio = keep_list
        fprintf('\n matrix completion: keep=%.0f%%\n', keep_ratio*100);

        num_obs = round(keep_ratio*M*N);
        Omega_idx = randperm(M*N, num_obs).';

        Mfun = @(v,flag) mask_op(v, flag, Omega_idx, [M N]);

        Xhat = zeros(M,N,3);
        for c=1:3
            A = I(:,:,c);
            y = A(Omega_idx);
            Xc = IST_MC(y, Mfun, [M N], err, x0, normfac, insweep, tol, decfac);
            Xhat(:,:,c) = min(max(Xc,0),1);
        end

        relerr = norm(I(:)-Xhat(:)) / norm(I(:));
        psnr_val = psnr_local(Xhat, I);
        fprintf('RELERR = %.4f, PSNR = %.2f dB\n', relerr, psnr_val);

        % Observed image
        Iobs = zeros(M,N,3);
        for c=1:3
            A = I(:,:,c); tmp = zeros(M,N); tmp(Omega_idx) = A(Omega_idx);
            Iobs(:,:,c) = tmp;
        end

        % visual part
        f = figure('Name', sprintf('Matrix completion – keep %.0f%%', keep_ratio*100), ...
                   'Color','w','Position',[100 100 1400 420]);
        tiledlayout(1,4,'Padding','compact','TileSpacing','compact');
        nexttile; imshow(I,[]); title('Original');
        nexttile; imshow(Iobs,[]); title(sprintf('Observed (%.0f%%)', keep_ratio*100));
        nexttile; imshow(Xhat,[]); 
        title(sprintf('Reconstruction\nRELERR=%.4f  PSNR=%.2f dB', relerr, psnr_val));
        errmap = mean(abs(I - Xhat),3);
        nexttile; imagesc(errmap); axis image off; colorbar; title('Error map (|I - \hat{I}|)');
        exportgraphics(f, sprintf('compare_matrix_IST_%d.png', round(keep_ratio*100)));
        imwrite(Xhat, sprintf('recon_matrix_IST_%d.png', round(keep_ratio*100)));
        close(f);

        results   = [results; keep_ratio, relerr, psnr_val];
        keep_all  = [keep_all; keep_ratio];
        Xhat_all{end+1} = Xhat;
        Iobs_all{end+1} = Iobs;
    end

    %% Save metrics
    T = array2table(results, 'VariableNames', {'keep_ratio','relerr','psnr_dB'});
    disp(T);
    writetable(T, 'metrics_matrix_IST.csv');

    %% summary
    fsum = figure('Name','Summary – Matrix completion (IST)','Color','w','Position',[50 50 1200 900]);
    tiledlayout(numel(keep_list), 3, 'Padding','compact', 'TileSpacing','compact');
    for i = 1:numel(keep_list)
        kr = keep_all(i);
        Xh = Xhat_all{i};
        em = mean(abs(I - Xh),3);

        nexttile; imshow(I,[]); if i==1, title('Original'); end
        ylabel(sprintf('keep %.0f%%', kr*100));

        r = results(i,2); p = results(i,3);
        nexttile; imshow(Xh,[]); if i==1, title('Reconstruction'); end
        text(5,15, sprintf('RELERR=%.4f\nPSNR=%.2f dB', r, p), ...
            'Color','w','FontSize',10,'FontWeight','bold','VerticalAlignment','top', ...
            'BackgroundColor',[0 0 0 0.35]);

        nexttile; imagesc(em); axis image off; colorbar; 
        if i==1, title('Error map'); end
    end
    exportgraphics(fsum, 'summary_matrix_IST.png');
    close(fsum);
end

%% Mask operator: 1-forward, 2-adjoint
function z = mask_op(v, flag, Omega_idx, sizeX)
    if flag==1
        if ~isvector(v), v = v(:); end
        z = v(Omega_idx);
    elseif flag==2
        z = zeros(prod(sizeX),1);
        z(Omega_idx) = v;
    else
        error('flag must be 1 or 2');
    end
end

%% PSNR for images in [0,1]
function val = psnr_local(Ahat, A)
    mse = mean((Ahat(:) - A(:)).^2);
    if mse == 0
        val = Inf;
    else
        val = 10*log10(1.0/mse); 
    end
end
