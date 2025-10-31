function matrix_completion_rgb()
    rng(1);

    outdir = "results_matrix";
    if ~exist(outdir, "dir")
        mkdir(outdir);
    end

    image_list = {"image1.tiff","image2.tiff","image3.tiff"};
    keep_list = [0.50, 0.75, 0.85];
    err = 1e-6;
    normfac = 1;
    insweep = 200;
    tol = 1e-4;
    decfac = 0.9;

    PSNRs_all = cell(numel(image_list),1);
    RERRs_all = cell(numel(image_list),1);

    for img = 1:numel(image_list)
        im_path = image_list{img};

        % Load and normalize
        Iraw = imread(im_path);
        if ndims(Iraw)==2, Iraw = repmat(Iraw,[1 1 3]); end
        [M,N,C] = size(Iraw);
        if isa(Iraw,'uint16')
            I = double(Iraw)/65535;
        else
            I = im2double(Iraw);
        end

        % Low-rank verify
        chans = {'R','G','B'};
        fprintf(im_path);
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

        PSNRs  = zeros(numel(keep_list),1);
        RERRs  = zeros(numel(keep_list),1);
        TIMES  = zeros(numel(keep_list),1);
        results = [];
        keep_all = [];
        Xhat_all = {};
        Iobs_all = {};
        x0 = zeros(M*N,1);

        for k = 1:numel(keep_list)
            keep_ratio = keep_list(k);
            fprintf('\n[matrix completion %s] keep=%.0f%%\n', im_path, keep_ratio*100);
            num_obs = round(keep_ratio*M*N);
            Omega_idx = randperm(M*N, num_obs).';
            Mfun = @(v,flag) mask_op(v, flag, Omega_idx, [M N]);

            t0 = tic;
            Xhat = zeros(M,N,3);
            for c=1:3
                A = I(:,:,c);
                y = A(Omega_idx);
                Xc = IST_MC(y, Mfun, [M N], err, x0, normfac, insweep, tol, decfac);
                Xhat(:,:,c) = min(max(Xc,0),1);
            end
            t_elapsed = toc(t0);


            relerr   = norm(I(:)-Xhat(:)) / norm(I(:));
            psnr_val = psnr_local(Xhat, I);
            fprintf('RELERR = %.4f, PSNR = %.2f dB, time = %.3fs\n', relerr, psnr_val, t_elapsed);
            PSNRs(k) = psnr_val;
            RERRs(k) = relerr;
            TIMES(k) = t_elapsed;

            % observed img
            Iobs = zeros(M,N,3);
            for c=1:3
                A = I(:,:,c); tmp = zeros(M,N); tmp(Omega_idx) = A(Omega_idx);
                Iobs(:,:,c) = tmp;
            end

            % figures
            f = figure('Name', sprintf('Matrix completion – %s – keep %.0f%%', im_path, keep_ratio*100), 'Color','w','Position',[100 100 1400 420]);
            tiledlayout(1,4,'Padding','compact','TileSpacing','compact');
            nexttile; imshow(I,[]); title('Original');
            nexttile; imshow(Iobs,[]); title(sprintf('Observed (%.0f%%)', keep_ratio*100));
            nexttile; imshow(Xhat,[]);
            title(sprintf('Reconstruction\nRELERR=%.4f  PSNR=%.2f dB', relerr, psnr_val));
            errmap = mean(abs(I - Xhat),3);
            nexttile; imagesc(errmap); axis image off; colorbar; title('Error map (|I - \hat{I}|)');

            base = erase(im_path,{'.tiff','.tif'});
            exportgraphics(f, fullfile(outdir, sprintf('compare_matrix_%s_%d.png', base, round(keep_ratio*100))));
            imwrite(Xhat, fullfile(outdir, sprintf('recon_matrix_%s_%d.png', base, round(keep_ratio*100))));
            close(f);

            results   = [results; keep_ratio, relerr, psnr_val];
            keep_all  = [keep_all; keep_ratio];
            Xhat_all{end+1} = Xhat;
            Iobs_all{end+1} = Iobs;
        end

        %% Save metrics per image
        T = array2table(results, 'VariableNames', {'keep_ratio','relerr','psnr_dB'});
        disp(T);
        base = erase(im_path,{'.tiff','.tif'});
        writetable(T, fullfile(outdir, sprintf('metrics_matrix_IST_%s.csv', base)));
        fsumM = figure('Name',sprintf('Summary – Matrix (IST) – %s', base),'Color','w','Position',[50 50 1350 360]);
        tiledlayout(1,3,'Padding','compact','TileSpacing','compact');
        nexttile; plot(keep_list*100, PSNRs, '-o','LineWidth',1.5); grid on;
        xlabel('Keep [%]'); ylabel('PSNR [dB]'); title(sprintf('%s: PSNR vs keep', base));
        nexttile; plot(keep_list*100, RERRs, '-o','LineWidth',1.5); grid on;
        xlabel('Keep [%]'); ylabel('RelErr'); title(sprintf('%s: RelErr vs keep', base));
        nexttile; bar(keep_list*100, TIMES); grid on;
        xlabel('Keep [%]'); ylabel('Time [s]'); title(sprintf('%s: Time vs keep', base));
        exportgraphics(fsumM, fullfile(outdir, sprintf('matrix_summary_metrics_%s.png', base)));
        close(fsumM);

        fsum = figure('Name',sprintf('Summary – Matrix completion (IST) – %s', base),'Color','w','Position',[50 50 1200 900]);
        tiledlayout(numel(keep_list), 3, 'Padding','compact', 'TileSpacing','compact');
        for i = 1:numel(keep_list)
            kr = keep_all(i);
            Xh = Xhat_all{i};
            em = mean(abs(I - Xh),3);

            nexttile; imshow(I,[]); if i==1, title('Original'); end
            ylabel(sprintf('keep %.0f%%', kr*100));
            r = results(i,2); p = results(i,3);
            nexttile; imshow(Xh,[]); if i==1, title('Reconstruction'); end
            text(5,15, sprintf('RELERR=%.4f\nPSNR=%.2f dB', r, p), 'Color','w','FontSize',10,'FontWeight','bold','VerticalAlignment','top', 'BackgroundColor',[0 0 0 0.35]);
            nexttile; imagesc(em); axis image off; colorbar;
            if i==1, title('Error map'); end
        end
        exportgraphics(fsum, fullfile(outdir, sprintf('summary_matrix_IST_%s.png', base)));
        close(fsum);

        PSNRs_all{img} = PSNRs;
        RERRs_all{img} = RERRs;
    end

    fall = figure('Name','Matrix IST – All images','Color','w','Position',[100 100 900 360]);
    tiledlayout(1,2,'Padding','compact','TileSpacing','compact');

    nexttile; hold on; grid on;
    for img = 1:numel(image_list)
        base = erase(image_list{img},{'.tiff','.tif'});
        plot(keep_list*100, PSNRs_all{img}, '-o','LineWidth',1.5, 'DisplayName', base);
    end
    xlabel('Keep [%]'); ylabel('PSNR [dB]'); title('PSNR vs keep (all images)'); legend('Location','best');

    nexttile; hold on; grid on;
    for img = 1:numel(image_list)
        base = erase(image_list{img},{'.tiff','.tif'});
        plot(keep_list*100, RERRs_all{img}, '-o','LineWidth',1.5, 'DisplayName', base);
    end
    xlabel('Keep [%]'); ylabel('RelErr'); title('RelErr vs keep (all images)'); legend('Location','best');

    exportgraphics(fall, fullfile(outdir, 'matrix_summary_metrics_ALL.png'));
    close(fall);
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
