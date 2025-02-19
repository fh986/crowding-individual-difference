n_subjects = 300;
n_thresholds = 200;

% Ground truth
%   Bouma factor
gt.bouma.std  =  .2;  % log10 units
gt.bouma.mean = -.7;  % log10 units

%   Acuity
gt.acuity.std =    .1; % log10 units
gt.acuity.mean = -1.7; % log10 units

% Bouma v Acuity (correlation)
gt.BoumaAcuity.R = 0.5;

% Covariance matrix for the two measures 
%       [Var(bouma)     Cov(b vs a); 
%       Covar(b vs a)   Var(acuity)]
sigma(1,1) = gt.bouma.std.^2; 
sigma(2,2) = gt.acuity.std.^2;
sigma(1,2) = gt.bouma.std * gt.acuity.std * gt.BoumaAcuity.R;
sigma(2,1) = sigma(1,2);

% ground truth data
xx = mvnrnd([gt.bouma.mean gt.acuity.mean], sigma, n_subjects);
gt.bouma.data  = xx(:,1);
gt.acuity.data = xx(:,2);

% measurement noise (per threshold) - additive Gaussian in log space
noise.bouma.std  = .25;
noise.acuity.std = .15;


noise.bouma.data  = gt.bouma.data  + randn(n_subjects,n_thresholds)*noise.bouma.std;
noise.acuity.data = gt.acuity.data + randn(n_subjects,n_thresholds)*noise.acuity.std;

% % check
% figure, h=histogram(noise.bouma.data); hold on;
% bins = h.BinEdges;
% histogram(gt.bouma.data, bins);

figure(1),tiledlayout(3,2,"TileSpacing","compact");

% Bouma, ground truth vs data
nexttile();
scatter(gt.bouma.data, mean(noise.bouma.data,2))
xlabel('Ground Truth')
ylabel('Measurement')
r = corr(gt.bouma.data, mean(noise.bouma.data,2));
title(sprintf('Bouma factor, r = %3.2f', r)); hold on;
yl = get(gca, 'YLim'); axis([yl yl]); axis square;
hold on, plot(yl, yl, 'k--')
set(gca, 'FontSize', 20);

% Bouma, even vs odd measurements
nexttile();
data1 = mean(noise.bouma.data(:,1:2:end),2); % odd thresholds
data2 = mean(noise.bouma.data(:,2:2:end),2); % even thresholds
scatter(data1, data2)
xlabel('Odd thresholds ')
ylabel('Even thresholds')
r = corr(data1, data2);
title(sprintf('Bouma factor, r = %3.2f', r)); hold on;
yl = get(gca, 'YLim'); axis([yl yl]); axis square;
hold on, plot(yl, yl, 'k--')
set(gca, 'FontSize', 20);


% Acuity, ground truth vs data
nexttile();
scatter(gt.acuity.data, mean(noise.acuity.data,2))
xlabel('Ground Truth')
ylabel('Measurement')
r = corr(gt.acuity.data, mean(noise.acuity.data,2));
title(sprintf('Acuity, r = %3.2f', r)); hold on;
yl = get(gca, 'YLim'); axis([yl yl]); axis square;
hold on, plot(yl, yl, 'k--')
set(gca, 'FontSize', 20);

% Acuity, even vs odd measurements
nexttile();
data1 = mean(noise.acuity.data(:,1:2:end),2); % odd thresholds
data2 = mean(noise.acuity.data(:,2:2:end),2); % even thresholds
scatter(data1, data2)
xlabel('Odd thresholds ')
ylabel('Even thresholds')
r = corr(data1, data2);
title(sprintf('Acuity, r = %3.2f', r)); hold on;
yl = get(gca, 'YLim'); axis([yl yl]); axis square;
hold on, plot(yl, yl, 'k--')
set(gca, 'FontSize', 20);



% Bouma vs acuity, ground truth
nexttile();
scatter(gt.acuity.data, gt.bouma.data)
xlabel('Acuity (ground truth)')
ylabel('Bouma (ground truth)')
r = corr(gt.acuity.data, gt.bouma.data);
title(sprintf('r = %3.2f', r)); hold on;
axis square;
set(gca, 'FontSize', 20);

% Bouma vs acuity, data
nexttile();
data1 = mean(noise.acuity.data,2); 
data2 = mean(noise.bouma.data,2);  
scatter(data1, data2)
xlabel('Acuity (measurement)')
ylabel('Bouma (measurement)')
r = corr(data1, data2);
title(sprintf('r = %3.2f', r)); hold on;
axis square
set(gca, 'FontSize', 20);