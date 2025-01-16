% This code solves the model using third-order perturbation.

%----------------------------------------------------------------
% 0. Housekeeping
%----------------------------------------------------------------

clc
clear
close all

%----------------------------------------------------------------
% 1. Parameters
%----------------------------------------------------------------

% Parameters of the procedure
model.znum = 3; % Number of points for technology grid
model.zgridstd = 1; % Number of standard deviations for technology grid
model.kgrid_cover = 0.25; % Percentage coverage for capital grid
model.knum_fin = 100; % Number of points for capital grid for final policy functions

% Parameters of the model
model.beta = 0.97;
model.delta = 0.1;
model.alpha = 0.33;
model.rrho = 0.95;
model.sigma = 0.007;

% Simulation
dropT     = 1000;  % Burn-in

%----------------------------------------------------------------
% 2. Steady State + Tauchen 
%----------------------------------------------------------------

k_to_l = ((1/model.alpha)*(1/model.beta - 1 + ...
    model.delta))^(1/(model.alpha-1));
c_to_l = k_to_l^model.alpha - model.delta*k_to_l;
lss = ((1-model.alpha)*(k_to_l^model.alpha)/c_to_l)^0.5;
css = c_to_l*lss;
kss = k_to_l*lss;
model.kss = kss;

[model.Pi, model.z_grid] = tauchen(model.znum, model.rrho, model.sigma, 0, model.zgridstd);

% Load exact solution obtained using endogenous grid method for comparison
load('EGM_sol.mat');
load('EGM_simul.mat');
load('EGM_IRF.mat');

%----------------------------------------------------------------
% 3. Use Dynare to get third-order perturbation
%----------------------------------------------------------------

dynare Q3

% Obtain policy functions
model = model_pol_fn(model, oo_);

% Obtain distribution of variables
mod_sim.kSeries = exp(oo_.endo_simul(strmatch('kk', M_.endo_names, 'exact'), dropT+1:end));  
mod_sim.cSeries = exp(oo_.endo_simul(strmatch('cc', M_.endo_names, 'exact'), dropT+1:end));  
mod_sim.lSeries = exp(oo_.endo_simul(strmatch('ll', M_.endo_names, 'exact'), dropT+1:end));
mod_sim.ySeries = exp(oo_.endo_simul(strmatch('yy', M_.endo_names, 'exact'), dropT+1:end)); 
mod_sim.rkSeries = exp(oo_.endo_simul(strmatch('rk', M_.endo_names, 'exact'), dropT+1:end));
mod_sim.z = oo_.endo_simul(strmatch('zz', M_.endo_names, 'exact'), dropT+1:end);

%----------------------------------------------------------------
% 4. Figures: plot comparison with exact solution 
%----------------------------------------------------------------

% Define colors for the dashed lines
cmap = colororder();

% Policy functions
figure(2)
subplot(2,2,1)
plot(model.k_grid_complete,EGM_res.vf,'--')
title('Value Function')
subplot(2,2,2)
plot(model.k_grid_complete,model.pol_c)
title('Consumption Decision Rule')
hold on;
for j = 1:model.znum
    plot(model.k_grid_complete,EGM_res.pol_c(:,j),'--', 'Color', cmap(j,:));
end
hold off;
subplot(2,2,3)
plot(model.k_grid_complete,model.pol_l)
title('Labor Decision Rule')
hold on;
for j = 1:model.znum
    plot(model.k_grid_complete,EGM_res.pol_l(:,j),'--', 'Color', cmap(j,:));
end
hold off;
subplot(2,2,4)
plot(model.k_grid_complete,model.pol_kp)
title('Capital Decision Rule')
hold on;
for j = 1:model.znum
    plot(model.k_grid_complete,EGM_res.pol_kp(:,j),'--', 'Color', cmap(j,:));
end
hold off;

% Distribution of simulated variables

[f_z,x_z]   = ksdensity(mod_sim.z);
[f_c,x_c]   = ksdensity(mod_sim.cSeries);
[f_k,x_k]   = ksdensity(mod_sim.kSeries);
[f_y,x_y]   = ksdensity(mod_sim.ySeries);
[f_l,x_l]   = ksdensity(mod_sim.lSeries);
[f_rk,x_rk] = ksdensity(mod_sim.rkSeries);

[f_z_EGM,x_z_EGM]   = ksdensity(mod_sim_EGM.z);
[f_c_EGM,x_c_EGM]   = ksdensity(mod_sim_EGM.cSeries);
[f_k_EGM,x_k_EGM]   = ksdensity(mod_sim_EGM.kSeries);
[f_y_EGM,x_y_EGM]   = ksdensity(mod_sim_EGM.ySeries);
[f_l_EGM,x_l_EGM]   = ksdensity(mod_sim_EGM.lSeries);
[f_rk_EGM,x_rk_EGM] = ksdensity(mod_sim_EGM.rkSeries);

figure(3)
subplot(3,2,1)
plot(x_c,f_c)
title('Density of Consumption')
hold on;
plot(x_c_EGM,f_c_EGM,'--', 'Color', cmap(1,:));
hold off;
subplot(3,2,2)
plot(x_l,f_l)
title('Density of Labor')
hold on;
plot(x_l_EGM,f_l_EGM,'--', 'Color', cmap(1,:));
hold off;
subplot(3,2,3)
plot(x_k,f_k)
title('Density of Capital')
hold on;
plot(x_k_EGM,f_k_EGM,'--', 'Color', cmap(1,:));
hold off;
subplot(3,2,4)
plot(x_y,f_y)
title('Density of Output')
hold on;
plot(x_y_EGM,f_y_EGM,'--', 'Color', cmap(1,:));
hold off;
subplot(3,2,5)
plot(x_rk,f_rk)
title('Density of Return on Capital')
hold on;
plot(x_rk_EGM,f_rk_EGM,'--', 'Color', cmap(1,:));
hold off;
subplot(3,2,6)
plot(x_z,f_z)
title('Density of Technological Shock')
hold on;
plot(x_z_EGM,f_z_EGM,'--', 'Color', cmap(1,:));
hold off;

%----------------------------------------------------------------
% 5. Functions
%----------------------------------------------------------------

% This function computes policy functions given Dynare output

function model = model_pol_fn(model, oo_)

    % Compute vector of state variables and shocks to be used for
    % generating policy functions
    % Joint grid
    
    model.k_min = model.kss*(1-model.kgrid_cover);
    model.k_max = model.kss*(1+model.kgrid_cover);
    model.k_grid = linspace(model.k_min, model.k_max, model.knum_fin);
    model.k_grid_complete = model.k_grid;
    
    [model.k_grid_j, model.eps_grid_j] = meshgrid(model.k_grid, model.z_grid);
    model.k_grid_j = model.k_grid_j';
    model.eps_grid_j = (1/model.sigma)*model.eps_grid_j';
    model.z_grid_j = zeros(model.knum_fin,model.znum);
    model.khat_grid_j = model.k_grid_j - model.kss;
    
    % Compute policy function
    g_l = zeros(model.knum_fin,model.znum);
    g_c = zeros(model.knum_fin,model.znum);
    g_k = zeros(model.knum_fin,model.znum);
    
    for k_index = 1:model.knum_fin
        for z_index = 1:model.znum
    
            % Get value of the vector with states and shocks
            xxi = [model.khat_grid_j(k_index,z_index); ...
                model.z_grid_j(k_index,z_index); model.eps_grid_j(k_index,z_index)];
    
            % Compute Kronecker products for cross-products
            kron1 = kron(xxi, xxi);
            kron2 = kron(kron1,xxi);
    
            % Remove repeated elements
            kron1 = [kron1(1);2*kron1(2:3);kron1(5);2*kron1(6);kron1(9)];
            kron2 = [kron2(1);3*kron2(2:3);3*kron2(5);6*kron2(6);...
                3*kron2(9);kron2(14);3*kron2(15);3*kron2(18);kron2(27)];
    
            % Compute policy functions using Dynare output
            g_l(k_index,z_index) = oo_.dr.ys(5) + oo_.dr.g_0(oo_.dr.inv_order_var(5)) + ...
                oo_.dr.g_1(oo_.dr.inv_order_var(5),:)*xxi + oo_.dr.g_2(oo_.dr.inv_order_var(5),:)*kron1 + ...
                 oo_.dr.g_3(oo_.dr.inv_order_var(5),:)*kron2;
            g_c(k_index,z_index) = oo_.dr.ys(3) + oo_.dr.g_0(oo_.dr.inv_order_var(3)) + ...
                oo_.dr.g_1(oo_.dr.inv_order_var(3),:)*xxi + oo_.dr.g_2(oo_.dr.inv_order_var(3),:)*kron1 + ...
                 oo_.dr.g_3(oo_.dr.inv_order_var(3),:)*kron2;
            g_k(k_index,z_index) = oo_.dr.ys(4) + oo_.dr.g_0(oo_.dr.inv_order_var(4)) + ...
                oo_.dr.g_1(oo_.dr.inv_order_var(4),:)*xxi + oo_.dr.g_2(oo_.dr.inv_order_var(4),:)*kron1 + ...
                 oo_.dr.g_3(oo_.dr.inv_order_var(4),:)*kron2;
    
        end

    end
    
    g_l = exp(g_l);
    g_c = exp(g_c);
    g_k = exp(g_k);
    
    model.pol_kp = g_k;
    model.pol_l = g_l;
    model.pol_c = g_c;

    

end