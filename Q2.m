% This code solves the model using finite elements and Galerkin weighting scheme.

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
model.knum = 8; % Number of nodes for capital
model.kgrid_cover = 0.25; % Percentage coverage for capital grid
model.knum_fin = 100; % Number of points for capital grid for final policy functions
model.intN = 1000; % Number of points for computing integrals

% Parameters of the model
model.beta = 0.97;
model.delta = 0.1;
model.alpha = 0.33;
model.rrho = 0.95;
model.sigma = 0.007;

% Simulation
T         = 10000; % Number of periods for the simulation of the economy
dropT     = 1000;  % Burn-in

% IRFs
T_irf = 100;

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
% 3. Finite elements using Galerkin weighting scheme
%----------------------------------------------------------------

tic

model.k_min = kss*(1-model.kgrid_cover);
model.k_max = kss*(1+model.kgrid_cover);
interval = kss*2*model.kgrid_cover;

% Partition of space for k
model.k_grid = linspace(model.k_min, model.k_max, model.knum);

% Initial guess for coefficients
theta_guess = ones(model.knum*model.znum,1)*lss;

options = optimset('Display','Iter','TolFun',10^(-8),'TolX',10^(-8),...
    'MaxFunEvals',10000);
[theta_coefs, err] = fsolve(@(x) syst_eq(x, model), theta_guess, options);

% Obtain policy functions
model = model_pol_fn(theta_coefs, model);

toc;

% Simulate the model to get distributions
mod_sim = simulate_model(theta_coefs, model, T, dropT, false);

mean_lerror    = sum(mod_sim.leeSeries)/(T-dropT);
max_lerror_sim = max(mod_sim.leeSeries);

disp(' ')
disp('Integral of Euler Equation Error:')
disp(mean_lerror)
disp('Max Euler Equation Error Simulation:')
disp(max_lerror_sim)

% Simuate the model to get IRFs
mod_IRF = simulate_model(theta_coefs, model, T_irf, 0, true);

%----------------------------------------------------------------
% 4. Figures: plot comparison with exact solution 
%----------------------------------------------------------------

% Define colors for the dashed lines
cmap = colororder();

% Policy functions
figure(1)
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

% Euler Equation Error on the Grid
figure(2)
plot(model.k_grid_complete,model.euler_lerror)
title('Log10 Euler Error')

% IRFs
tt = 1:T_irf;
figure(3)
subplot(3,2,1)
plot(tt,mod_IRF.z);
title('Technological Shock')
hold on;
plot(tt,EGM_IRF.z,'--', 'Color', cmap(1,:));
hold off;
subplot(3,2,2)
plot(tt,mod_IRF.lSeries);
title('Labor')
hold on;
plot(tt,EGM_IRF.lSeries,'--', 'Color', cmap(1,:));
hold off;
subplot(3,2,3)
plot(tt,mod_IRF.kSeries);
title('Capital')
hold on;
plot(tt,EGM_IRF.kSeries,'--', 'Color', cmap(1,:));
hold off;
subplot(3,2,4)
plot(tt,mod_IRF.ySeries);
title('Output')
hold on;
plot(tt,EGM_IRF.ySeries,'--', 'Color', cmap(1,:));
hold off;
subplot(3,2,5)
plot(tt,mod_IRF.cSeries);
title('Consumption')
hold on;
plot(tt,EGM_IRF.cSeries,'--', 'Color', cmap(1,:));
hold off;
subplot(3,2,6)
plot(tt,mod_IRF.rkSeries);
title('Return on Capital')
hold on;
plot(tt,EGM_IRF.rkSeries,'--', 'Color', cmap(1,:));
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

figure(4)
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

% Define finite elements
function vv = fin_elem(ind,k,k_grid,knum)

    if ind == 1
        vv = (k < k_grid(2))*(k_grid(2)-k)/(k_grid(2) - k_grid(1));
    elseif ind == knum
        vv = (k >= k_grid(knum-1))*(k-k_grid(knum-1))/(k_grid(knum)-k_grid(knum-1));
    else
        vv = (k >= k_grid(ind-1) & k < k_grid(ind))*(k-k_grid(ind-1))/...
            (k_grid(ind)-k_grid(ind-1)) + (k >= k_grid(ind) & k < k_grid(ind+1))* ...
            (k_grid(ind+1)-k)/(k_grid(ind+1)-k_grid(ind));
    end

end

% This function computes residual of Euler's equation
% given current k, z_index and coefficients of approximation

function residual = Euler_resid(k, z_index, theta_coefs, model)
    
    % Choose appropriate part of theta_coefs
    theta_section = theta_coefs(((z_index-1)* ...
            model.knum+1):z_index*model.knum);

    % Evaluate model.knum finite elements
    fin_elem_eval = zeros(model.knum,1);
    for i = 1:model.knum
        fin_elem_eval(i) = fin_elem(i,k,model.k_grid,model.knum);
    end
    
    % Compute labor
    l = dot(theta_section,fin_elem_eval);
    if(l<0.01)
        fprintf('l break lower bound: %d \n',l);
        l = 0.01;
    end

    % Compute consumption, output, kp
    c = (1-model.alpha)*exp(model.z_grid(z_index))*...
                k^model.alpha*l^(-1-model.alpha);
    y = exp(model.z_grid(z_index))*k^model.alpha*l^(1-model.alpha);
    kp = y+(1-model.delta)*k-c;

    if(kp < model.k_min)
        fprintf('kp break lower bound: %d \n',kp);
        kp = model.k_min+0.01;
    elseif(kp > model.k_max)
        fprintf('kp break upper bound: %d \n',kp);
        kp = model.k_max - 0.01;
    end

    c = y+(1-model.delta)*k-kp;
            
    if(c<0)
        disp('warning: c < 0')
        c = 10^(-10);
    end

    % Compute next period choices
    % Compute model.knum finite elements
    fin_elem_eval = zeros(model.knum,1);
    for i = 1:model.knum
        fin_elem_eval(i) = fin_elem(i,kp,model.k_grid,model.knum);
    end

    for zp_index = 1:model.znum
    
        theta_section = theta_coefs(((zp_index-1)*...
            model.knum+1):zp_index*model.knum);
        lp = dot(theta_section,fin_elem_eval);  
    
        if(lp<0.01)
            fprintf('lp break lower bound: %d \n',lp);
            lp = 0.01;
        end
    
        yp = exp(model.z_grid(zp_index))*...
            kp^model.alpha*lp^(1-model.alpha);
        cp = (1-model.alpha)*exp(model.z_grid(zp_index))*...
            kp^model.alpha*lp^(-1-model.alpha);
    
        kpp = yp+(1-model.delta)*kp-cp;
    
        if(kpp<model.k_min)
            fprintf('kpp break lower bound: %d \n',kpp);
            kpp = model.k_min+0.01;
        elseif(kpp>model.k_max)
            fprintf('kpp break upper bound: %d \n',kpp);
            kpp = model.k_max - 0.01;
        end
    
        cp = yp+(1-model.delta)*kp-kpp;
        if(cp<0)
            disp('warning: cp < 0')
            cp = 10^(-10);
        end             
    
        Ucp = 1/cp;
        Fkp = model.alpha*exp(model.z_grid(zp_index))*...
            kp^(model.alpha-1)*lp^(1-model.alpha);
        temp(zp_index) = Ucp*(Fkp+1-model.delta);
                
    end

    euler_rhs = model.beta*dot(model.Pi(z_index,:),temp);
                    
    euler_lhs = 1/c;
    
    residual = euler_rhs - euler_lhs;

end

% This function computes residuals of the system of equations in 
% coefficients used for approximation (theta)
function res = syst_eq(theta_coefs, model)
    
    res = zeros(model.knum,model.znum);

    for j = 1:model.znum
        for i = 1:model.knum

            % Define parameters for integral
            % Starting point
            if i == 1
                st_point = model.k_grid(1);
            else
                st_point = model.k_grid(i-1);
            end
            if i == model.knum
                end_point = model.k_grid(model.knum);
            else
                end_point = model.k_grid(i+1);
            end

            vval = 0;

            % intN is the amount of intervals used for approximating intergral
            for currentInt=1:model.intN
        
                % Current interval bounds
                a = st_point + (currentInt - 1) * (end_point - st_point) / model.intN;
                b = st_point + currentInt * (end_point - st_point) / model.intN;
    
                % Value of the integral
                resid = Euler_resid((a+b)/2, j, theta_coefs, model);
                vval = vval + (b-a) * fin_elem(i,(a+b)/2,model.k_grid,model.knum) * resid; 

            end

            res(i,j) = vval;
            
        end
    end

    res = res(:);

end

% This function computes policy functions given coefficients for 
% approximation
function model = model_pol_fn(theta_coefs, model)

    model.k_grid_complete = linspace(model.k_min, model.k_max, model.knum_fin)';

    euler_error = zeros(model.knum_fin,model.znum);
    g_l = zeros(model.knum_fin,model.znum);
    g_c = zeros(model.knum_fin,model.znum);
    g_k = zeros(model.knum_fin,model.znum);
    fin_elem_eval = zeros(model.knum,1);

    for z_index = 1:model.znum

        theta_section = theta_coefs(((z_index-1)*model.knum+1):z_index*model.knum);
        
        for k_index = 1:model.knum_fin % Loop 1 over collocation point on k

            % Compute model.knum finite elements       
            for i = 1:model.knum
                fin_elem_eval(i) = fin_elem(i,model.k_grid_complete(k_index),...
                    model.k_grid,model.knum);
            end
            
            % Compute labor
            g_l(k_index,z_index) = dot(theta_section,fin_elem_eval);

            y = exp(model.z_grid(z_index))*model.k_grid_complete(k_index)^...
                model.alpha*g_l(k_index,z_index)^(1-model.alpha);
            g_c(k_index,z_index) = (1-model.alpha)*exp(model.z_grid(z_index))*...
                model.k_grid_complete(k_index)^model.alpha*...
                g_l(k_index,z_index)^(-1-model.alpha);            
            g_k(k_index,z_index) = y+(1-model.delta)*model.k_grid_complete(k_index)-...
                g_c(k_index,z_index);            

        end % Loop 1 over collocation point on k ends

        % Calculate residual        
        for k_index = 1:model.knum_fin % Loop 2 over collocation point on k              
            
            temp = zeros(model.znum,1);
        
            for zp_index = 1:model.znum

                % Compute model.knum finite elements -- for k prime
                fin_elem_eval = zeros(model.knum,1);
                for i = 1:model.knum
                    fin_elem_eval(i) = fin_elem(i,g_k(k_index,z_index),...
                        model.k_grid,model.knum);
                end 
            
                theta_section = theta_coefs(((zp_index-1)*model.knum+1):zp_index*model.knum);
                lp = dot(theta_section,fin_elem_eval);

                yp = exp(model.z_grid(zp_index))*g_k(k_index,z_index)^...
                    model.alpha*lp^(1-model.alpha);
                cp = (1-model.alpha)*exp(model.z_grid(zp_index))*...
                    g_k(k_index,z_index)^model.alpha*...
                    lp^(-1-model.alpha);  

                Ucp = 1/cp;
                Fkp = model.alpha*exp(model.z_grid(zp_index))*...
                    g_k(k_index,z_index)^(model.alpha-1)*lp^(1-model.alpha);
                temp(zp_index) = Ucp*(Fkp+1-model.delta);
            
            end
            
            euler_rhs = model.beta*dot(model.Pi(z_index,:),temp);

            euler_error(k_index,z_index) = euler_rhs - 1/g_c(k_index,z_index);

        end % Loop 2 over k ends

    end % Loop over z ends

    euler_lerror = log10(abs(euler_error));
    max_error = max(euler_error(:),[],1);
    max_lerror = max(euler_lerror(:),[],1);

    model.pol_kp = g_k;
    model.pol_l = g_l;
    model.pol_c = g_c;
    model.euler_error = euler_error;
    model.euler_lerror = euler_lerror;
    model.max_euler_error = max_error;
    model.max_euler_lerror = max_lerror;

end

% This function simulates the model given coefficients for
% approximation
function mod_sim = simulate_model(theta_coefs, model, T, dropT, one_time)

    mod_sim.kSeries      = zeros(T,1);
    mod_sim.cSeries      = zeros(T,1);
    mod_sim.lSeries      = zeros(T,1);
    mod_sim.ySeries      = zeros(T,1);
    mod_sim.eeSeries     = zeros(T,1);
    mod_sim.leeSeries    = zeros(T,1);
    mod_sim.rkSeries     = zeros(T,1);
    mod_sim.rkCondSeries = zeros(T,1);

    if one_time % One-time positive z shock
        
        mod_sim.z_index(1) = round(model.znum/2); % Start with zero shock
        mod_sim.z_index(2) = model.znum; % Then maximum positive shock
        mod_sim.z_index(3:T) = round(model.znum/2); % Then zero shock
    
    else % Simulation to get distributions with shocks arriving each period

        % Generate Markov Chain
        rng(123);                        % Set random seed for reproducibility
        mu = zeros(1,model.znum);        % initial distribution
        mu(1,round(model.znum/2)) = 1;   % Always start from Z=0
        mod_sim.z_index = zeros(1,T+1);  % Stores indexes of the realization of "z" over the simulation
        mod_sim.z_index(1) = rando(mu);  % generate first ind value (time 0, not time 1)
    
        for i=1:T
            mod_sim.z_index(i+1) = rando(model.Pi(mod_sim.z_index(i),:));
        end

    end
    
    mod_sim.kSeries(1) = model.kss;
    
    for t_index = 1:T
        
        cursim.z_index = mod_sim.z_index(t_index);
        cursim.k = mod_sim.kSeries(t_index); 
        sim_step = simulate_one_step(theta_coefs, model, cursim);

        mod_sim.kSeries(t_index+1) = sim_step.kp;
        mod_sim.cSeries(t_index) = sim_step.c;
        mod_sim.lSeries(t_index) = sim_step.l;
        mod_sim.ySeries(t_index) = sim_step.y;
        mod_sim.eeSeries(t_index) = sim_step.euler_error;
        mod_sim.leeSeries(t_index) = sim_step.euler_lerror;
        mod_sim.rkSeries(t_index) = sim_step.rk;
        mod_sim.rkCondSeries(t_index) = sim_step.rkCond;

    end

    mod_sim.kSeries = mod_sim.kSeries(dropT+1:T);   
    mod_sim.cSeries = mod_sim.cSeries(dropT+1:T);   
    mod_sim.lSeries = mod_sim.lSeries(dropT+1:T); 
    mod_sim.ySeries = mod_sim.ySeries(dropT+1:T);   
    mod_sim.eeSeries = mod_sim.eeSeries(dropT+1:T);   
    mod_sim.leeSeries = mod_sim.leeSeries(dropT+1:T);  
    mod_sim.rkSeries = mod_sim.rkSeries(dropT+1:T); 
    mod_sim.rkCondSeries = mod_sim.rkCondSeries(dropT+1:T);
    mod_sim.z_index = mod_sim.z_index(dropT+1:T)';
    mod_sim.z = model.z_grid(mod_sim.z_index)';

end

% This function computes one step of the simulation given
% coefficients for approximation

function sim_step = simulate_one_step(theta_coefs, model, cursim)
    
    z_index = cursim.z_index;
    z = model.z_grid(z_index);

    % Evaluate finite elements
    fin_elem_eval = zeros(model.knum,1);
    for i = 1:model.knum
        fin_elem_eval(i) = fin_elem(i,cursim.k,model.k_grid,model.knum);
    end
  
    theta_section = theta_coefs(((z_index-1)*model.knum+1):z_index*model.knum);
    l = dot(theta_section,fin_elem_eval);       % Labor at each collocation points

    y = exp(z)*cursim.k^model.alpha*l^(1-model.alpha);
    c = (1-model.alpha)*exp(z)*cursim.k^model.alpha*l^(-1-model.alpha);            
    kp = y+(1-model.delta)*cursim.k-c;
    
    % Evaluate finite elements -- for k prime
    fin_elem_eval = zeros(model.knum,1);
    for i = 1:model.knum
        fin_elem_eval(i) = fin_elem(i,kp,model.k_grid,model.knum);
    end

    % calculate residual  
    fkp  = zeros(model.znum,1);
    temp = zeros(model.znum,1);

    for zp_index = 1:model.znum

        theta_section = theta_coefs(((zp_index-1)*model.knum+1):zp_index*model.knum);
        lp = dot(theta_section,fin_elem_eval);

        yp = exp(model.z_grid(zp_index))*kp^model.alpha*lp^(1-model.alpha);
        cp = (1-model.alpha)*exp(model.z_grid(zp_index))*...
            kp^model.alpha*lp^(-1-model.alpha); 

        Ucp = 1/cp;
        fkp(zp_index) = model.alpha*exp(model.z_grid(zp_index))*...
            kp^(model.alpha-1)*lp^(1-model.alpha);
        temp(zp_index) = Ucp*(fkp(zp_index)+1-model.delta);
    
    end    

    euler_rhs = model.beta*dot(model.Pi(z_index,:),temp);
    sim_step.euler_error = euler_rhs - 1/c;
    sim_step.euler_lerror = log10(abs(sim_step.euler_error));

    sim_step.rk = model.alpha*y/cursim.k-model.delta;       % risky asset return between t-1 and t
    sim_step.rkCond = dot(model.Pi(z_index,:),fkp);  % conditional rk next period

    sim_step.kp = kp;
    sim_step.c = c;
    sim_step.l = l;
    sim_step.y = y;

end