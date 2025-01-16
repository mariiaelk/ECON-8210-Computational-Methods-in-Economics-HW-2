% This code solves the model using value function iteration.

%----------------------------------------------------------------
%0. Housekeeping
%----------------------------------------------------------------

clc
clear
close all

%----------------------------------------------------------------
% 1. Parameters
%----------------------------------------------------------------

% Parameters of the procedure
modpar.znum = 3; % Number of nodes for technology grid
modpar.zgridstd = 1; % Number of standard deviations for technology grid
modpar.knum = 100; % Number of nodes for capital
modpar.kgrid_cover = 0.25; % Percentage coverage for capital grid

% Parameters of the model
modpar.beta = 0.97;
modpar.delta = 0.1;
modpar.alpha = 0.33;
modpar.rrho = 0.95;
modpar.sigma = 0.007;

% Simulation
T         = 10000; % Number of periods for the simulation of the economy
dropT     = 1000;  % Burn-in

% IRFs
T_irf = 100;

%----------------------------------------------------------------
% 2. Steady State + Tauchen 
%----------------------------------------------------------------

k_to_l = ((1/modpar.alpha)*(1/modpar.beta - 1 + ...
    modpar.delta))^(1/(modpar.alpha-1));
c_to_l = k_to_l^modpar.alpha - modpar.delta*k_to_l;
lss = ((1-modpar.alpha)*(k_to_l^modpar.alpha)/c_to_l)^0.5;
modpar.lss = lss;
css = c_to_l*lss;
kss = k_to_l*lss;
modpar.kss = kss;
vss = log(css) - 0.5*lss^2;

[modpar.Pi, modpar.z_grid] = tauchen(modpar.znum, modpar.rrho, modpar.sigma, 0, modpar.zgridstd);

%----------------------------------------------------------------
% 3. Value function iteration: step 1, fix labor at lss
%----------------------------------------------------------------

tic

% Grid for tomorrow capital (k')
modpar.k_min = kss*(1-modpar.kgrid_cover);
modpar.k_max = kss*(1+modpar.kgrid_cover);
modpar.k_grid = linspace(modpar.k_min, modpar.k_max, modpar.knum);

% Grid for market resources
% Joint grid for k' and z'
[modpar.k_grid_j, modpar.z_grid_j] = meshgrid(modpar.k_grid, modpar.z_grid);
modpar.k_grid_j = modpar.k_grid_j';
modpar.z_grid_j = modpar.z_grid_j';

% Compute y' on the grid
modpar.y_grid_j = exp(modpar.z_grid_j) .* (modpar.k_grid_j .^ ...
    modpar.alpha) .* (lss^(1 - modpar.alpha)) + (1 - ...
    modpar.delta) .* modpar.k_grid_j;

% Guess for value function
modpar.ev = modpar.y_grid_j;                     % Expected value function EV(k',z)
modpar.vf = zeros(modpar.knum, modpar.znum);     % Value function v(y,z) or equiv. y(k,z)
modpar.pol_y = zeros(modpar.knum, modpar.znum);  % Policy function y(k',z)
modpar.pol_k = zeros(modpar.knum, modpar.znum);  % Policy function for k(k',z)
modpar.pol_kp = zeros(modpar.knum, modpar.znum); % Policy function for kp(k,z)
modpar.pol_l = zeros(modpar.knum, modpar.znum);  % Policy function for labor(k,z)
modpar.pol_c = zeros(modpar.knum, modpar.znum);  % Policy function for consumption(k,z)


% Get solution of the model
modpar = vfi_perf_s1(modpar);

% Plot value function and policy function for k'
% Decision Rules
%figure(1)
%subplot(1,2,1)
%plot(modpar.k_grid,modpar.vf)
%title('Value Function: fixed labor')
%subplot(1,2,2)
%plot(modpar.k_grid,modpar.pol_kp)
%title('Capital Decision Rule: fixed labor')

%----------------------------------------------------------------
% 4. Value function iteration: step 2, account for endo labor
%----------------------------------------------------------------

% Initial step of VFI to obtain guesses for value function
modpar = initial_update_s2(modpar);

% Get solution of the model
modpar = vfi_perf_s2(modpar);

toc;

% Compute Euler equation errors
modpar.euler_error = Euler_err(modpar);
modpar.euler_lerror = log10(abs(modpar.euler_error));
modpar.max_error = max(modpar.euler_error(:),[],1);
modpar.max_lerror = max(modpar.euler_lerror(:),[],1);

% Compute resource constraint errors
err2 = modpar.pol_kp + modpar.pol_c - exp(modpar.z_grid_j).*...
    modpar.k_grid_j.^modpar.alpha.*modpar.pol_l.^(1-modpar.alpha) - ...
    (1-modpar.delta)*modpar.k_grid_j;

EGM_res = modpar;

save('EGM_sol.mat', 'EGM_res');

% Simulate the model to get distributions
mod_sim_EGM = simulate_model(modpar, T, dropT, false);

save('EGM_simul.mat', 'mod_sim_EGM');

% Simuate the model to get IRFs
EGM_IRF = simulate_model(modpar, T_irf, 0, true);

save('EGM_IRF.mat', 'EGM_IRF');

% Decision Rules
figure(2)
subplot(2,2,1)
plot(modpar.k_grid,modpar.vf)
title('Value Function')
subplot(2,2,2)
plot(modpar.k_grid,modpar.pol_c)
title('Consumption Decision Rule')
subplot(2,2,3)
plot(modpar.k_grid,modpar.pol_l)
title('Labor Decision Rule')
subplot(2,2,4)
plot(modpar.k_grid,modpar.pol_kp)
title('Capital Decision Rule')

%----------------------------------------------------------------
% 5. Functions
%----------------------------------------------------------------

% This function updates modpar.ev, modpar.vf, modpar.pol_y for first part
% of the algorithm when l is fixed at its steady state value
function modpar = one_step_update_s1(modpar)

    % Compute values of derivative of the value function
    [N, M] = size(modpar.ev);    
    dk = diff(modpar.k_grid);  % Step sizes for k' grid (length N-1)
    
    % Preallocate matrix for derivative w.r.t. k'
    dG_dk = zeros(N, M);
    
    % Compute partial derivative w.r.t. k' for each z
    for j = 1:M
        for i = 1:N
            if i == 1  % Forward difference at lower boundary
                dG_dk(i, j) = (modpar.ev(i+1, j) - modpar.ev(i, j)) / dk(i);
            elseif i == N  % Backward difference at upper boundary
                dG_dk(i, j) = (modpar.ev(i, j) - modpar.ev(i-1, j)) / dk(i-1);
            else  % Central difference
                forward_slope = (modpar.ev(i+1, j) - modpar.ev(i, j)) / dk(i);
                backward_slope = (modpar.ev(i, j) - modpar.ev(i-1, j)) / dk(i-1);
                dG_dk(i, j) = (forward_slope + backward_slope) / 2;
            end
        end
    end

    % Compute optimal consumption
    cons = (1-modpar.beta) ./dG_dk;

    % Compute market resources
    modpar.pol_y = cons + modpar.k_grid_j; 

    % Update value function
    modpar.vf = (1-modpar.beta)*(log(cons)-0.5*modpar.lss^2) + modpar.ev;

    % Adjust modpar.vf so that it is defined on grid modpar.y_grid_j
    for i = 1:M
        modpar.vf(:,i) = interp1(modpar.pol_y(:,i),modpar.vf(:,i),...
            modpar.y_grid_j(:,i),'linear','extrap');
    end

    % Update expected value function
    modpar.ev = modpar.vf*modpar.beta*modpar.Pi';

end

% This function performs endogenous grid algorithm when
% l is fixed at its steady state value and solves for
% value function modpar.vf and policy function modpar.pol_kp

function modpar = vfi_perf_s1(modpar, tol, maxit)

    % Default values
    if nargin < 2
        tol = 1e-8; 
    end

    if nargin < 3
        maxit = 10000;
    end

    % Start iterations
    it = 0;
    dist = 100;

    while dist > tol && it < maxit
        
        it = it + 1;
        V_enter = modpar.ev;

        % Update value function
        modpar = one_step_update_s1(modpar);

        % Check convergence
        dist = max(abs(V_enter - modpar.ev), [], 'all');

        if mod(it, 25) == 0
                fprintf('Finished iteration %d with dist of %f\n', it, dist);
        end

        if dist <= tol
            disp('Algorithm converged');
            
            % Obtain policy function for capital for each (k',z)
            options = optimset('Display','off','TolFun',tol,'TolX',tol);
            for j = 1:modpar.znum
                for i = 1:modpar.knum
                    modpar.pol_k(i,j) = fsolve(@(k) capital_resid(k, modpar.z_grid(j), ...
                        modpar.pol_y(i,j), modpar.lss, modpar.alpha, ...
                        modpar.delta), modpar.kss, options);
                end
            end

            % Obtain policy function for capital prime for each (k,z)
            for j = 1:modpar.znum
                
                % Get correspondence between k and kp for given z
                corresp = zeros(modpar.knum,2);
                corresp(:,1) = modpar.pol_k(:,j);
                corresp(:,2) = modpar.k_grid'; 
                corresp = sortrows(corresp); 

                % Get policy function for kp
                modpar.pol_kp(:,j) = interp1(corresp(:,1),corresp(:,2),...
                    modpar.k_grid,'linear','extrap');

            end

        end

    end

end

% This function performs initial update of the policy function for part 
% two of the algorithm when we account for endogenous labor. It updates
% modpar.vf and modpar.ev

function modpar = initial_update_s2(modpar)

    % Default values
    if nargin < 2
        tol = 1e-8; 
    end

    options = optimset('Display','off','TolFun',tol,'TolX',tol);

        V_enter = modpar.vf;
    
        for j = 1:modpar.znum
            
            zcur = modpar.z_grid(j);
    
            for i = 1:modpar.knum
    
                kcur = modpar.k_grid(i);
                
                cons = zeros(modpar.knum,1);
                labor = zeros(modpar.knum,1);
                curval = zeros(modpar.knum,1);
                futval = zeros(modpar.knum,1);
                objective = zeros(modpar.knum,1);
    
                % Iterate over possible values of k'
                for ip = 1:modpar.knum
    
                    kpcur = modpar.k_grid(ip);
                    labor(ip) = fsolve(@(l) labor_resid(l, kcur, kpcur, zcur, ...
                        modpar.alpha, modpar.delta), modpar.lss, options);
                    cons(ip) = (1-modpar.alpha)*exp(zcur)*kcur^modpar.alpha*...
                        labor(ip)^(-1-modpar.alpha);
                    curval(ip) = log(cons(ip)) - 0.5*labor(ip)^2;
                    futval(ip) = modpar.Pi(j,:)*modpar.vf(ip,:)';
                    objective(ip) = (1-modpar.beta)*curval(ip) + ...
                        modpar.beta*futval(ip);
                end
    
                % Find maximum of the objective
                [val, ind] = max(objective);
                modpar.vf(i,j) = val;
                modpar.pol_kp(i,j) = modpar.k_grid(ind);
                modpar.pol_l(i,j) = labor(ind);
                modpar.pol_c(i,j) = cons(ind);
    
            end
        end

        % Update expected value function
        modpar.ev = modpar.vf*modpar.beta*modpar.Pi';
    
        % Check convergence
        dist = max(abs(V_enter - modpar.vf), [], 'all');

        fprintf('Finished initial iteration with VF dist of %f\n', dist);    

end

% This function updates modpar.ev, modpar.vf, modpar.pol_y, modpar.pol_k 
% for second part of the algorithm when l is endogenous

function modpar = one_step_update_s2(modpar)

    % Default values
    if nargin < 2
        tol = 1e-8; 
    end
    options = optimset('Display','off','TolFun',tol,'TolX',tol);
    
    % This function updates modpar.ev, modpar.vf, modpar.pol_y

    % Compute values of derivative of the value function
    [N, M] = size(modpar.ev);    
    dk = diff(modpar.k_grid);  % Step sizes for k' grid (length N-1)
    
    % Preallocate matrix for derivative w.r.t. k'
    dG_dk = zeros(N, M);
    
    % Compute partial derivative w.r.t. k' for each z
    for j = 1:M
        for i = 1:N
            if i == 1  % Forward difference at lower boundary
                dG_dk(i, j) = (modpar.ev(i+1, j) - modpar.ev(i, j)) / dk(i);
            elseif i == N  % Backward difference at upper boundary
                dG_dk(i, j) = (modpar.ev(i, j) - modpar.ev(i-1, j)) / dk(i-1);
            else  % Central difference
                forward_slope = (modpar.ev(i+1, j) - modpar.ev(i, j)) / dk(i);
                backward_slope = (modpar.ev(i, j) - modpar.ev(i-1, j)) / dk(i-1);
                dG_dk(i, j) = (forward_slope + backward_slope) / 2;
            end
        end
    end

    % Compute optimal consumption
    cons = (1-modpar.beta) ./dG_dk;

    % Compute market resources
    modpar.pol_y = cons + modpar.k_grid_j; 
    
    % Solve for optimal current capital and labor for each (k',z)
    for j = 1:modpar.znum
        for i = 1:modpar.knum
            modpar.pol_k(i,j) = fsolve(@(k) capital_resid2(k, modpar.z_grid(j), ...
                modpar.pol_y(i,j), cons(i,j), modpar.alpha, modpar.delta), ...
                modpar.kss, options);
        end
    end
    labor = (((1-modpar.alpha)*exp(modpar.z_grid_j).*(modpar.pol_k.^...
        modpar.alpha))./cons).^(1/(1+modpar.alpha));

    % Update value function
    modpar.vf = (1-modpar.beta)*(log(cons)-0.5*labor.^2) + modpar.ev;

    % Adjust modpar.vf so that it is defined on grid for kp
    for i = 1:M
        modpar.vf(:,i) = interp1(modpar.pol_k(:,i),modpar.vf(:,i),...
            modpar.k_grid','linear','extrap');
    end

    % Update expected value function
    modpar.ev = modpar.vf*modpar.beta*modpar.Pi';

end

% This function performs second part of endogenous grid algorithm when
% l is endogenous and solves for value function modpar.vf and policy functions

function modpar = vfi_perf_s2(modpar)

    % Default values
    if nargin < 2
        tol = 1e-8; 
    end

    if nargin < 3
        maxit = 10000;
    end

    % Start iterations
    it = 0;
    dist = 100;

    while dist > tol && it < maxit
        
        it = it + 1;
        V_enter = modpar.vf;

        % Update value function
        modpar = one_step_update_s2(modpar);

        % Check convergence
        dist = max(abs(V_enter - modpar.vf), [], 'all');

        if mod(it, 25) == 0
                fprintf('Finished iteration %d with dist of %f\n', it, dist);
        end

        if dist <= tol
            disp('Algorithm converged');
            
            % Compute c(k(k',z),z)
            cons = modpar.pol_y - modpar.k_grid_j;

            % Solve for labor on the endogenous grid of k: l(k(k',z),z)
            labor = (((1-modpar.alpha)*exp(modpar.z_grid_j).*(modpar.pol_k.^...
                modpar.alpha))./cons).^(1/(1+modpar.alpha));

            % Obtain policy function for capital prime, labor and consumption for each (k,z)
            for j = 1:modpar.znum
                
                % Get correspondence between k, c and kp for given z
                corresp = zeros(modpar.knum,3);
                corresp(:,1) = modpar.pol_k(:,j);
                corresp(:,2) = modpar.k_grid'; 
                corresp(:,3) = cons(:,j); 
                corresp = sortrows(corresp); 

                % Get policy function for kp and consumption
                modpar.pol_kp(:,j) = interp1(corresp(:,1),corresp(:,2),...
                    modpar.k_grid,'linear','extrap');
                modpar.pol_c(:,j) = interp1(corresp(:,1),corresp(:,3),...
                    modpar.k_grid,'linear','extrap');

            end

            % Obtain policy function for labor
            modpar.pol_l = (((1-modpar.alpha)*exp(modpar.z_grid_j).*(modpar.k_grid_j.^...
                modpar.alpha))./modpar.pol_c).^(1/(1+modpar.alpha));

        end

    end

end

% This function computes errors of the Euler equation given solution 
% of the model

function err = Euler_err(modpar)

    err = ones(modpar.knum, modpar.znum)*999;

    for j = 1:modpar.znum
        for i = 1:modpar.knum
            lhs = 1/modpar.pol_c(i,j);
            kp = modpar.pol_kp(i,j);

            % RHS
            rhs = 0;
            for s = 1:modpar.znum
                lp = interp1(modpar.k_grid',modpar.pol_l(:,s),...
                    kp,'linear','extrap');
                cp = interp1(modpar.k_grid',modpar.pol_c(:,s),...
                    kp,'linear','extrap');
                futkret = modpar.alpha*kp^(modpar.alpha-1)*exp(modpar.z_grid(s))* ...
                    lp^(1-modpar.alpha) + 1 - modpar.delta;
                rhs = rhs + modpar.beta*modpar.Pi(j,s)*futkret/cp;
            end

            err(i,j) = lhs - rhs;
        end
    end

end

% This function simulates the model given a solution of the model
function mod_sim = simulate_model(model, T, dropT, one_time)

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
        sim_step = simulate_one_step(model, cursim);

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

% This function performs one step of the model's simulation

function sim_step = simulate_one_step(model, cursim)
    
    z_index = cursim.z_index;
    z = model.z_grid(z_index);

    l = interp1(model.k_grid',model.pol_l(:,z_index),...
                    cursim.k,'linear','extrap');
    y = exp(z)*cursim.k^model.alpha*l^(1-model.alpha);
    c = (1-model.alpha)*exp(z)*cursim.k^model.alpha*l^(-1-model.alpha);            
    kp = y+(1-model.delta)*cursim.k-c;

    % calculate residual  
    fkp  = zeros(model.znum,1);
    temp = zeros(model.znum,1);

    for zp_index = 1:model.znum

        lp = interp1(model.k_grid',model.pol_l(:,z_index),...
                     kp,'linear','extrap');
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

% Auxiliary functions 

function res = capital_resid(k, z, y, lss, aalpha, ddelta)

    res = y - exp(z)*k^aalpha*lss^(1-aalpha) - (1-ddelta)*k;

end

function res = labor_resid(l, k, kp, z, aalpha, ddelta)
    
    c = (1-aalpha)*exp(z)*k^aalpha*l^(-1-aalpha);
    y = exp(z)*k^aalpha*l^(1-aalpha);
    res = kp + c - y - (1-ddelta)*k;

end

function res = capital_resid2(k, z, y, c, aalpha, ddelta)
    
    l = (((1-aalpha)*exp(z)*k^aalpha)/c)^(1/(1+aalpha));
    res = y - (1-ddelta)*k - exp(z)*k^aalpha*l^(1-aalpha);

end