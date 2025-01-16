% This code solves the model using a neural network.

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
model.knum = 25; % Number of nodes per layer for capital
model.nlayers = 3; % Number of layers for neural network
model.kgrid_cover = 0.25; % Percentage coverage for capital grid
model.knum_fin = 100; % Number of points for capital grid

% Parameters of the model
model.beta = 0.97;
model.delta = 0.1;
model.alpha = 0.33;
model.rrho = 0.95;
model.sigma = 0.007;

% Simulation
T         = 10000; % Number of periods for the simulation of the economy
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
% 3. Neural network
%----------------------------------------------------------------

model.k_min = kss*(1-model.kgrid_cover);
model.k_max = kss*(1+model.kgrid_cover);

% Partition of space for k
model.k_grid = linspace(model.k_min, model.k_max, model.knum_fin);
model.k_grid_complete = model.k_grid;

% Joint grid
[model.k_grid_j, model.z_grid_j] = meshgrid(model.k_grid, model.z_grid);
model.k_grid_j = model.k_grid_j';
model.z_grid_j = model.z_grid_j';

% Initialize policy function for labor
%model.pol_l = ones(model.knum_fin,model.znum)*lss;
model.pol_l = ones(model.knum_fin,model.znum)*lss;

% Define neural network structure
layers = [
    featureInputLayer(1, 'Name', 'input')
    fullyConnectedLayer(25, 'Name', 'fc1')
    reluLayer('Name', 'relu1')
    fullyConnectedLayer(25, 'Name', 'fc2')
    reluLayer('Name', 'relu2')
    fullyConnectedLayer(25, 'Name', 'fc3')
    reluLayer('Name', 'relu3')
    fullyConnectedLayer(3, 'Name', 'output')
    expLayer('expOutput')];

% Convert to dlnetwork
net = dlnetwork(layerGraph(layers));

% Prepare data for neural network
X = model.k_grid';
Y = model.pol_l;


% Train the network
% Training settings
preTrainEpochs = 10000; % Number of epochs for pre-training
preTrainLearningRate = 1e-3; % Learning rate for pre-training
numEpochs = 1500; % Number of epochs for main training
learningRate = 1e-3; % Learning rate for main training
miniBatchSize = 100;
clipValue = 1.0; % Maximum gradient value
decayRate = 0.96;

% Adam optimizer parameters
beta1 = 0.9; % Exponential decay rate for the 1st moment
beta2 = 0.999; % Exponential decay rate for the 2nd moment
epsilon = 1e-8; % Small value to prevent division by zero

% Initialize moments
m = dlupdate(@(v) zeros(size(v), 'like', v), net.Learnables); % 1st moment
v = dlupdate(@(v) zeros(size(v), 'like', v), net.Learnables); % 2nd moment


% Prepare data
dlX = dlarray(X', 'CB'); % Transpose X to match 'CB' format
dlY = dlarray(Y', 'CB'); % Transpose Y 

rng(123);
% Pre-training loop
for epoch = 1:preTrainEpochs
    total_loss = 0;

    for i = 1:miniBatchSize:size(X, 1)
        idx = i:min(i + miniBatchSize - 1, size(X, 1));
        XBatch = dlX(:, idx); % Select mini-batch
        YBatch = dlY(:, idx);

        % Compute loss and gradients
        [loss, gradients] = dlfeval(@preTrainLoss, net, XBatch, YBatch);
        total_loss = total_loss + loss;

        % Adam optimizer update
        t = (epoch - 1) * ceil(size(X, 1) / miniBatchSize) + ceil(i / miniBatchSize); % Iteration count
        m = dlupdate(@(m, g) beta1 * m + (1 - beta1) * g, m, gradients);
        v = dlupdate(@(v, g) beta2 * v + (1 - beta2) * (g.^2), v, gradients);

        % Bias correction
        mHat = dlupdate(@(m) m / (1 - beta1^t), m);
        vHat = dlupdate(@(v) v / (1 - beta2^t), v);

        net = dlupdate(@(v, mH, vH) v - preTrainLearningRate * mH ./ (sqrt(vH) + epsilon), net, mHat, vHat);
    end

    % Display progress
    fprintf('Pre-Train Epoch %d, Loss: %.4f\n', epoch, total_loss);

    if total_loss < 1e-8
        fprintf('Early stopping: Loss < 1e-8 at Pre-Train Epoch %d\n', epoch);
        break;
    end
end

% Initialize variables for tracking the best network
constNet = net;
bestNet = net; % Initialize the best network with the starting network
bestLoss = Inf; % Set the initial lowest error to infinity

[loss, gradients] = dlfeval(@modelLoss, net, XBatch, YBatch, model);

% Main training loop
% Continue training with obtained function
rng(456);
for epoch = 1:numEpochs
    total_loss = 0;

    for i = 1:miniBatchSize:size(X, 1)
        idx = i:min(i + miniBatchSize - 1, size(X, 1));
        XBatch = dlX(:, idx); % Select mini-batch
        YBatch = dlY(:, idx);

        % Compute loss and gradients
        [loss, gradients] = dlfeval(@modelLoss, net, XBatch, YBatch, model);
        total_loss = total_loss + loss;
        gradients = dlupdate(@(g) min(max(g, -clipValue), clipValue), gradients);


        % Adam optimizer update
        t = (epoch - 1) * ceil(size(X, 1) / miniBatchSize) + ceil(i / miniBatchSize); % Iteration count
        m = dlupdate(@(m, g) beta1 * m + (1 - beta1) * g, m, gradients);
        v = dlupdate(@(v, g) beta2 * v + (1 - beta2) * (g.^2), v, gradients);

        % Bias correction
        mHat = dlupdate(@(m) m / (1 - beta1^t), m);
        vHat = dlupdate(@(v) v / (1 - beta2^t), v);

        % Update parameters
        net = dlupdate(@(v, mH, vH) v - learningRate * mH ./ (sqrt(vH) + epsilon), net, mHat, vHat);

    end

    if isnan(total_loss)
        net = bestNet;

        for i = 1:miniBatchSize:size(X, 1)
            idx = i:min(i + miniBatchSize - 1, size(X, 1));
            XBatch = dlX(:, idx); % Select mini-batch
            YBatch = dlY(:, idx);

            % Compute loss and gradients
            [loss, gradients] = dlfeval(@modelLoss, net, XBatch, YBatch, model);
            total_loss = total_loss + loss;
            gradients = gradients/2;


            % Adam optimizer update
            t = (epoch - 1) * ceil(size(X, 1) / miniBatchSize) + ceil(i / miniBatchSize); % Iteration count
            m = dlupdate(@(m, g) beta1 * m + (1 - beta1) * g, m, gradients);
            v = dlupdate(@(v, g) beta2 * v + (1 - beta2) * (g.^2), v, gradients);

            % Bias correction
            mHat = dlupdate(@(m) m / (1 - beta1^t), m);
            vHat = dlupdate(@(v) v / (1 - beta2^t), v);

            % Update parameters
            net = dlupdate(@(v, mH, vH) v - learningRate * mH ./ (sqrt(vH) + epsilon), net, mHat, vHat);

        end

    end

    % Save the best network every 10 epochs
    if mod(epoch, 10) == 0
        if total_loss < bestLoss
            fprintf('New best model at epoch %d, Loss: %.4f\n', epoch, total_loss);
            bestNet = net; % Save the best network
            bestLoss = total_loss; % Update the lowest error
        end
    end

    % Return to the best network every 100 epochs
    if mod(epoch, 100) == 0
        if total_loss > bestLoss
            net = bestNet; % Load the best network
            fprintf('Optimization resumed from the best model with Loss: %.4f\n', bestLoss);
        end
    end

    % Display progress
    fprintf('Epoch %d, Loss: %.4f\n', epoch, total_loss);

    if total_loss < 1e-8
        bestNet = net; % Save the best network
        bestLoss = total_loss; % Update the lowest error
        fprintf('Early stopping: Loss < 1e-8 at Epoch %d\n', epoch);
        break;
    end
end

net = bestNet;
total_loss = bestLoss;
save('Q4_net.mat', 'net');

% Update policy function using the trained network
dlX = dlarray(X', 'CB'); % Create dlarray for inputs
new_policy_l = predict(net, dlX);  % Predicted labor
new_policy_l = extractdata(new_policy_l);
% Reshape into a matrix
model.pol_l = reshape(new_policy_l', [model.knum_fin, model.znum]);
model.pol_c = (1 - model.alpha) * exp(model.z_grid_j) .* ...
    model.k_grid_j.^model.alpha .* model.pol_l.^(-1 - model.alpha);
model.pol_kp = exp(model.z_grid_j) .* model.k_grid_j.^model.alpha .* ...
    model.pol_l.^(1 - model.alpha) + (1 - model.delta) * model.k_grid_j ...
    - model.pol_c;

% Simulate the model to get distributions
mod_sim = simulate_model(net, model, T, dropT, false);

mean_lerror    = sum(mod_sim.leeSeries)/(T-dropT);
max_lerror_sim = max(mod_sim.leeSeries);

disp(' ')
disp('Integral of Euler Equation Error:')
disp(mean_lerror)
disp('Max Euler Equation Error Simulation:')
disp(max_lerror_sim)

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
model.euler_lerror = EulerErrors(model, net);
figure(2)
plot(model.k_grid_complete,model.euler_lerror)
title('Log10 Euler Error')

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

function [loss, gradients] = preTrainLoss(net, X, YTrue)
    % Forward pass
    YPred = predict(net, X);

    % Compute mean squared error loss
    loss = sum(sqrt((YPred - YTrue).^2), 'all');

    % Compute gradients
    gradients = dlgradient(loss, net.Learnables);
end

% Define the Custom Loss Function
function [loss, gradients] = modelLoss(net, X, YTrue, model)
    
    % Extract explanatory variables
    k = repmat(X,3,1);
    z = repmat(model.z_grid',1,size(X,2));
    z = dlarray(z, 'CB');
    
    % Forward pass for labor prediction
    l = predict(net, X);
    
    % Compute consumption and next-period capital
    c = (1 - model.alpha) * exp(z) .* k.^model.alpha .* l.^(-1 - model.alpha);
    y = exp(z) .* k.^model.alpha .* l.^(1 - model.alpha);
    kp = y + (1 - model.delta) * k - c;
    
    % Initialize loss
    totalLoss = 0;
    
    % Compute errors of the Euler equation
    for i = 1:size(l,2) % Loop over the grid of (z,k)
        for j = 1:size(l,1)

            zcur_ind = j;
            kp_cur = repmat(kp(j,i),size(l,1),1);
            lp = predict(net, kp(j,i));
            cp = (1 - model.alpha)*exp(model.z_grid').*(kp_cur.^model.alpha).*lp.^(-1 - model.alpha);
            
            aux = model.Pi(zcur_ind,:)'.*(model.alpha*exp(model.z_grid').*kp_cur.^(model.alpha-1).*...
                    lp.^(1-model.alpha)+(1-model.delta))./cp;
            RHS = model.beta*sum(aux);
    
            % Add to loss (example loss term)
            totalLoss = totalLoss + sqrt((1/c(j,i) - RHS)^2);

        end
    end
    
    % Compute gradients
    loss = totalLoss;
    gradients = dlgradient(loss, net.Learnables);
    
end

% Compute errors of Euler equation on the grid
function euler_lerror = EulerErrors(model, net)

    % Data
    X = model.k_grid_complete;
    dlX = dlarray(X, 'CB'); % Transpose X to match 'CB' format

    % Extract explanatory variables
    k = repmat(X,3,1);
    z = repmat(model.z_grid',1,size(X,2));
    z = dlarray(z, 'CB');

    l = predict(net, dlX); 

    % Compute consumption and next-period capital
    c = (1 - model.alpha) * exp(z) .* k.^model.alpha .* l.^(-1 - model.alpha);
    y = exp(z) .* k.^model.alpha .* l.^(1 - model.alpha);
    kp = y + (1 - model.delta) * k - c;

    % Store errors
    err = zeros(size(l));

    % Compute errors of the Euler equation
    for i = 1:size(l,2) % Loop over the grid of (z,k)
        for j = 1:size(l,1)

            zcur_ind = j;
            kp_cur = repmat(kp(j,i),size(l,1),1);
            lp = predict(net, kp(j,i));
            cp = (1 - model.alpha)*exp(model.z_grid').*(kp_cur.^model.alpha).*lp.^(-1 - model.alpha);
            
            aux = model.Pi(zcur_ind,:)'.*(model.alpha*exp(model.z_grid').*kp_cur.^(model.alpha-1).*...
                    lp.^(1-model.alpha)+(1-model.delta))./cp;
            RHS = model.beta*sum(aux);
    
            % Add to loss (example loss term)
            err(j,i) = 1/c(j,i) - RHS;

        end
    end

    euler_lerror = log10(abs(err))';
    
end

% This function simulates the model
function mod_sim = simulate_model(net, model, T, dropT, one_time)

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
        sim_step = simulate_one_step(net, model, cursim);

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

% This function computes one step of the simulation

function sim_step = simulate_one_step(net, model, cursim)
    
    z_index = cursim.z_index;
    z = model.z_grid(z_index);

    l = predict(net, cursim.k);
    l = l(z_index);

    y = exp(z)*cursim.k^model.alpha*l^(1-model.alpha);
    c = (1-model.alpha)*exp(z)*cursim.k^model.alpha*l^(-1-model.alpha);            
    kp = y+(1-model.delta)*cursim.k-c;

    lp_vec = predict(net, kp);

    % calculate residual  
    fkp  = zeros(model.znum,1);
    temp = zeros(model.znum,1);

    for zp_index = 1:model.znum

        lp = lp_vec(zp_index);

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