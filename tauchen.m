function [Pi, yy] = tauchen(N, rho, sigma, mu, n_std)

    % Tauchen method to discretize an AR(1) process
    %
    % Inputs:
    % N - Number of grid points
    % rho - Autoregressive parameter
    % sigma - Standard deviation of shocks
    % mu - Mean of the process (default = 0)
    % n_std - Number of standard deviations for grid boundaries (default = 3)
    %
    % Outputs:
    % Pi - Transition matrix (NxN)
    % yy - Discretized state values (1xN)

    if nargin < 4
        mu = 0; % Default mean
    end
    if nargin < 5
        n_std = 3; % Default number of standard deviations
    end

    % Compute grid boundaries
    a_bar = n_std * sqrt(sigma^2 / (1 - rho^2));
    y = linspace(-a_bar, a_bar, N); % Discretized grid
    d = y(2) - y(1); % Distance between grid points

    % Initialize transition matrix
    Pi = zeros(N, N);

    % Transition probabilities
    for row = 1:N
        % Endpoints
        Pi(row, 1) = normcdf((y(1) - rho * y(row) + d / 2) / sigma);
        Pi(row, N) = 1 - normcdf((y(N) - rho * y(row) - d / 2) / sigma);

        % Middle columns
        for col = 2:(N-1)
            Pi(row, col) = normcdf((y(col) - rho * y(row) + d / 2) / sigma) - ...
                           normcdf((y(col) - rho * y(row) - d / 2) / sigma);
        end
    end

    % Shift the grid to center it around the mean
    yy = y + mu / (1 - rho);

    % Renormalize rows to sum to 1
    Pi = Pi ./ sum(Pi, 2);

end



