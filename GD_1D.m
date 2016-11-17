% Edward Rusu
% Math 6590
% Variational Image Processing
% Project 1
% 1D Signal - Gradient Descent

% This program will solve the Tikhanov optimization problem using gradient
% descent. We will just do the one-dimensional noisy signal here.

clearvars
clc


%% Load Data
[uxact u0] = LoadData; % Load the 1-dimensional signal
N = length(u0); % Number of nodes

maxIter = 100000; % Maximum number of iterations
tol = 1e-2; % Convergence tolerance
exitflag = 0; % Convergence condition
L = 2; % Norm to check convergence

lambda = 0.001; % Value for lambda
r = 1/5; % CFL conditions, 1/4 is max for stability

% Rather than using ghost points for the Neumann condition, we simply treat
% them as algebraic constraints, do some solving, and find the correct
% expression for our interior points


%% Create 1D Neumann-Laplacian Operator
e = ones(N,1);
Lapl = spdiags([e -2*e e],[-1 0 1],N,N); % Interior Operator
Lapl(1,2) = 2; % Left boundary
Lapl(N,N-1) = 2; % Right boundary


%% Gradient Descent
uN = u0; % Initial condition

figure(1), clf
for j = 1:maxIter
    % Iterate
    uO = uN;
    uN = uO + r*2*Lapl*uO - r*lambda*(uO-u0);
    
    % Animation
    if (mod(j,10) == 0) % Only occasionally plot
        plot(1:N,uxact,'r-',1:N,uN,'b.'), axis([1 1000 0 4.5])
        title(['Iteration: ' num2str(j)])
        drawnow
        pause(0.01)
    end
    
    % Check convergence
    if (norm(uN - uO,L) < tol)
        exitflag = 1; % Converged
        break;
    end
end

if (exitflag == 1)
    disp(['Converged in ' num2str(j) ' iterations.']);
else
    disp('Did not converge. Consider increasing max iteration or loosening tolerance.');
end





