% Edward Rusu
% Math 6590
% Variational Image Processing
% Project 1
% 2D Signal - Gradient Descent

% This program will solve the Tikhanov optimization problem using gradient
% descent. We will just do the two-dimensional noisy image here.

clearvars
clc


%% Load Data
[~,~,uxact u0_] = LoadData; % Load the 2-dimensional signal
N = length(u0_); % Number of nodes in a single dimension
u0 = u0_(:); % Reshape by stacking columns

maxIter = 5000; % Maximum number of iterations
tol = 1e-3; % Convergence tolerance
exitflag = 0; % Convergence condition
L = 2; % Norm to check convergence

lambda = 0.2; % Value for lambda
r = 1/9; % CFL conditions, 1/8 is max for stability

% Rather than using ghost points for the Neumann condition, we simply treat
% them as algebraic constraints, do some solving, and find the correct
% expression for our interior points


%% Create 2D Neumann-Laplacian Operator

% First create 1D operator
e = ones(N,1);
Lapl1 = spdiags([e -2*e e],[-1 0 1],N,N); % Interior Operator
Lapl1(1,2) = 2; 
Lapl1(N,N-1) = 2; % Right boundary

% Then kron into 2D operator
Lapl2 = kron(Lapl1,speye(N)) + kron(speye(N),Lapl1);


%% Gradient Descent
uN = u0; % Initial Condition

figure(1), clf
for j = 1:maxIter
    % Iterate
    uO = uN;
    uN = uO + r*2*Lapl2*uO - r*lambda*(uO - u0);
    
%     Animation
    if (mod(j,10) == 1) % Only occasionally plot
        pcolor(flipud(reshape(uN,N,N))), axis tight, colormap gray, shading interp
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







