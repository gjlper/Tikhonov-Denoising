% Edward Rusu
% Math 6590
% Variational Image Processing
% Project 1
% 1D Signal - GS and CG

% This program will solve the Tikhanov optimization problem using
% Gauss-Seidel and Conjugate Gradient Methods. We will just do the
% one-dimensional noisy signal here.

clearvars
clc


%% Load Data
[uxact u0] = LoadData; % Load the 1-dimensional signal
N = length(u0); % Number of nodes

maxIter = 1000; % Maximum number of iterations
tol = 1e-6; % Convergence tolerance
exitflag = 0; % Convergence condition
L = 2; % Norm to check convergence

lambda = 0.025; % Value for lambda

% Rather than using ghost points for the Neumann condition, we simply treat
% them as algebraic constraints, do some solving, and find the correct
% expression for our interior points


%% Create 1D Elliptic Operator and RHS
e = ones(N,1);
Elli = lambda*speye(N) + spdiags([-2*e 4*e -2*e],[-1 0 1],N,N); % Set up Elliptic Operator
Elli(1,2) = -4; Elli(N,N-1) = -4;

rhs = lambda*u0;


%% Exact

% Before doing anything, we will perform a direct solve for comparison
udir = Elli\rhs; % Direct solve with tri-diagonal solver

figure(1), clf
plot(1:N, uxact,'r-','linewidth',2), hold on
plot(1:N, udir,'b.');
title('Exact Solution');


%% Gauss-Seidel
% Gauss-Seidel is x_{k+1} = (I-inv(P)*A)x_{k} + inv(P)*b

Elli_diag = spdiags(Elli); % Grab lower and main diagonal from Elli
PC = spdiags([Elli_diag(:,1) Elli_diag(:,2)], [-1 0],N,N); % Create GS preconditioner
PCinv = PC\speye(N); % Invert the preconditioner
rhsPC = PCinv*rhs; % Modify the rhs vector
GS = speye(N) - PCinv*Elli; % Gauss-Seidel Matrix = I-inv(P)*A

% Iteration here
uN = u0; % Initial condition

figure(2), clf
for n = 1:maxIter
    % Iterate
    uO = uN;
    uN = GS*uO + rhsPC;
    
    % Animate
    if (mod(n,5) == 0) % Only occasionally plot
        plot(1:N,uxact,'r-',1:N,uN,'b.'), axis([1 1000 0 4.5])
        title(['GS iteration: ' num2str(n)]);
        drawnow
        pause(0.01)
    end
    
    % Check Convergence
    if (norm(uN-uO,L) < tol)
        exitflag = 1;
        break;
    end
end

if (exitflag == 1)
    disp(['GS converged in ' num2str(n) ' iterations.']);
else
    disp('GS did not converge. Consider increasing max iteration or loosening tolerance.');
end


%% Conjugate Gradient

[uCG,exitflag,res,iterOut] = cgs(Elli,rhs,tol,maxIter,[],[],u0);

if (exitflag == 0)
    disp(['CG converged in ' num2str(iterOut) ' iterations.']);
else
    disp('CG did not converge. Consider increasing max iteration or loosening tolerance.');
end

figure(3), clf
plot(1:N, uxact,'r-','linewidth',2), hold on
plot(1:N, uCG,'b.');
title('CG solution');




