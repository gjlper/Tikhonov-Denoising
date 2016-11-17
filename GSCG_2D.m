% Edward Rusu
% Math 6590
% Variational Image Processing
% Project 1
% 2D Signal - GS and CG

% This program will solve the Tikhanov optimization problem using
% Gauss-Seidel and Conjugate Gradient Methods. We will just do the
% two-dimensional noisy signal here.

clearvars
clc


%% Load Data
[~,~,uxact u0] = LoadData; % Load the 2-dimensional signal
N = length(u0); % Number of nodes in one dimension
u0 = u0(:); % Stack the columns

maxIter = 1000; % Maximum number of iterations
tol = 1e-3; % Convergence tolerance
exitflag = 0; % Convergence condition
L = 2; % Norm to check convergence

lambda = 0.25; % Value for lambda

% Rather than using ghost points for the Neumann condition, we simply treat
% them as algebraic constraints, do some solving, and find the correct
% expression for our interior points


%% Create 2D Elliptic Operator and RHS
% First create 1D
e = ones(N,1);
Elli = spdiags([e -2*e e],[-1 0 1],N,N); % 1D
Elli(1,2) = 2; Elli(N,N-1) = 2;

% Then kron for 2d
Elli2_ = kron(speye(N),Elli) + kron(Elli,speye(N));
Elli2 = -2*Elli2_ + lambda*speye(N^2);

rhs = lambda*u0;


%% Exact
% Before doing anything, we will perform a direct solve for comparison
udir = Elli2\rhs; % Direct solve with elimination

figure(1), clf
subplot(1,2,1)
pcolor(flipud(uxact)), colormap gray, axis tight, shading interp
subplot(1,2,2)
pcolor(flipud(reshape(udir,N,N))), colormap gray, axis tight, shading interp


%% Gauss-Seidel

% Gauss-Seidel involves the inversion of a lower-triangular matrix. For
% this case, the matrix is so large (almost 60,000) that this inversion is
% impractical. Instead, we will implement Gauss-Seidel with a for loop.

figure(2), clf
uN = rhs;
for n = 1:maxIter
    % Iterate
    uO = uN;
    for j = 1:N^2
        E_dot = Elli2*uN;
        uN(j) = 1./Elli2(j,j)*(rhs(j) - E_dot(j) + Elli2(j,j)*uN(j));
    end
    
    % Animate
    if (mod(n,1) == 0) % Only occasionally plot
        pcolor(flipud(reshape(uN,N,N))), axis tight, colormap gray, shading interp
        title(['GS iteration: ' num2str(n)]);
        drawnow
        pause(0.01)
    end
    
    % Check convergence
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

[uCG,exitflag,res,iterOut] = cgs(Elli2,rhs,tol,maxIter,[],[],u0);

if (exitflag == 0)
    disp(['CG converged in ' num2str(iterOut) ' iterations.']);
else
    disp('CG did not converge. Consider increasing max iteration or loosening tolerance.');
end

figure(3), clf
temp = reshape(uCG,N,N);
pcolor(flipud(reshape(uCG,N,N))), axis tight, colormap gray, shading interp
title('CG Solution');







