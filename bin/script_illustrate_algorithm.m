%% Description
% This script illustrates the algorithm on a simple example.
% Written by Gabriel Provencher Langlois

%% Setup
close all

m = 2;
n = 3;
A = [1,0,1;0,1,3];
b = [1;2];


%% Invoke the algorithm
tol_exact = 1e-10;
display_iterations = true;

[sol_exact_x,sol_exact_p,exact_path] = ...
    lasso_homotopy_solver(A,b,tol_exact,display_iterations);

%% Plt the 3D linfinity ball; plot the one changed by A
% Plot the cube
coor_Scope = [ -1, 1;
              -1, 1;
              -1, 1];
M=[coor_Scope(1,1),coor_Scope(2,2),coor_Scope(3,2);
   coor_Scope(1,1),coor_Scope(2,2),coor_Scope(3,1);
   coor_Scope(1,2),coor_Scope(2,2),coor_Scope(3,1);
   coor_Scope(1,2),coor_Scope(2,2),coor_Scope(3,2);
   coor_Scope(1,1),coor_Scope(2,1),coor_Scope(3,2);
   coor_Scope(1,1),coor_Scope(2,1),coor_Scope(3,1);
   coor_Scope(1,2),coor_Scope(2,1),coor_Scope(3,1);
   coor_Scope(1,2),coor_Scope(2,1),coor_Scope(3,2)];
d=[1 2 3 4 8 5 6 7 3 2 6 5 1 4 8 7];
X = M(d,3);
Y = M(d,1);
Z = M(d,2);
figure
plot3(M(d,3),M(d,1),M(d,2));
hold on

xlabel('X')
ylabel('Y')
zlabel('Z')

% Plot the vector A.'*p0
v0 = -A.'*sol_exact_p(:,1);
quiver3(0,0,0,v0(1),v0(2),v0(3))

v1 = -A.'*sol_exact_p(:,2);
quiver3(0,0,0,v1(1),v1(2),v1(3))

legend('3D Cube', '-At*p0', '-At*p1')
hold off

% Do some stuff...