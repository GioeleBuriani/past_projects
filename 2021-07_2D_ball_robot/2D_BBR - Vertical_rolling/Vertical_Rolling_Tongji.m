m_1 = 5;        %Kg
m_2 = 20;       %Kg
r_1 = 0.2;      %m
r_2 = 0.5;      %m
mu_s = 0.65;
mu_d = 0.5;
mu_r = mu_s*10^-2;

J_1 = m_1*r_1^2/2;
J_2 = m_2*r_2^2/2;

resolution = 5e-3;
% Set the resolution for the Simout

theta_1_0 = pi/3;


%%
% Vertical rolling control

x_0 = [theta_1_0;0;0;0];

% Define state matrices
A = [0       1      0       0;
     2.616   -0.013 1.8686  0.0052;
     0       0      0       1;
     -1.2034 0.0052 -0.8595 -0.0021];
B = [0;
     2;
     0;
     -0.8];
C = [0   0 0.5    0;
     0.4 0 0.2857 0];
D = [0;
     0];

% Create state space object
sys = ss(A,B,C,D);

% Define transfer matrix
tm = zpk(tf(sys))


% Test initial response
initial(sys,x_0)

%
% Check open-loop eigenvalues
E = eig(A)
%
% Desired closed loop eigenvalues
P = [-133.35 -131.71 -1.19 -1.06];

% Solve for K using pole placement
K = place(A,B,P)

% Check for closed loop eigenvalues
Acl = A - B*K;
Ecl = eig(Acl)

% Create closed loop system
syscl = ss(Acl,B,C,D);

% Check initial response
initial(syscl,x_0)
grid on

%

% LQR control

% Define initial conditions

% Define Q and R matrices
Q = [1 0 0 0;
     0 1 0 0;
     0 0 1 0;
     0 0 0 1];
Qu = [1e3 0;
      0 1e3];
R = 1;

% Define matrix K
Klqr = lqr(sys,transpose(C)*Qu*C,R)
% Klqr1 = lqr(sys,transpose(C)*Qu1*C,R)

% Define new state space
lqrsys = ss((A - B*Klqr),B,C,D);
% lqrsys1 = ss((A - B*Klqr1),B,C,D);

% % Plot
initial(lqrsys,x_0)
grid on



%
% Transfer function of the inner loop

s = tf('s');

Cil = [0.4 0 0.2857 0];
Dil = [0];
sysil = ss((A-B*Klqr),B,Cil,Dil);
Gil = zpk(tf(sysil))
margin(Gil)

% Static regulator
Rs = 1/s;
margin(Rs*Gil)
mu = 20;
R = mu*Rs;
L = R*Gil;
margin(L)
grid on
zpk(R)

% Wanted crossing pulsation
omega_c = 6;

% Poles position
taup = 1/(10*omega_c);
Rd = 1/(s*(1+taup*s));
margin(Rd*Gil)

% Define tauz and the provisional regulator
rho = tand(30);
tauz = rho/omega_c;
Rd = (1+tauz*s)/(s*(1+taup*s));
bode(Rd*Gil)

% Define gain mu and the final regulator



%

s = tf('s');

A = [0       1      0        0;
     7848    -0.013 5.60571  0.0052;
     0       0      0        1;
     -3.1392 0.0052 -2.24229 -0.00208];
B = [0;
     2;
     0;
     -0.8];
C1 = [0.4 0 0.285714 0];
D1 = 0;

% Create transfer function
[b,a] = ss2tf(A,B,C1,D1);
G = tf(b,a)
Gzpk = zpk(G)
margin(G)


%%
% Script to create a graphic representation of the controlled system

theta = linspace(-pi,pi);
i = 1;
theta_1 = 0;
theta_2 = 0;

xc1 = r_1*cos(theta) - (r_1 + r_2)*sin(theta_1_0*(r_1/r_2));
yc1 = r_1*sin(theta) + (2*r_2+r_1) - (r_1 + r_2)*(1 - cos(theta_1_0*(r_1/r_2)));
xc2 = r_2*cos(theta);
yc2 = r_2 - r_2*sin(theta);

axis equal

hold on

circle_1 = plot(xc1,yc1,'b');
circle_1.LineWidth = 2;
line_1 = line([0 - (r_1 + r_2)*sin(theta_1_0*(r_1/r_2)),r_1*cos(theta_1) - (r_1 + r_2)*sin(theta_1_0*(r_1/r_2))],[(2*r_2+r_1),(2*r_2+r_1) - r_1*sin(theta_1)]);
line_1.Color = 'b';
line_1.LineWidth = 2;
% Draw first circle

circle_2 = plot(xc2,yc2,'r');
circle_2.LineWidth = 2;
line_2 = line([0,r_2*cos(theta_2)],[r_2,r_2 - r_2*sin(theta_2)]);
line_2.Color = 'r';
line_2.LineWidth = 2;
% Draw second circle

ground = line([-1.5,1.5],[0,0]);
ground.Color = 'g';
ground.LineWidth = 2;
% Draw ground

time_string = sprintf('t = %.2f',out.tout(1));
time_label = text(-0.75,1.3,time_string);
time_label.FontSize = 14;
%Draw time label

C_1_string = sprintf('C_1 = %.2f',out.u(1));
C_1_label = text(-0.75,1.15,C_1_string);
C_1_label.FontSize = 14;
%Draw C_1 label

Distance_string = sprintf('d = %.2f',out.d(1));
Distance_label = text(0.55,1.3,Distance_string);
Distance_label.FontSize = 14;
%Draw Distance label
    
Alpha_string = sprintf('alpha = %.2f',out.alpha(1));
Alpha_label = text(0.55,1.15,Alpha_string);
Alpha_label.FontSize = 14;
%Draw Distance label

hold off

k = length(out.tout)/length(out.theta1);

for i = 1:length(out.theta1)
    
    yc1m = r_1*sin(theta) + (2*r_2+r_1) - (r_1 + r_2)*(1 - cos(out.alpha(i)));
    set(circle_1,'YData',yc1m)
    % Vertical movement of circle 1
    
    xc2m = xc2 - out.theta2(i)*r_2;
    set(circle_2,'XData',xc2m)
    % Horizontal movement of circle 2
    
    line_1.XData = [0 - (r_1 + r_2)*sin(theta_1_0*(r_1/r_2)),r_1*cos(- out.theta1(i)) - (r_1 + r_2)*sin(theta_1_0*(r_1/r_2))];
    line_1.YData = [(2*r_2+r_1) - (r_1 + r_2)*(1 - cos(out.alpha(i))),(2*r_2+r_1) - r_1*sin(- out.theta1(i)) - (r_1 + r_2)*(1 - cos(out.alpha(i)))];
    line_2.XData = [0 - out.theta2(i)*r_2 , r_2*cos(- out.theta2(i)) - out.theta2(i)*r_2];
    line_2.YData = [r_2,r_2 - r_2*sin(- out.theta2(i))];
    % Modify line positions
    
    ground.XData = [-1.5 - out.d(i),1.5 - out.d(i)];
    
    time_string = sprintf('t = %.2f',out.tout(round(i*k)));
    time_label.String = time_string;
    time_label.Position = [-0.75 - out.d(i),1.3];
    % Modify time label
    
    C_1_string = sprintf('C_1 = %.2f',out.u(i));
    C_1_label.String = C_1_string;
    C_1_label.Position = [-0.75 - out.d(i),1.15];
    % Modify C_1 label
    
    Distance_string = sprintf('d = %.2f',out.d(i));
    Distance_label.String = Distance_string;
    Distance_label.Position = [0.55 - out.d(i),1.3];
    % Modify Reference label
    
    Alpha_string = sprintf('alpha = %.2f',out.alpha(i));
    Alpha_label.String = Alpha_string;
    Alpha_label.Position = [0.55 - out.d(i),1.15];
    % Modify Reference label
    
    
    drawnow % display updates
end