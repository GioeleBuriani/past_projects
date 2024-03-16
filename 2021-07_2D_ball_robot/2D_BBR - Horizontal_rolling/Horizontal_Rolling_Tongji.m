m_1 = 5;        %Kg
m_2 = 20;       %Kg
r_1 = 0.2;      %m
r_2 = 0.5;      %m
N = 50;          %N
mu_s = 0.65;
mu_d = 0.5;
mu_r = mu_s*10^-2;

J_1 = m_1*r_1^2/2;
J_2 = m_2*r_2^2/2;

resolution = 1e-2;
% Set the resolution for the Simout



%%
% Script to create a graphic representation of Horizontal_Rolling

theta = linspace(-pi,pi);
i = 1;
theta_1 = 0;
theta_2 = 0;

xc1 = r_1*cos(theta);
yc1 = (2*r_2+r_1) - r_1*sin(theta);
xc2 = r_2*cos(theta);
yc2 = r_2 - r_2*sin(theta);

axis equal

hold on

circle_1 = viscircles([0,(2*r_2+r_1)],r_1,'Color','b');
line_1 = line([0,r_1*cos(theta_1)],[(2*r_2+r_1),(2*r_2+r_1) - r_1*sin(theta_1)]);
line_1.Color = 'b';
line_1.LineWidth = 2;
% Draw first circle

circle_2 = viscircles([0,r_2],r_2,'Color','r');
line_2 = line([0,r_2*cos(theta_2)],[r_2,r_2 - r_2*sin(theta_2)]);
line_2.Color = 'r';
line_2.LineWidth = 2;
% Draw second circle

time_string = sprintf('t = %.2f',out.tout(1));
time_label = text(0.5,1.3,time_string);
%Draw time label

C_1_string = sprintf('C_1 = %.2f',out.C_1(1));
C_1_label = text(0.5,1.1,C_1_string);
%Draw C_1 label

C_2_string = sprintf('C_2 = %.2f',out.C_2(1));
C_2_label = text(0.5,1.0,C_2_string);
%Draw C_2 label


state_label = text(-0.8,1.3,'Pure rolling');
state_label.Color = 'g';
state_label.FontWeight = 'bold';
% Draw state label

hold off

k = length(out.tout)/length(out.theta1);

for i = 1:length(out.theta1)
    
    line_1.XData = [0,r_1*cos(- out.theta1(i))];
    line_1.YData = [(2*r_2+r_1),(2*r_2+r_1) - r_1*sin(- out.theta1(i))];
    line_2.XData = [0,r_2*cos(- out.theta2(i))];
    line_2.YData = [r_2,r_2 - r_2*sin(- out.theta2(i))];
    % Modify line positions
    
    time_string = sprintf('t = %.2f',out.tout(round(i*k)));
    time_label.String = time_string;
    % Modify time label
    
    C_1_string = sprintf('C_1 = %.2f',out.C_1(i));
    C_1_label.String = C_1_string;
    % Modify C_1 label
    
    C_2_string = sprintf('C_2 = %.2f',out.C_2(i));
    C_2_label.String = C_2_string;
    % Modify C_1 label
    
    if out.v_rel(i) < 0.001
        state_label.String = 'Pure rolling';
        state_label.Color = 'g';
    else
        state_label.String = 'Rolling with sliding';
        state_label.Color = 'r';
    end
    % Modify state label
    
    drawnow % display updates
end


%%
% Horizontal rolling control

s = tf('s');

% Define the 4 matrices
A = [0 1 0 0 ;
     0 -(mu_r*J_1*r_2^2)/(J_1*J_2*r_1^2+J_1^2*r_2^2) 0 (mu_r*J_1*r_1*r_2)/(J_1*J_2*r_1^2+J_1^2*r_2^2) ;
     0 0 0 1 ;
     0 (mu_r*J_2*r_1*r_2)/(J_2^2*r_1^2+J_1*J_2*r_2^2) 0 -(mu_r*J_2*r_1^2)/(J_2^2*r_1^2+J_1*J_2*r_2^2)];
B = [0 ;
     (J_1*r_2^2)/(J_1*J_2*r_1^2+J_1^2*r_2^2) ;
     0 ;
     -(J_2*r_1*r_2)/(J_2^2*r_1^2 + J_1*J_2*r_1^2)];
C = [0 0 r_2 0];
D = 0;

% Define eigenvalues of A
ea = eig(A)

% Define the plant transfer function G
[b,a] = ss2tf(A,B,C,D);
G = zpk(tf(b,a))

% Define the controller R(s)
R = -1;

% Define the open loop transfer function L(s)
L = R * G;
margin(L)
grid on

% Define the closed loop transfer function F(s)
F = feedback(L,1);
step(F)
grid on

% Define the wanted crossing pulsation
omega_c = 0.7;

% Define the constant for the physical realizability poles and the
% provisional regulator
taup = 1/(10*omega_c);
Rd = -1/(1+taup*s)^2;
margin(Rd*G)
% Need 75°+10.2° compensation

% Define tauz and the provisional regulator
rho = tand(85.2/2);
tauz = rho/omega_c;
Rd = -(1+tauz*s)^2/(1+taup*s)^2;
bode(Rd*G)

% Define gain mu and the final regulator
mu = 10^(1.61/20);
R = mu*Rd;
L = R*G;
margin(L)
grid on

% Plot the step response
F = feedback(L,1);
step(F)
grid on

% Design the prefilter Rpf(s)
taupf = 0.65/omega_c;
Rpf = 1/(1+taupf*s)^2;



% Define the controller R(s)
R1 = -1/s;

% Define the open loop transfer function L(s)
L1 = R1 * G;
margin(L1)

% Define the constant for the physical realizability poles and the
% provisional regulator
taup1 = 1/(10*omega_c);
Rd1 = -1/(s*(1+taup1*s)^3);
margin(Rd1*G)

% Define tauz and the provisional regulator
rho1 = tand(183/3);
tauz1 = rho1/omega_c;
Rd1 = -((1+tauz1*s)^3)/(s*(1+taup1*s)^3);
bode(Rd1*G)

% Define gain mu and the final regulator
mu1 = 10^(-18.4/20);
R1 = mu1*Rd1;
L1 = R1*G;
margin(L1)

% Plot the step response
F1 = feedback(L1,1);
step(F1)



omega_c2 = 5;

% Define the controller R(s)
R2 = -1/s;

% Define the open loop transfer function L(s)
L2 = R2 * G;
margin(L2)

% Define the constant for the physical realizability poles and the
% provisional regulator
taup2 = 1/(10*omega_c2);
Rd2 = -1/(s*(1+taup2*s)^3);
margin(Rd2*G)

% Define tauz and the provisional regulator
rho2 = tand(182/3);
tauz2 = rho2/omega_c2;
Rd2 = -((1+tauz2*s)^3)/(s*(1+taup2*s)^3);
bode(Rd2*G)

% Define gain mu and the final regulator
mu2 = 10^(33.3/20);
R2 = mu2*Rd2;
L2 = R2*G;
margin(L2)

% Plot the step response
F2 = feedback(L2,1);
step(F2)



% Anti-windup circuit
Gamma = (s+10)^4;
Ra = (-mu2*(1+tauz2*s)^3)/(Gamma);
Rb = (Gamma - (s*(1+taup2*s)^3))/(Gamma);



%%

% Plot the input variable u
plot(out.tout,out.u)
% axis([0 50 -10 10])
grid on

% Plot the system reponse y
hold on
plot(out.tout,out.reference,'r')
plot(out.tout,out.y,'b')
% axis([0 50 -2 12])
grid on
hold off



%%
% Graphic animation of controlled system 

% Script to create a graphic representation of Vertical_Rolling



theta = linspace(-pi,pi);
i = 1;
theta_1 = 0;
theta_2 = 0;

xc1 = r_1*cos(theta);
yc1 = (2*r_2+r_1) - r_1*sin(theta);
xc2 = r_2*cos(theta);
yc2 = r_2 - r_2*sin(theta);

axis equal

hold on

circle_1 = plot(xc1,yc1,'b');
circle_1.LineWidth = 2;
line_1 = line([0,r_1*cos(theta_1)],[(2*r_2+r_1),(2*r_2+r_1) - r_1*sin(theta_1)]);
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

Reference_string = sprintf('Ref = %.2f',out.reference(1));
Reference_label = text(0.55,1.3,Reference_string);
Reference_label.FontSize = 14;
%Draw Reference label

Distance_string = sprintf('d = %.2f',out.y(1));
Distance_label = text(0.55,1.15,Distance_string);
Distance_label.FontSize = 14;
%Draw Distance label

hold off

k = length(out.tout)/length(out.theta1);

for i = 1:length(out.theta1)
    
    xc1m = xc1 + out.y(i);
    set(circle_1,'XData',xc1m)
    % Horizontal movement of circle 2
    
    xc2m = xc2 + out.y(i);
    set(circle_2,'XData',xc2m)
    % Horizontal movement of circle 2
    
    line_1.XData = [0 + out.y(i) , r_1*cos(out.theta1(i)) + out.y(i)];
    line_1.YData = [(2*r_2+r_1) , (2*r_2+r_1) - r_1*sin(out.theta1(i))];
    line_2.XData = [0 + out.y(i) , r_2*cos(out.theta2(i)) + out.y(i)];
    line_2.YData = [r_2 , r_2 - r_2*sin(out.theta2(i))];
    % Modify line positions
    
    ground.XData = [-1.5 + out.y(i),1.5+ out.y(i)];
    
    time_string = sprintf('t = %.2f',out.tout(round(i*k)));
    time_label.String = time_string;
    time_label.Position = [-0.75 + out.y(i),1.3];
    % Modify time label
    
    C_1_string = sprintf('C_1 = %.2f',out.u(i));
    C_1_label.String = C_1_string;
    C_1_label.Position = [-0.75 + out.y(i),1.15];
    % Modify C_1 label
    
    Reference_string = sprintf('Ref = %.2f',out.reference(i));
    Reference_label.String = Reference_string;
    Reference_label.Position = [0.55 + out.y(i),1.3];
    % Modify Reference label
    
    Distance_string = sprintf('d = %.2f',out.y(i));
    Distance_label.String = Distance_string;
    Distance_label.Position = [0.55 + out.y(i),1.15];
    % Modify Reference label
    
    
    drawnow % display updates
end