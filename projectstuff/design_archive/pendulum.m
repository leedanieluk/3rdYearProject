M = .5;
m = 1;
b = 0.1;
I = 0.006;
g = 9.8;
l = 0.3;

p = I*(M+m)+M*m*l^2; %denominator for the A and B matrices

A = [0      1              0           0;
     0 -(I+m*l^2)*b/p  (m^2*g*l^2)/p   0;
     0      0              0           1;
     0 -(m*l*b)/p       m*g*l*(M+m)/p  0];
B = [     0;
     (I+m*l^2)/p;
          0;
        m*l/p];

C = [1 0 0 0;
    0 1 0 0;
    0 0 1  0;
    0 0 0 1];
 
D = 0;

Q = [1 0 0 0;
    0 0.01 0 0;
    0 0 1 0;
    0 0 0 0.01];
R = 0.01;

Q = [0.5 0 0 0;
    0 0.1 0 0;
    0 0 0.8 0;
    0 0 0 0.05];
R = 0.2;

% We obtain Discrete-time dynamics
t_var = 1;
end_t = 6;
samp_step = 0.01;
samples = (end_t / samp_step)+1;
if t_var
A(1,2) = 8;A(3,1) = -6;A(3,3) = -13; 
A(3,4) = -13; A(4,1) = 1;
end
Kc = lqr(A,B,Q,R);

sysc = ss(A-B*Kc,B,C,D);
sysc2 = ss(A,B,C,D);

t1 = 0:samp_step:end_t;
r = ones(samples,1);
[yc,tc,xc] = lsim(sysc2,r,t1);
%plot(tc,yc(:,1),tc,yc(:,2),tc,yc(:,3),tc,yc(:,4));

sysd = c2d(sysc2,samp_step,'zoh');
[Pgoal,e,Kgoal] = dare(sysd.A,sysd.B,Q,R);


% Using for-loop
A = sysd.A;
B = sysd.B;
x = ones(4,samples);
xgoal = ones(4,samples);
y = zeros(4,samples);
ygoal = zeros(4,samples);
u = zeros(1,samples);
ugoal = zeros(1,samples);

K = [3 3 3 -1];
K = [2 1 2 -0.5];
Ktestplot(1,:) = K;
W = zeros(15,samples);
err = zeros(samples,1);
disc = 1;
pol_it = 201;
Kerr = zeros(pol_it,1);
init_error = 99999;

for j=1:pol_it
    P = 100*eye(15);
    for k=1:samples
        y(:,k) = C*x(:,k) + D*u(k);
        x(:,k+1) = A*x(:,k) + B*u(k);
        if rand < 0.3
            u(k+1) = -K*x(:,k+1) + rand - rand;
        else
            u(k+1) = -K*x(:,k+1);
        end
        
        % 1. compute quadratic basis
        mx1 = [x(:,k).' u(k)].' * [x(:,k).' u(k)];
        mx2 = [x(:,k+1).' u(k+1)].' * [x(:,k+1).' u(k+1)];
        
        % 2. Perform Update on W using RLS
        basis_k1 = [mx1(1,1) mx1(1,2) mx1(1,3) mx1(1,4) mx1(1,5) mx1(2,2) mx1(2,3) mx1(2,4) mx1(2,5) mx1(3,3) mx1(3,4) mx1(3,5) mx1(4,4) mx1(4,5) mx1(5,5)].';
        basis_k2 = [mx2(1,1) mx2(1,2) mx2(1,3) mx2(1,4) mx2(1,5) mx2(2,2) mx2(2,3) mx2(2,4) mx2(2,5) mx2(3,3) mx2(3,4) mx2(3,5) mx2(4,4) mx2(4,5) mx2(5,5)].';
        basis_vector = (basis_k1 - disc*basis_k2);
        reward = 0.5*(x(:,k).'*Q*x(:,k)+u(:,k).'*R*u(:,k));
        
        % RLS
        e = reward - basis_vector.' * W(:,k);
        W(:,k+1) = W(:,k) + ((P*basis_vector*e)/(1+basis_vector.'*P*basis_vector));
        P = P - ((P*basis_vector)*basis_vector.'*P)/(1+basis_vector.'*P*basis_vector);
        err(k) = immse(W(:,k), W(:,k+1));

        % Regression
%         errorr = basis_vector-reward;
%         W(:,k+1) = W(:,k) - 0.1*basis_vector*(W(:,k).'*basis_vector-reward);
%         err(k) = immse(W(:,k), W(:,k+1));
        
    end
     
    Sxx = [ W(1,k) W(2,k)/2 W(3,k)/2 W(4,k+1)/2; W(2,k)/2 W(6,k+1) W(7,k)/2 W(8,k)/2; W(3,k)/2 W(7,k)/2 W(10,k) W(11,k)/2; W(4,k)/2 W(8,k)/2 W(11,k)/2 W(13,k) ];
    Sxu = [ W(5,k)/2; W(9,k)/2; W(12,k)/2; W(14,k)/2 ];
    Sux = [ W(5,k)/2 W(9,k)/2 W(12,k)/2 W(14,k)/2 ];
    Suu = W(15,k);
    S = [Sxx Sxu; Sux Suu];
    K = (1/Suu)*Sux;
    Ktestplot(j+1,:) = (1/Suu)*Sux;
    Kerr(j,1) = immse(Kgoal,K);
    if Kerr(j,1) < init_error
        bestk = K;
    end
    
    W(:,1) = W(:,k+1);
    
end

for k=1:samples
    ygoal(:,k) = C*xgoal(:,k) + D*ugoal(k);
    ugoal(k) = -Kgoal*xgoal(:,k);
    xgoal(:,k+1) = A*xgoal(:,k) + B*ugoal(k);
end

for k=1:samples
    y(:,k) = C*x(:,k) + D*u(k);
    u(k) = -K*x(:,k);
    x(:,k+1) = A*x(:,k) + B*u(k);
end


t = 0:samp_step:end_t;
figure
subplot(2,1,1)
plot(t,y(1,:),t,y(2,:),t,y(3,:),t,y(4,:))
title('States')
xlabel('time') % x-axis label
ylabel('states(t)') % y-axis label
legend('x','dx','\phi','d\phi')
subplot(2,1,2)
u = u(:,1:601);
plot(t,u)
title('Actions')
xlabel('time') % x-axis label
ylabel('u(t)') % y-axis label

t = 0:samp_step:end_t;
figure
subplot(2,1,1)
plot(t,ygoal(1,:),t,ygoal(2,:),t,ygoal(3,:),t,ygoal(4,:))
title('Optimal States')
xlabel('time') % x-axis label
ylabel('states(t)') % y-axis label
legend('x','dx','\phi','d\phi')

subplot(2,1,2)
plot(t,ugoal)
title('Optimal Actions')
xlabel('time') % x-axis label
ylabel('u(t)') % y-axis label

figure
tk = 0:1:pol_it-1;
plot(tk, Kerr)
title('Mean Square Error')

figure
tk3 = 0:1:pol_it;
plot(tk3,Ktestplot(:,1),'b')
hold on
plot(tk3,Ktestplot(:,2),'g')
plot(tk3,Ktestplot(:,3),'r')
plot(tk3,Ktestplot(:,4),'m')
legend('K(1)','K(2)','K(3)','K(4)')
xlabel('Episodes') % x-axis label
ylabel('K') % y-axis label
k1op = Kgoal(1)*ones(pol_it+1);
k2op = Kgoal(2)*ones(pol_it+1);
k3op = Kgoal(3)*ones(pol_it+1);
k4op = Kgoal(4)*ones(pol_it+1);
plot(tk3, k1op,'b');
plot(tk3, k2op,'g');
plot(tk3, k3op,'r');
plot(tk3, k4op,'m');
axis([0 200 -1 6])
title('Policy K')
xlabel('Episodes') % x-axis label
ylabel('K') % y-axis label

K
Kgoal
bestk

% for k=1:samples
%     y(:,k) = C*x(:,k) + D*u(k);
%     x(:,k+1) = A*x(:,k) + B*u(k);
%     u(k+1) = -bestk*x(:,k+1) + rand - rand;
% end
% 
% plot(t,y(1,:),t,y(2,:),t,y(3,:),t,y(4,:))
% 
% title('BestK')