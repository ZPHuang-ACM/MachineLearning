% REINFORCEMENT LEARNING CONTROL OF TWO TANK LIQUID LEVEL SYSTEM

% Dr. Mathew Mithra Noel
% School of Electrical Engineering
% Vellore Institute of Technology

clear all;
clc;

% Define final desired goal state
global state_desired;
state_desired= 7;

% Flow to Tank 2 is not controlled and hence set to zero
global Q2;
Q2=0;

% Discretize state space
global h1;
global h2;
h1=linspace(0,10,15);
h2=h1;

global delta;
delta= (h1(2)-h1(1))/2;

% Discretize action space
global action;
Q1=linspace(0,20,10);

N1 = length(h1);
N2 = length(h2);

% Initialize policy and value.
pibest = zeros(N1,N2);

gamma =0.99;

% Set the initial guess for V(s) to be zero for each state s.
V = zeros(N1,N2);

policy = zeros(N1,N2);

% Compute the optimal value function using the Value Iteration algorithm.
for runs=1:1000
    
    for m=1:N1
        for n=1:N2
            for p =1:length(Q1)
                
                % Take all possible actions.
                action = Q1(p);
                
                snext = [h1(m); h2(n)]+ 0.1*tank(0,[h1(m); h2(n)]);
                
                % Compute the closest discretized state.
                [r,s] = closest(snext);
                
                nextV(p)=V(r,s);
            end
            
            [Vbest,bestind] = max(nextV);
            
            % Improve value function estimate using Bellman's equation.
            V(m,n)= Reward([h1(m); h2(n)] ) +  gamma*Vbest ;
        end
    end
    
end


% Compute the optimal policy from the optimal value function.
for m=1:N1
    for n=1:N2
        
        % Take all possible actions.
        for p =1:length(Q1)
            
            action = Q1(p);
            
            snext = [h1(m); h2(n)]+ 0.1*tank(0,[h1(m); h2(n)]);
            
            % Compute the closest discretized state.
            [r,s] = closest(snext);
            
            nextV(p)=V(r,s);
        end
        
        [Vbest,bestind] = max(nextV);
        
        pibest(m,n) = Q1(bestind);
    end
end


N = 100;
state=[1 0]; %Initial state
states = zeros(N,2);
states(1,:)= state ;
Ts = 0.1; % Define time between control actions.

% Simulate the system with the optimal control policy.
for n=2:N
    
    [r,s] = closest(state);
    
    % Use linear regression to interpolate between control actions for
    % discretized states
    
%  A Feedfroward Neural Network can also be used as in the following paper:
%     Control of a nonlinear liquid level system using a new artificial neural network based reinforcement learning approach,
% Applied Soft Computing, Volume 23, 2014, Pages 444-451, ISSN 1568-4946, https://doi.org/10.1016/j.asoc.2014.06.037.
% (http://www.sciencedirect.com/science/article/pii/S1568494614003111)

    if r > 1 && s > 1 && r < N1 && s < N2
        X = [h1(r) h2(s);h1(r-1) h2(s);h1(r+1) h2(s);h1(r) h2(s-1);h1(r) h2(s+1)];
        Y = [pibest(r,s) pibest(r-1,s) pibest(r+1,s) pibest(r,s-1) pibest(r,s+1)]';
        lin_model = fitlm(X,Y);
        action = predict(lin_model,state);
    else
        action = pibest(r,s);
    end
    
    %Simulate the system for one time step.
    [t,y]=ode45(@tank,[0 Ts],state);
    state = real(y(end,:));
    states(n,:) = state;
    
end

% Plot time history of states with optimal policy.
time = (1:length(states))*Ts;
plot(time,states);
xlabel('time (s)');
ylabel('state: liquid levels h_1 and h_2');
