% Generate plot of timesteps against total average return
bandit_n = 100;
actions_n = 5;
timesteps = 100;
actual_distr = [0,1];
noise_distr = [0,1];

% Values array
values = zeros(1,timesteps+1);
steps = (0:1:timesteps);

for timestep=1:timesteps
    values(timestep+1) = greedyreturn(bandit_n,actions_n,timestep,actual_distr,noise_distr);
end

% Plot
figure('name','Greedy algorithm')
plot(steps,values,'r')
title('Greedy algorithm for 100-bandit problem')
xlabel('Timesteps')
ylabel('Total average return')

% Plot optimal average return line
hold on
line([0,timesteps],[1,1])

    
    