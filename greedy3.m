function total_average_return = greedyreturn(bandit_n, actions_n, timesteps, actual_distr[], noise_distr[])
% This function generates performance plots for n-bandit problems with the
% assigned number of bandits, number of timesteps, number of actions, 
% actual values and noise.

% Algorithm Implementation Steps
% Step 1: Set actual action-values Q* from nrand(0,1)
% Step 2: Sort action-values in ascending order
% Step 3: Initialize action-values adding Gaussian Noise
% Step 4: Start Greedy decision process
% Step 5: Choose action with the maximum estimated action-value
% Step 6: Update accumulated reward
% Step 7: Update estimated action-value
% Step 8: Store total average return for each bandit
% Step 9: Plot average total return against optimal average return


% CONSTANTS
% bandit_n = 20;
% actions_n = 5;
% timesteps = 1000;
% actual_distr = [0,1];
% noise_distr = [0,1];

% External variables
tot_avg_rew = zeros(1,bandit_n);

for bandit=1:bandit_n
    
    % Clean action-values every loop
    actual_q = zeros(1,actions_n);
    estimate_q = zeros(1,actions_n);
    
    bar_vals = zeros(1,timesteps);
    
    % 1. Generate actual values q*(A) and store them in q[] OK
    for i=1:actions_n
        actual_q(i) = normrnd(actual_distr(1),actual_distr(2));
    end
    
    % 2. Sort actual action-values in ascending order
    actual_q = sort(actual_q);
    
    % 3. Initiliaze action-values
    for i=1:actions_n
        gauss_noise = normrnd(noise_distr(1),noise_distr(2));
        estimate_q(i) = actual_q(i) + gauss_noise;
    end

    % 4. Greedy decision process
    % Clean values every loop
    imm_reward = 0;
    total_reward = 0;
    acc_reward = zeros(1, actions_n);
    action_counter = zeros(1,actions_n);
    
    for i=1:timesteps
        % 5. Choose option with max(estimate_q)
        [value, index] = max(estimate_q);
        gauss_noise = normrnd(noise_distr(1),noise_distr(2));
        imm_reward = actual_q(index) + gauss_noise;
        total_reward = total_reward + imm_reward;
        
        % 6. Update accumulated reward
        acc_reward(index) = acc_reward(index) + imm_reward;
        
        % 7. Update estimated value
        action_counter(index) = action_counter(index) + 1;
        estimate_q(index) = acc_reward(index) / action_counter(index); 
    end
    
    % 8. Store total return for each bandit
    tot_avg_rew(bandit) = total_reward / timesteps;
end

% 9. Plot total average return against optimal average return
total_average_return = sum(tot_avg_rew) / bandit_n;


