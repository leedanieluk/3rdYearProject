function [total_average_return, optimal_action, optimal_return] = egreedy(bandit_n, actions_n, timesteps, actual_distr, noise_distr, e)
% EGREEDY Returns the total average return, optimal return and optimal action percentage for the 
% n-bandit problem with the given number of bandits, actions, timesteps, action-values and noise distributions
% and epsilon.
% EGREEDY(bandits, actions, timesteps, action_value_distribution[mean,
% variance], noise_distribution[mean,variance], epsilon)

% Algorithm Implementation Steps
% Step 1: Set actual action-values Q* from nrand(0,1)
% Step 2: Sort action-values in ascending or
% Step 3: Initialize action-values adding Gaussian Noise
% Step 4: Start e-Greedy decision process
% Step 5: Check if x is lower than epsilon
% Step 6: Choose action with the maximum estimated action-value
% Step 7: Update accumulated reward
% Step 8: Update estimated action-value
% Step 9: Store total average return for each bandit
% Step 10: Plot average total return against optimal average return

% DEBUG VARIABLES
% bandit_n = 20;
% actions_n = 5;
% timesteps = 1000;
% actual_distr = [0,1];
% noise_distr = [0,1];

% External variables
tot_avg_rew = zeros(1,bandit_n);
tot_opt_rew = zeros(1,bandit_n);
optimal_choice = 0;

for bandit=1:bandit_n
    
    % Clean action-values every loop
    actual_q = zeros(1,actions_n);
    estimate_q = zeros(1,actions_n);
    
    % 1. Generate actual values q*(A) and store them in q[] OK
    for i=1:actions_n
        actual_q(i) = normrnd(actual_distr(1),actual_distr(2));
    end
    
    % 2. Initiliaze action-values
    for i=1:actions_n
        gauss_noise = normrnd(noise_distr(1),noise_distr(2));
        estimate_q(i) = actual_q(i) + gauss_noise;
    end

    % 3. Greedy decision process
    % Clean values every loop
    total_reward = 0;
    acc_reward = zeros(1, actions_n);
    opt_reward = 0;
    action_counter = zeros(1,actions_n);
    
    for i=1:timesteps
        % 4. Choose option with max(estimate_q)
        [~, index] = max(estimate_q);
        
        % 5. If we fall under epsilon, choose random action
        x = rand;
        if x < e
            index = randi(actions_n);
        end
        
        [maxval,~] = max(actual_q);
        gauss_noise = normrnd(noise_distr(1),noise_distr(2));
        gauss_noise2 = normrnd(noise_distr(1),noise_distr(2));
        
        % Rewards update
        imm_reward = actual_q(index) + gauss_noise;
        total_reward = total_reward + imm_reward;
        opt_reward = opt_reward + maxval + gauss_noise2;
        
        % 6. Update accumulated reward
        acc_reward(index) = acc_reward(index) + imm_reward;
        
        % 7. Update estimated value
        action_counter(index) = action_counter(index) + 1;
        estimate_q(index) = acc_reward(index) / action_counter(index); 
    end
    
    % 8. Store total return for each bandit
    tot_avg_rew(bandit) = total_reward / timesteps;
    tot_opt_rew(bandit) = opt_reward / timesteps;
    
    % 9. Check it it converged to the optimal value
    [~, optimal_index] = max(action_counter);
    [~, actual_index] = max(actual_q);
    if optimal_index == actual_index
        optimal_choice = optimal_choice + 1;
    end
end

% 10. Plot total average return against optimal average return
total_average_return = sum(tot_avg_rew) / bandit_n;
optimal_action = optimal_choice / bandit_n;
optimal_return = sum(tot_opt_rew) / bandit_n;