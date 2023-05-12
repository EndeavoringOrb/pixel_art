import numpy as np

def my_update_q_network(q_network, state, action, reward, next_state, learning_rate, discount_factor):

    # Q(s, a) = Q(s, a) + α * (R + γ * max[Q(s', a')] - Q(s, a))
    # Q(s, a) = Q(s, a) + α * (immediate_reward + expected_change_in_reward)
    # you want the change because otherwise you would never want a negative but -1 > -2 so that is why it is difference. -1 is bad but -1 > -2.

    # Predict the Q-values for the current and next states
    current_q_values = q_network.predict(state)
    next_q_values = q_network.predict(next_state)

    # Calculate the target Q-value for the current state-action pair
    target_q_value = reward + discount_factor * np.max(next_q_values) - current_q_values[action]

    # Update the Q-value for the current state-action pair
    current_q_values[action] = current_q_values[action] + learning_rate * target_q_value

    # Update the weights of the neural network based on the updated Q-values
    q_network.fit(state, current_q_values, verbose=0)

    return q_network

def update_q_network(q_network, state, action, reward, next_state, learning_rate, discount_factor):

    # Q(s, a) = Q(s, a) + α * (R + γ * max[Q(s', a')] - Q(s, a))

    # Predict the Q-values for the current and next states
    current_q_values = q_network.predict(state)
    next_q_values = q_network.predict(next_state)

    # Calculate the target Q-value for the current state-action pair
    target_q_value = reward + discount_factor * np.max(next_q_values)

    # Update the Q-value for the current state-action pair
    current_q_values[0, action] = (1 - learning_rate) * current_q_values[0, action] + learning_rate * target_q_value

    # Update the weights of the neural network based on the updated Q-values
    q_network.fit(state, current_q_values, verbose=0)

    return q_network
