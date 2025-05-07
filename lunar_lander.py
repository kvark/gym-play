from enum import Enum
import gymnasium as gym

class Action(Enum):
    NOTHING = 0
    LEFT_ENGINE = 1
    MAIN_ENGINE = 2
    RIGHT_ENGINE = 3

total_reward = 0.0
env = gym.make("LunarLander-v3", render_mode="human")
observation, info = env.reset()

episode_over = False
while not episode_over:
    linear = (observation[0], observation[1])
    linear_vel = (observation[2], observation[3])
    angular = observation[4]
    angular_vel = observation[5]

    time_horizon = 1
    future_linear = (linear[0] + linear_vel[0] * time_horizon, linear[1] + linear_vel[1] * time_horizon)
    future_angular = angular + angular_vel * time_horizon
    if linear[1] > 0.5:
        desired_fall_velocity = -0.3
    elif linear[1] > 0.1:
        desired_fall_velocity = -0.05
    else:
        desired_fall_velocity = 0

    scores = [0.0, 0.0, 0.0, 0.0]
    scores[Action.NOTHING.value] = 1
    scores[Action.LEFT_ENGINE.value] = max(future_linear[0], 0) ** 1.5 * 5 + max(-future_angular, 0) ** 1.5 * 10
    scores[Action.RIGHT_ENGINE.value] = max(-future_linear[0], 0) ** 1.5 * 5 + max(future_angular, 0) ** 1.5 * 10
    scores[Action.MAIN_ENGINE.value] = max(desired_fall_velocity - linear_vel[1], 0) ** 1.5 * 20
    action_id = 0
    for i in range(1,4):
        if scores[i] > scores[action_id]:
            action_id = i
    if False:
        print("---------")
        print("Observation:", observation)
        print("Future:", future_linear, future_angular)
        print("Part1",  max(future_linear[0], 0) ** 2.0 * 1.1,  max(-future_linear[0], 0) ** 2.0 * 1.1)
        print("Part2",  max(-future_angular, 0) ** 4 * 10,  max(future_angular, 0) ** 4 * 10)
        print("Actions", scores, "chosen", action_id)

    observation, reward, terminated, truncated, info = env.step(action_id)
    total_reward += float(reward)
    episode_over = terminated or truncated

env.close()
print("Total Reward:", total_reward)
