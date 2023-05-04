import gym
env = gym.make('Tennis-v0')

# record the game
env = gym.wrappers.Monitor(env, 'tennis-1', force=True)

num_episodes = 100
num_time_steps = 50

# for i in range(num_episodes):
#     Return = 0
#     state = env.reset()
#     for t in range(num_time_steps):
#         env.render()
#         action = env.action_space.sample()
#         next_state, reward, done, info = env.step(action)
#         Return += reward
#         if done:
#             break
#     if i % 10 == 0:
#         print('Episode: ', i, 'Return: ', Return)

env.reset()

for _ in range(5000):
    env.render()
    _, _, done, _ = env.step(env.action_space.sample()) # take a random action
    
    if done:
        break

env.close()