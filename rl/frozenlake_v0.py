import gym

env = gym.make('FrozenLake-v0')


num_episodes = 10
num_timesteps = 20

for i in range(num_episodes):
    
    state = env.reset()
    print('Time step: ', 0)
    env.render()
    
    for t in range(num_timesteps):
        random_action = env.action_space.sample()
        
        next_state, reward, done, info = env.step(random_action)
        print('Time step: ', t+1)
        
        env.render()
        
        if done:
            break