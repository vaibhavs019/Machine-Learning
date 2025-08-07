env = Maze()

initial_state = env.reset()
print(f"The new episode will start in state: {initial_state}")

frame = env.render(mode='rgb_array')
plt.axis('off')
plt.title(f"State: {initial_state}")
plt.imshow(frame)

env.close()

env = Maze()

env.reset()

env.observation_space

env.action_space

env.action_space.sample()

env = Maze()
state = env.reset()
episode = []
done = False
while not done:
    action = env.action_space.sample()
    next_state, reward, done, extra_info = env.step(action)
    episode.append([state, action, reward, done, next_state])
    state = next_state
env.close()    

episode

env = Maze()
state = env.reset()
done = False
gamma = 0.98
G_0 = 0
t = 0
while not done:
    action = env.action_space.sample()
    next_state, reward, done, extra_info = env.step(action)
    G_0 += gamma ** t * reward
    t += 1
env.close()

#No of iterations
t

#Total reward
G_0