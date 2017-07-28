import gym
from gym.envs.registration import register
from colorama import init

init(autoreset=True)

LEFT = 0
DOWN = 1
RIGHT = 2
UP = 3

arrow_keys = {
    'w':UP,
    's':DOWN,
    'a':LEFT,
    'd':RIGHT
}
def inkey():
    return input('wasd 중 하나의 문자를 입력하세요:')

register( id='FrozenLake-v3',
          entry_point='gym.envs.toy_text:FrozenLakeEnv',
          kwargs={'map_name':'4x4', 'is_slippery':False}
)

env = gym.make('FrozenLake-v3')
env.render()

while True:
    key = inkey()
    if key not in arrow_keys.keys():
        print("Invalid input! Game aborted!")
        break

    action = arrow_keys[key]
    state, reward, done, info = env.step(action)
    env.render()
    print("State:", state, "Action:", action, "Reward:", reward, "Info:", info)

    if done:
        print("Finished with reward:", reward)
        break