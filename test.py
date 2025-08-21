from connect_four_gymnasium import ConnectFourEnv
from connect_four_gymnasium.players import AdultSmarterPlayer
from stable_baselines3 import PPO

env = ConnectFourEnv(opponent=AdultSmarterPlayer())
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=1000000)
model.save("/home/alon/ros_ws/src/connect4/models/AdultSmarterPlayer")
