from gym.envs.registration import register

register(
    id="Pong-v69",
    entry_point="gym_game.envs:PongEnv",
    max_episode_steps=2000,
)