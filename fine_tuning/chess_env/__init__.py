from gymnasium.envs.registration import register

register(
    id='ChessFetch-v0',
    entry_point='chess_env.chess_fetch_env:ChessFetchEnv',
    max_episode_steps=175,  # increased from 100 to allow for complex corner movements
)

register(
    id='ChessFetchDense-v0',
    entry_point='chess_env.chess_fetch_dense_env:ChessFetchDenseEnv',
    max_episode_steps=175,
)
