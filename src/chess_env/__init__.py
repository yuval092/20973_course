from gymnasium.envs.registration import register

register(
    id="ChessFetchTask-v0",
    entry_point="src.chess_env.task:ChessTaskEnv",
    max_episode_steps=200,
)
