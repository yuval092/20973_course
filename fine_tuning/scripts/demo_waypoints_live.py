import time
import numpy as np
import gymnasium as gym
import chess_env
import mujoco.viewer
from scripts.coordinate_mapping import ChessCoordinateMapper
from stable_baselines3 import SAC

def live_waypoint_demo(start_sq='e2', end_sq='e4', model_path='models/sac-FetchPickAndPlace-v4.zip', speed=0.02):
    print(f"--- Live Waypoint Demo: {start_sq} -> {end_sq} ---")
    
    mapper = ChessCoordinateMapper()
    start_pos = mapper.notation_to_world(start_sq)
    end_pos = mapper.notation_to_world(end_sq)
    
    # Waypoint Z-Altitudes
    SAFE_Z, GRASP_Z, RELEASE_Z = 0.550, 0.425, 0.435
    
    env = gym.make('ChessFetch-v0')
    model = SAC.load(model_path, env=env)
    obs, _ = env.reset()
    unwrapped = env.unwrapped
    
    # Setup Piece
    obj_joint_id = unwrapped.model.joint("object0:joint").id
    qpos_start = unwrapped.model.jnt_qposadr[obj_joint_id]
    unwrapped.data.qpos[qpos_start : qpos_start + 3] = start_pos
    unwrapped.data.qpos[qpos_start + 3 : qpos_start + 7] = [1, 0, 0, 0]
    
    print("Launching MuJoCo Viewer...")
    with mujoco.viewer.launch_passive(unwrapped.model, unwrapped.data) as viewer:
        time.sleep(1.0) # Let viewer initialize
        
        def move_to(target_pos, gripper_action=0.0, name="Move"):
            nonlocal obs
            print(f"Executing Stage: {name}")
            unwrapped.goal = target_pos.copy()
            for _ in range(150): # Max steps per waypoint
                if not viewer.is_running(): break
                action, _ = model.predict(obs, deterministic=True)
                # Override gripper action
                full_action = action.copy()
                full_action[3] = gripper_action
                
                obs, _, _, _, _ = env.step(full_action)
                viewer.sync()
                time.sleep(speed)
                
                dist = np.linalg.norm(obs['achieved_goal'] - target_pos)
                if dist < 0.01:
                    break

        # --- THE WAYPOINT SEQUENCE ---
        
        # 1. Hover over SRC
        move_to(np.array([start_pos[0], start_pos[1], SAFE_Z]), gripper_action=1.0, name="1_Hover_SRC")
        
        # 2. Descend to SRC
        move_to(np.array([start_pos[0], start_pos[1], GRASP_Z]), gripper_action=1.0, name="2_Descend_SRC")
        
        # 3. Grasp
        print("Executing Stage: 3_Grasp")
        for _ in range(20):
            obs, _, _, _, _ = env.step(np.array([0, 0, 0, -1.0]))
            viewer.sync()
            time.sleep(speed)
        
        # 4. Lift
        move_to(np.array([start_pos[0], start_pos[1], SAFE_Z]), gripper_action=-1.0, name="4_Lift")
        
        # 5. Transit to DST
        move_to(np.array([end_pos[0], end_pos[1], SAFE_Z]), gripper_action=-1.0, name="5_Transit_DST")
        
        # 6. Deliver
        move_to(np.array([end_pos[0], end_pos[1], RELEASE_Z]), gripper_action=-1.0, name="6_Deliver")
        
        # 7. Release
        print("Executing Stage: 7_Release")
        for _ in range(20):
            obs, _, _, _, _ = env.step(np.array([0, 0, 0, 1.0]))
            viewer.sync()
            time.sleep(speed)
        
        # 8. Home
        move_to(np.array([0.700, 0.264, 0.550]), gripper_action=1.0, name="8_Home")
        
        print("Demo complete. Window will close in 3 seconds.")
        time.sleep(3.0)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--start', type=str, default='e2')
    parser.add_argument('--end', type=str, default='e4')
    parser.add_argument('--speed', type=float, default=0.01)
    args = parser.parse_args()
    
    live_waypoint_demo(start_sq=args.start, end_sq=args.end, speed=args.speed)
