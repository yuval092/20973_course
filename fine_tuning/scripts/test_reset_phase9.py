import gymnasium as gym
import numpy as np
import chess_env

def test_scenario(scenario_name, num_tests=10):
    print(f"\n--- Testing Scenario: {scenario_name} ---")
    env = gym.make('ChessFetchDense-v0', force_scenario=scenario_name)
    
    for i in range(num_tests):
        try:
            obs, info = env.reset()
            env_unwrapped = env.unwrapped
            # Check for NaNs
            if np.isnan(obs['observation']).any():
                print(f"Test {i+1}: FAILED! NaN detected in observation.")
                return False

            if len(obs['observation']) != 26:
                print(f"Test {i+1}: FAILED! Observation length is {len(obs['observation'])}, expected 26.")
                return False

            scenario_id = obs['observation'][25]
            expected_ids = {"transit": 0.0, "descend": 0.5, "ascend": 1.0}
            if scenario_id != expected_ids[scenario_name]:
                print(f"Test {i+1}: FAILED! Expected scenario_id {expected_ids[scenario_name]}, got {scenario_id}.")
                return False

            start_pos = obs['observation'][:3]
            if scenario_name == 'ascend':
                # Arm starts at 0.415
                if abs(start_pos[2] - 0.415) >= 0.005:
                    print(f"Test {i+1}: FAILED! Ascend start Z {start_pos[2]:.4f} too far from 0.415.")
                    return False
            elif scenario_name == 'descend':
                if abs(start_pos[2] - env_unwrapped.SAFE_Z) >= 0.01:
                    print(f"Test {i+1}: FAILED! Descend start Z {start_pos[2]:.4f} too far from {env_unwrapped.SAFE_Z:.4f}.")
                    return False

            if scenario_name in ('descend', 'ascend'):
                drift = np.linalg.norm(start_pos[:2] - env_unwrapped.tube_center_xy)
                if drift >= env_unwrapped.current_drift_limit:
                    print(f"Test {i+1}: FAILED! Reset drift {drift:.4f} exceeds limit {env_unwrapped.current_drift_limit:.4f}.")
                    return False
            
            # Step once to make sure simulation doesn't explode
            obs, reward, terminated, truncated, info = env.step(np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32))
            
            if np.isnan(obs['observation']).any():
                print(f"Test {i+1}: FAILED! NaN detected after step.")
                return False
            
            print(f"Test {i+1}: PASSED. Start pos: {start_pos} | Scenario ID: {scenario_id} | Drift limit: {env_unwrapped.current_drift_limit:.4f}")
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"Test {i+1}: FAILED! Exception thrown: {e}")
            return False
            
    print(f"Scenario {scenario_name} passed {num_tests}/{num_tests} tests.")
    return True

if __name__ == "__main__":
    scenarios = ['transit', 'descend', 'ascend']
    all_passed = True
    for s in scenarios:
        if not test_scenario(s):
            all_passed = False
            
    if all_passed:
        print("\nAll reset teleportation tests PASSED! Physics are stable.")
    else:
        print("\nSome reset teleportation tests FAILED! Do not start training.")
        exit(1)
