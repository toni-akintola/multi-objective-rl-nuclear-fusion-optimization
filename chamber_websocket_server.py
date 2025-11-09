"""
WebSocket server for chamber visualization.
Streams tokamak simulation data to the Next.js chamber visualization page.
"""
import asyncio
import websockets
import json
import gymnasium as gym
import gymtorax
from agent import RandomAgent
import importlib.util
from pathlib import Path

# Import shape guard
spec = importlib.util.spec_from_file_location(
    "shape_guard",
    Path(__file__).parent / "optimization-for-constraints" / "shape_guard.py"
)
shape_guard = importlib.util.module_from_spec(spec)
spec.loader.exec_module(shape_guard)
shape_violation = shape_guard.shape_violation


async def chamber_simulation_server(websocket):
    """Run simulation and stream chamber data to connected client.
    
    Uses the same logic as visualize_chamber_live.py for consistency.
    """
    print("\n✅ Client connected! Starting chamber visualization...")
    
    # Setup environment and agent (same as visualize_chamber_live.py)
    print("🔧 Initializing environment...")
    env = gym.make("gymtorax/IterHybrid-v0")
    agent = RandomAgent(
        action_space=env.action_space,
        shape_penalty=0.1,
        damp_on_violation=True,
        damp_factor=0.5,
    )
    
    print("🔄 Resetting environment...")
    observation, info = env.reset()
    agent.reset_state(observation)
    print("✨ Ready! Streaming chamber data...\n")
    
    step_count = 0
    prev_severity = float('inf')
    
    try:
        while True:
            # Get action and step (same as visualize_chamber_live.py)
            action = agent.act(observation)
            observation, reward, terminated, truncated, info = env.step(action)
            reward = agent.apply_shape_safety(reward, observation)
            
            # Extract shape data (same logic as visualize_chamber_live.py)
            if agent.last_shape_info:
                shape_info = agent.last_shape_info
                beta_N = float(shape_info["shape"][0])
                q_min = float(shape_info["shape"][1])
                q95 = float(shape_info["shape"][2])
                violation_severity = float(shape_info["severity"])
                is_ok = shape_info["ok"]
                in_box = shape_info["in_box"]
                smooth = shape_info["smooth"]
                
                # Check if this is a corrective action (same logic as visualize_chamber_live.py)
                was_in_violation = (prev_severity != float('inf') and 
                                  not (is_ok and prev_severity == 0))
                is_corrective = False
                if was_in_violation and not is_ok:
                    is_corrective = violation_severity < prev_severity
                
                prev_severity = violation_severity
                
                # Send data to client in the format expected by the React component
                data = {
                    "type": "chamber_data",
                    "data": {
                        "beta_N": beta_N,
                        "q_min": q_min,
                        "q95": q95,
                        "ok": is_ok,
                        "violation": violation_severity,
                        "in_box": in_box,
                        "smooth": smooth,
                        "is_corrective": is_corrective,
                    }
                }
                
                await websocket.send(json.dumps(data))
                
                if step_count % 10 == 0:  # Print every 10 steps (same as visualize_chamber_live.py)
                    status = "🟢 SAFE" if is_ok else ("🟠 SELF-FIXING" if is_corrective else "🔴 VIOLATION")
                    print(f"{status} Step {step_count}: β_N={beta_N:.2f} | violation={violation_severity:.3f} | in_box={in_box} | smooth={smooth}")
            
            step_count += 1
            
            # Reset if episode ends (same as visualize_chamber_live.py)
            if terminated or truncated:
                observation, info = env.reset()
                agent.reset_state(observation)
                step_count = 0
                prev_severity = float('inf')
                print("🔄 Episode ended, resetting...")
            
            # Control update rate (approximately 10 FPS, matching visualize_chamber_live.py interval=100ms)
            await asyncio.sleep(0.1)
            
    except websockets.exceptions.ConnectionClosed:
        print("\n❌ Client disconnected")
    except Exception as e:
        print(f"\n❌ Error: {e}")
    finally:
        env.close()
        print("🔒 Environment closed")


async def main():
    """Start the WebSocket server."""
    print("=" * 60)
    print("Chamber Visualization WebSocket Server")
    print("=" * 60)
    print("Starting server on ws://localhost:8765")
    print("Connect from: http://localhost:3000/chamber or http://localhost:3001/chamber")
    print("=" * 60)
    
    # Allow connections from any origin (for development)
    # Listen on 127.0.0.1 (IPv4) to ensure browser compatibility
    async with websockets.serve(
        chamber_simulation_server, 
        "127.0.0.1", 
        8765,
        ping_interval=20,
        ping_timeout=10
    ):
        await asyncio.Future()  # run forever


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\n👋 Server stopped")

