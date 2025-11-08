"""
WebSocket server to stream tokamak simulation data to the Three.js visualization.
Connects agent.py to index.html via WebSocket.
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


async def simulation_server(websocket, path):
    """Run simulation and stream data to connected client."""
    print("\n✅ Client connected! Starting simulation...")
    
    # Setup environment and agent
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
    print("✨ Ready! Streaming data...\n")
    
    step_count = 0
    
    try:
        while True:
            # Get action and step
            action = agent.act(observation)
            observation, reward, terminated, truncated, info = env.step(action)
            reward = agent.apply_shape_safety(reward, observation)
            
            # Extract shape data
            if agent.last_shape_info:
                shape_info = agent.last_shape_info
                beta_N = float(shape_info["shape"][0])
                q_min = float(shape_info["shape"][1])
                q95 = float(shape_info["shape"][2])
                violation_severity = float(shape_info["severity"])
                is_ok = shape_info["ok"]
                in_box = shape_info["in_box"]
                smooth = shape_info["smooth"]
                
                # Send data to client
                data = {
                    "beta_N": beta_N,
                    "q_min": q_min,
                    "q95": q95,
                    "shape_violation": violation_severity,
                    "ok": is_ok,
                    "in_box": in_box,
                    "smooth": smooth,
                    "step": step_count,
                }
                
                await websocket.send(json.dumps(data))
                if step_count % 10 == 0:  # Print every 10 steps to reduce spam
                    status_emoji = "🟢" if is_ok else ("🟠" if violation_severity < 0.5 else "🔴")
                    print(f"{status_emoji} Step {step_count}: β_N={beta_N:.3f}, q_min={q_min:.3f}, q95={q95:.3f}, violation={violation_severity:.3f}, ok={is_ok}")
            
            step_count += 1
            
            # Reset if episode ends
            if terminated or truncated:
                observation, info = env.reset()
                agent.reset_state(observation)
                step_count = 0
            
            # Control update rate (30 FPS)
            await asyncio.sleep(1.0 / 30.0)
            
    except websockets.exceptions.ConnectionClosed:
        print("\n👋 Client disconnected")
    except Exception as e:
        print(f"\n❌ Error during simulation: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("🧹 Cleaning up...")
        env.close()
        print("✅ Done")


async def main():
    """Start WebSocket server."""
    print("=" * 60)
    print("🚀 Starting WebSocket server...")
    print("=" * 60)
    print("📍 Server: ws://localhost:8765")
    print("🌐 Open index.html in your browser")
    print("   (or visit http://localhost:8000/index.html)")
    print("=" * 60)
    print("⏳ Waiting for client connection...")
    print("   (Press Ctrl+C to stop)\n")
    
    async with websockets.serve(simulation_server, "localhost", 8765):
        await asyncio.Future()  # run forever


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\n🛑 Server stopped by user")
    except Exception as e:
        print(f"\n❌ Error starting server: {e}")
        import traceback
        traceback.print_exc()

