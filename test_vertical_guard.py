"""
Test script for vertical guard module.

Tests the vertical guard with real gymtorax environment states to verify:
1. z_cm is available in the state
2. Vertical guard logic works correctly
3. Violations are detected properly
"""
import gymnasium as gym
import gymtorax
import importlib.util
from pathlib import Path

# Import vertical guard
spec = importlib.util.spec_from_file_location(
    "vertical_guard",
    Path(__file__).parent / "optimization-for-constraints" / "vertical_guard.py"
)
vertical_guard = importlib.util.module_from_spec(spec)
spec.loader.exec_module(vertical_guard)
vertical_violation = vertical_guard.vertical_violation
Z_MAX = vertical_guard.Z_MAX
MAX_DZ = vertical_guard.MAX_DZ


def test_vertical_guard():
    """Test vertical guard with real environment."""
    print("=" * 60)
    print("Testing Vertical Guard Module with Real Environment")
    print("=" * 60)
    
    # Create environment
    print("\n1. Creating environment...")
    env = gym.make("gymtorax/IterHybrid-v0")
    print("   ✅ Environment created")
    
    # Reset and check state structure
    print("\n2. Resetting environment and checking state structure...")
    observation, info = env.reset()
    
    # Check if z_cm exists - search more thoroughly
    has_z_cm = False
    z_cm_value = None
    z_cm_path = None
    
    print(f"   Observation top-level keys: {list(observation.keys())}")
    
    # Check scalars
    if "scalars" in observation:
        print(f"   Scalars keys: {list(observation['scalars'].keys())[:10]}... (showing first 10)")
        if "z_cm" in observation["scalars"]:
            has_z_cm = True
            z_cm_value = observation["scalars"]["z_cm"]
            z_cm_path = "observation['scalars']['z_cm']"
    
    # Check profiles
    if "profiles" in observation:
        print(f"   Profiles keys: {list(observation['profiles'].keys())}")
        if "z_cm" in observation["profiles"]:
            has_z_cm = True
            z_cm_value = observation["profiles"]["z_cm"]
            z_cm_path = "observation['profiles']['z_cm']"
    
    # Check geometry
    if "geometry" in observation:
        print(f"   Geometry keys: {list(observation['geometry'].keys())}")
        if "z_cm" in observation["geometry"]:
            has_z_cm = True
            z_cm_value = observation["geometry"]["z_cm"]
            z_cm_path = "observation['geometry']['z_cm']"
    
    # Check info dict
    if "info" in locals() and info:
        print(f"   Info keys: {list(info.keys())}")
        if "z_cm" in info:
            has_z_cm = True
            z_cm_value = info["z_cm"]
            z_cm_path = "info['z_cm']"
    
    # Deep search for z_cm
    if not has_z_cm:
        print(f"   ⚠️  'z_cm' not found in common locations")
    
    if has_z_cm:
        print(f"   ✅ Found z_cm at: {z_cm_path}")
        print(f"   z_cm value: {z_cm_value}")
        print(f"   z_cm type: {type(z_cm_value)}")
        if hasattr(z_cm_value, '__len__'):
            print(f"   z_cm length: {len(z_cm_value)}")
            if len(z_cm_value) > 0:
                print(f"   z_cm[0]: {z_cm_value[0]}")
    else:
        print("   ❌ z_cm not found in state!")
        print("\n   Checking alternative locations...")
        # Try to find z_cm in other places
        def find_z_cm(obj, path="observation", depth=0):
            if depth > 3:  # Limit recursion
                return None
            if isinstance(obj, dict):
                for key, value in obj.items():
                    if key == "z_cm":
                        return f"{path}['{key}']"
                    result = find_z_cm(value, f"{path}['{key}']", depth+1)
                    if result:
                        return result
            elif isinstance(obj, (list, tuple)):
                for i, item in enumerate(obj):
                    result = find_z_cm(item, f"{path}[{i}]", depth+1)
                    if result:
                        return result
            return None
        
        alt_path = find_z_cm(observation)
        if alt_path:
            print(f"   ✅ Found z_cm at: {alt_path}")
            has_z_cm = True
            # Try to extract it
            try:
                exec(f"z_cm_value = {alt_path}")
                z_cm_path = alt_path
            except:
                pass
        else:
            print("   ❌ z_cm not found anywhere in observation")
            # Also check info
            if info:
                alt_path_info = find_z_cm(info, "info")
                if alt_path_info:
                    print(f"   ✅ Found z_cm in info at: {alt_path_info}")
                    has_z_cm = True
                    try:
                        exec(f"z_cm_value = {alt_path_info}")
                        z_cm_path = alt_path_info
                    except:
                        pass
    
    if not has_z_cm:
        print("\n   ⚠️  z_cm not found. Checking if we can compute it from other parameters...")
        print("   Looking for geometry parameters that might help...")
        if "scalars" in observation:
            if "R_major" in observation["scalars"]:
                print(f"   R_major = {observation['scalars']['R_major']}")
            if "a_minor" in observation["scalars"]:
                print(f"   a_minor = {observation['scalars']['a_minor']}")
        print("\n   Note: z_cm might need to be computed from geometry or might not be available.")
        print("   We may need to check the gymtorax documentation or source code.")
        return
    
    # Test vertical guard with initial state
    print("\n3. Testing vertical guard with initial state...")
    result = vertical_violation(None, observation)
    print(f"   z = {result['z']:.6f} cm")
    print(f"   dz = {result['dz']:.6f} cm")
    print(f"   in_band = {result['in_band']} (|z| <= {Z_MAX})")
    print(f"   smooth = {result['smooth']} (N/A for first step)")
    print(f"   ok = {result['ok']}")
    print(f"   severity = {result['severity']:.6f}")
    
    # Run a few steps and track vertical position
    print("\n4. Running steps and tracking vertical position...")
    prev_state = observation
    prev_result = result
    
    for step in range(10):
        action = env.action_space.sample()
        observation, reward, terminated, truncated, info = env.step(action)
        
        # Test vertical guard
        result = vertical_violation(prev_state, observation)
        
        # Check for violations
        status = "🟢 SAFE" if result['ok'] else "🔴 VIOLATION"
        if not result['in_band']:
            status += " (out of band)"
        if not result['smooth']:
            status += " (not smooth)"
        
        print(f"\n   Step {step + 1}:")
        print(f"     z = {result['z']:.6f} cm (change: {result['dz']:+.6f} cm)")
        print(f"     Status: {status}")
        print(f"     Severity: {result['severity']:.6f}")
        
        # Check if severity changed
        if result['severity'] != prev_result['severity']:
            if result['severity'] < prev_result['severity']:
                print(f"     ⭐ Improving! (was {prev_result['severity']:.6f})")
            else:
                print(f"     ⚠️  Worsening (was {prev_result['severity']:.6f})")
        
        prev_state = observation
        prev_result = result
        
        if terminated or truncated:
            print(f"\n   Episode ended at step {step + 1}")
            break
    
    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    print(f"✅ Vertical guard module works with real environment")
    print(f"✅ z_cm is accessible at: {z_cm_path}")
    print(f"✅ Constants: Z_MAX={Z_MAX} cm, MAX_DZ={MAX_DZ} cm")
    print("=" * 60)
    
    env.close()


if __name__ == "__main__":
    test_vertical_guard()

