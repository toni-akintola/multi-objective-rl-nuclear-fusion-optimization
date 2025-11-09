"""
FastAPI server for running SAC model timesteps and saving environment state.
"""
import os
import json
import logging
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any
import gymnasium as gym
import gymtorax  # Registers gymtorax envs
from gymnasium.spaces import Dict as SpaceDict, Tuple as SpaceTuple, Box
from gymnasium.spaces.utils import flatten_space, flatten, unflatten
from stable_baselines3 import SAC
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response, StreamingResponse
from pydantic import BaseModel
import torch.nn as nn
from starlette.middleware.base import BaseHTTPMiddleware

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('api_server.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


# --------- Wrappers (same as training/eval) ---------
class FlattenObsWrapper(gym.ObservationWrapper):
    """Flattens nested observation spaces into a single Box."""
    def __init__(self, env):
        super().__init__(env)
        self._orig_obs_space = env.observation_space
        self.observation_space = flatten_space(self._orig_obs_space)

    def observation(self, obs):
        flattened = flatten(self._orig_obs_space, obs)
        if np.any(~np.isfinite(flattened)):
            flattened = np.nan_to_num(flattened, nan=0.0, posinf=1e6, neginf=-1e6)
        return flattened


class FlattenActionWrapper(gym.ActionWrapper):
    """Flattens nested action spaces into a single Box."""
    def __init__(self, env):
        super().__init__(env)
        self._orig_action_space = env.action_space
        if isinstance(self._orig_action_space, (SpaceDict, SpaceTuple)):
            self.action_space = flatten_space(self._orig_action_space)
            self._needs_unflatten = True
        elif isinstance(self._orig_action_space, Box):
            self.action_space = self._orig_action_space
            self._needs_unflatten = False
        else:
            raise TypeError(f"Unsupported action space type: {type(self._orig_action_space)}")

    def action(self, action: np.ndarray):
        if self._needs_unflatten:
            return unflatten(self._orig_action_space, action)
        return action


class NormalizeObservation(gym.ObservationWrapper):
    """Normalize observations using running statistics."""
    def __init__(self, env, epsilon=1e-8, update_stats=True):
        super().__init__(env)
        self.epsilon = epsilon
        self.update_stats = update_stats
        self.obs_mean = np.zeros(env.observation_space.shape, dtype=np.float32)
        self.obs_var = np.ones(env.observation_space.shape, dtype=np.float32)
        self.obs_count = float(epsilon)

    def freeze(self):
        """Freeze normalization stats (for evaluation)."""
        self.update_stats = False

    def unfreeze(self):
        """Unfreeze normalization stats."""
        self.update_stats = True

    def get_state(self):
        """Get current normalization state."""
        return dict(
            mean=self.obs_mean.copy(),
            var=self.obs_var.copy(),
            count=float(self.obs_count)
        )

    def set_state(self, state):
        """Set normalization state."""
        self.obs_mean = state["mean"].copy()
        self.obs_var = state["var"].copy()
        self.obs_count = float(state["count"])

    def observation(self, obs):
        if self.update_stats:
            batch_mean = obs
            batch_var = np.zeros_like(obs)
            batch_count = 1
            delta = batch_mean - self.obs_mean
            total_count = self.obs_count + batch_count
            self.obs_mean += delta * batch_count / total_count
            m_a = self.obs_var * self.obs_count
            m_b = batch_var * batch_count
            M2 = m_a + m_b + np.square(delta) * self.obs_count * batch_count / total_count
            self.obs_var = M2 / total_count
            self.obs_count = total_count
        normalized = (obs - self.obs_mean) / np.sqrt(self.obs_var + self.epsilon)
        return np.clip(normalized, -10, 10)


def make_env(normalize=True):
    """Create and wrap the environment."""
    env = gym.make("gymtorax/IterHybrid-v0")
    
    # Flatten observations
    if isinstance(env.observation_space, (SpaceDict, SpaceTuple)) or not isinstance(env.observation_space, Box):
        env = FlattenObsWrapper(env)
    
    # Add normalization
    if normalize:
        env = NormalizeObservation(env)
    
    # Flatten actions
    if isinstance(env.action_space, (SpaceDict, SpaceTuple)) or not isinstance(env.action_space, Box):
        env = FlattenActionWrapper(env)
    
    assert isinstance(env.observation_space, Box), "Obs must be Box after wrapping"
    assert isinstance(env.action_space, Box), "Action must be Box after wrapping"
    return env


# FastAPI app
app = FastAPI(title="SAC Model API", description="API for running SAC model timesteps")

# Custom middleware to handle OPTIONS requests before route matching
class OptionsMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        if request.method == "OPTIONS":
            origin = request.headers.get("origin", "*")
            
            logger.info(f"OPTIONS request intercepted for {request.url.path} from {origin}")
            
            return Response(
                content="",
                status_code=200,
                headers={
                    "Access-Control-Allow-Origin": "*",
                    "Access-Control-Allow-Methods": "GET, POST, PUT, DELETE, OPTIONS, HEAD",
                    "Access-Control-Allow-Headers": "*",
                    "Access-Control-Max-Age": "3600",
                }
            )
        return await call_next(request)

# Add OPTIONS middleware BEFORE CORS middleware
app.add_middleware(OptionsMiddleware)

# Add CORS middleware - allow all origins (CORS disabled)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=False,  # Must be False when using allow_origins=["*"]
    allow_methods=["*"],  # Allow all methods
    allow_headers=["*"],  # Allow all headers
    expose_headers=["*"],
    max_age=3600,  # Cache preflight requests for 1 hour
)

# Global variables for model and environment
model: Optional[SAC] = None
env: Optional[gym.Env] = None
current_obs: Optional[np.ndarray] = None
episode_step: int = 0
# Get the directory where this script is located
script_dir = Path(__file__).parent.absolute()
saved_data_dir = script_dir / "saved_environment_data"
saved_data_dir.mkdir(exist_ok=True)


class StepResponse(BaseModel):
    """Response model for step endpoint."""
    observation: list  # Flattened observation
    observation_raw: Optional[Dict[str, Any]] = None  # Unflattened structured observation
    reward: float  # Cumulative reward for all steps
    terminated: bool
    truncated: bool
    info: Dict[str, Any]
    episode_step: int
    saved_path: Optional[str] = None
    num_steps: int = 1  # Number of steps executed
    rewards: Optional[list] = None  # List of individual step rewards
    action: Optional[Dict[str, Any]] = None  # Last action taken


class EpisodeResponse(BaseModel):
    """Response model for episode evaluation endpoint."""
    episode: int
    episode_return: float
    episode_length: int
    step_rewards: list
    terminated: bool
    truncated: bool


class EpisodesResponse(BaseModel):
    """Response model for multiple episodes evaluation."""
    total_episodes: int
    episode_rewards: list  # Episodic returns
    episode_lengths: list
    mean_episode_return: float
    std_episode_return: float
    mean_episode_length: float
    std_episode_length: float
    episodes: list[EpisodeResponse]  # Detailed episode data


class ResetResponse(BaseModel):
    """Response model for reset endpoint."""
    observation: list
    observation_raw: Optional[Dict[str, Any]] = None  # Unflattened structured observation
    info: Dict[str, Any]
    episode_step: int


def serialize_for_json(obj):
    """Recursively serialize numpy arrays and other non-JSON-serializable types."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.integer, np.floating)):
        return float(obj)
    elif isinstance(obj, dict):
        return {k: serialize_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [serialize_for_json(item) for item in obj]
    elif hasattr(obj, 'tolist'):
        return obj.tolist()
    else:
        return obj


def save_environment_data(obs: np.ndarray, reward: float, step: int, terminated: bool, truncated: bool, info: dict):
    """Save observation and reward to a JSON file."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    filename = saved_data_dir / f"env_data_step_{step}_{timestamp}.json"
    
    # Serialize observation
    if isinstance(obs, np.ndarray):
        serializable_obs = obs.tolist()
    elif hasattr(obs, 'tolist'):
        serializable_obs = obs.tolist()
    elif isinstance(obs, (list, tuple)):
        serializable_obs = [serialize_for_json(x) for x in obs]
    else:
        serializable_obs = serialize_for_json(obs)
    
    data = {
        "timestamp": timestamp,
        "step": step,
        "observation": serializable_obs,
        "reward": float(reward),
        "terminated": bool(terminated),
        "truncated": bool(truncated),
        "info": serialize_for_json(info)  # Recursively serialize info dict
    }
    
    with open(filename, 'w') as f:
        json.dump(data, f, indent=2)
    
    return str(filename)


@app.on_event("startup")
async def load_model():
    """Load the model and create environment on startup."""
    global model, env, current_obs, episode_step, flatten_wrapper
    
    try:
        model_path = script_dir / "saved_model.zip"
        
        if not model_path.exists():
            error_msg = f"Model file not found: {model_path}"
            logger.error(error_msg)
            print(f"ERROR: {error_msg}")
            return  # Don't crash, but model won't be loaded
        
        print(f"Loading model from {model_path}...")
        logger.info(f"Loading model from {model_path}...")
        env = make_env(normalize=True)
        
        # Find and freeze normalization wrapper, and store FlattenObsWrapper reference
        norm_wrapper = None
        temp_env = env
        flatten_wrapper = None
        while isinstance(temp_env, gym.Wrapper):
            if isinstance(temp_env, NormalizeObservation):
                norm_wrapper = temp_env
            if isinstance(temp_env, FlattenObsWrapper):
                flatten_wrapper = temp_env
            temp_env = temp_env.env
        
        if norm_wrapper is not None:
            norm_wrapper.freeze()
            print("Normalization stats frozen for evaluation")
            logger.info("Normalization stats frozen for evaluation")
        
        if flatten_wrapper is not None:
            print(f"FlattenObsWrapper found - can unflatten observations for visualization")
            logger.info(f"FlattenObsWrapper found - original obs space: {flatten_wrapper._orig_obs_space}")
        
        model = SAC.load(str(model_path), env=env, device="auto")
        print(f"Model loaded successfully!")
        logger.info("Model loaded successfully!")
        print(f"Observation space: {env.observation_space}")
        print(f"Action space: {env.action_space}")
        
        # Reset environment
        current_obs, _ = env.reset()
        episode_step = 0
        print("Environment reset. Ready to accept requests.")
        logger.info("Environment reset. Ready to accept requests.")
    except Exception as e:
        error_msg = f"Failed to load model: {e}"
        logger.error(error_msg, exc_info=True)
        print(f"ERROR: {error_msg}")
        import traceback
        traceback.print_exc()
        # Don't raise - let server start but model won't be available


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "SAC Model API",
        "status": "ready" if model is not None else "not_ready",
        "endpoints": {
            "/step": "Run one or more timesteps with the model",
            "/reset": "Reset the environment",
            "/run_episodes": "Run multiple episodes and log each step",
            "/status": "Get current status",
            "/saved_data": "List saved environment data files"
        }
    }


@app.get("/status")
async def get_status():
    """Get current status of the model and environment."""
    try:
        # Handle observation shape - it might be a tuple or numpy array
        obs_shape = None
        if current_obs is not None:
            if hasattr(current_obs, 'shape'):
                shape = current_obs.shape
                if isinstance(shape, tuple):
                    obs_shape = list(shape)
                elif hasattr(shape, 'tolist'):
                    obs_shape = shape.tolist()
                else:
                    obs_shape = list(shape) if hasattr(shape, '__iter__') else [shape]
        
        return {
            "model_loaded": model is not None,
            "environment_ready": env is not None and current_obs is not None,
            "episode_step": episode_step,
            "observation_shape": obs_shape,
            "saved_data_directory": str(saved_data_dir)
        }
    except Exception as e:
        logger.error(f"Error getting status: {e}", exc_info=True)
        return {
            "model_loaded": model is not None,
            "environment_ready": False,
            "episode_step": episode_step,
            "observation_shape": None,
            "saved_data_directory": str(saved_data_dir),
            "error": str(e)
        }


@app.post("/reset", response_model=ResetResponse)
async def reset_environment():
    """Reset the environment to initial state."""
    global current_obs, episode_step
    
    if env is None:
        raise HTTPException(status_code=500, detail="Environment not initialized")
    
    current_obs, info = env.reset()
    episode_step = 0
    
    # Convert to serializable format using helper function
    serializable_obs = serialize_for_json(current_obs)
    serializable_info = serialize_for_json(info)
    
    # Try to unflatten observation for visualization
    serializable_raw_obs = None
    if flatten_wrapper is not None:
        try:
            from gymnasium.spaces.utils import unflatten
            # Unflatten using the original observation space
            raw_obs = unflatten(flatten_wrapper._orig_obs_space, current_obs)
            serializable_raw_obs = serialize_for_json(raw_obs)
            logger.debug(f"   🔄 Successfully unflattened reset observation: {type(raw_obs)}")
        except Exception as e:
            logger.debug(f"   Could not unflatten reset observation: {e}")
    
    return ResetResponse(
        observation=serializable_obs,
        observation_raw=serializable_raw_obs,
        info=serializable_info,
        episode_step=episode_step
    )


@app.options("/step")
async def options_step(request: Request, deterministic: bool = None):
    """Handle CORS preflight for /step endpoint.
    
    Accepts query parameters but ignores them - this prevents FastAPI from rejecting
    OPTIONS requests with query parameters.
    CORS is disabled - allowing all origins.
    """
    origin = request.headers.get("origin", "*")
    
    logger.info(f"✅ OPTIONS /step request from {origin}, allowing all origins")
    print(f"✅ OPTIONS /step request from {origin}, allowing all origins")
    
    return Response(
        content="",
        status_code=200,
        headers={
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "POST, OPTIONS",
            "Access-Control-Allow-Headers": "*",
            "Access-Control-Max-Age": "3600",
        }
    )

@app.post("/step", response_model=StepResponse)
async def run_step(request: Request, deterministic: bool = True, num_steps: int = 1):
    """
    Run one or more timesteps with the loaded model.
    
    Args:
        deterministic: If True, use deterministic policy (default: True)
        num_steps: Number of steps to run (default: 151)
    
    Returns:
        StepResponse with observation, cumulative reward, and saved file path
    """
    global current_obs, episode_step
    
    # Log the request
    client_ip = request.client.host if request.client else "unknown"
    logger.info(f"🔵 STEP REQUEST from {client_ip} - deterministic={deterministic}, num_steps={num_steps}, episode_step={episode_step}")
    
    if model is None or env is None:
        logger.error("❌ Model or environment not initialized")
        raise HTTPException(status_code=500, detail="Model or environment not initialized")
    
    if current_obs is None:
        logger.warning("⚠️ Environment not reset. Call /reset first.")
        raise HTTPException(status_code=400, detail="Environment not reset. Call /reset first.")
    
    if num_steps < 1:
        raise HTTPException(status_code=400, detail="num_steps must be >= 1")
    
    # Run multiple steps
    cumulative_reward = 0.0
    rewards_list = []
    final_obs = current_obs
    final_info = {}
    terminated = False
    truncated = False
    saved_paths = []
    
    for step_idx in range(num_steps):
        # Check if episode already ended
        if terminated or truncated:
            logger.info(f"⚠️ Episode ended before completing all {num_steps} steps. Completed {step_idx} steps.")
            break
        
        # Get action from model
        # Note: model.predict() expects observation in the format the model was trained with
        # The model internally handles the DummyVecEnv wrapping, so we pass the raw observation
        logger.info(f"🤖 Step {step_idx + 1}/{num_steps}: Getting action from model (deterministic={deterministic})...")
        
        # Ensure observation is 1D array (not batched) for model.predict
        obs_for_pred = current_obs
        if isinstance(current_obs, np.ndarray) and len(current_obs.shape) > 1:
            # If somehow batched, take first element
            obs_for_pred = current_obs[0] if current_obs.shape[0] == 1 else current_obs.flatten()
        
        action, _ = model.predict(obs_for_pred, deterministic=deterministic)
        
        # Log action and observation info for debugging
        if step_idx < 3:  # Log first 3 steps in detail
            action_str = f"action={action.tolist()[:3] if hasattr(action, 'tolist') else action}"
            obs_summary = f"obs_sum={obs_for_pred.sum():.6f}, obs_mean={obs_for_pred.mean():.6f}, obs_std={obs_for_pred.std():.6f}" if hasattr(obs_for_pred, 'sum') else "obs=N/A"
            logger.info(f"   Action: {action_str}")
            logger.info(f"   Observation: {obs_summary}")
        
        # Step environment - ensure action is correct format
        # If env is wrapped in DummyVecEnv, it expects batched actions, but our env should handle 1D
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Log observation structure BEFORE any processing (only for first step to avoid spam)
        if step_idx == 0:
            logger.info(f"📊 OBSERVATION STRUCTURE ANALYSIS (Step {step_idx + 1}):")
            logger.info(f"   Type: {type(obs)}")
            
            if isinstance(obs, dict):
                logger.info(f"   Dictionary with keys: {list(obs.keys())}")
                for key, value in obs.items():
                    if isinstance(value, np.ndarray):
                        logger.info(f"     '{key}': shape={value.shape}, dtype={value.dtype}, min={value.min():.6f}, max={value.max():.6f}")
                    elif isinstance(value, (list, tuple)):
                        logger.info(f"     '{key}': list/tuple of length {len(value)}")
                    else:
                        logger.info(f"     '{key}': {type(value).__name__} = {value}")
            elif isinstance(obs, np.ndarray):
                logger.info(f"   NumPy array: shape={obs.shape}, dtype={obs.dtype}")
                logger.info(f"   Min={obs.min():.6f}, Max={obs.max():.6f}, Mean={obs.mean():.6f}, Std={obs.std():.6f}")
                if obs.ndim == 1:
                    logger.info(f"   Already 1D, length={len(obs)}")
                else:
                    logger.info(f"   Multi-dimensional, will be flattened to 1D")
            elif isinstance(obs, (list, tuple)):
                logger.info(f"   List/Tuple: length={len(obs)}")
                if len(obs) > 0:
                    logger.info(f"   First element type: {type(obs[0])}")
            else:
                logger.info(f"   Other type: {obs}")
            
            # Log info dict structure
            if info:
                logger.info(f"   INFO dict keys: {list(info.keys()) if isinstance(info, dict) else 'Not a dict'}")
                if isinstance(info, dict) and 'observation' in info:
                    logger.info(f"   INFO contains 'observation' key: {type(info['observation'])}")
            
            # Check if observation is already flattened (from wrapper)
            # Try to access unwrapped env to see original structure
            try:
                unwrapped_env = env
                wrapper_chain = []
                while hasattr(unwrapped_env, 'env'):
                    wrapper_chain.append(type(unwrapped_env).__name__)
                    unwrapped_env = unwrapped_env.env
                wrapper_chain.append(type(unwrapped_env).__name__)
                logger.info(f"   🔗 Environment wrapper chain: {' -> '.join(wrapper_chain)}")
                
                # Check if FlattenObsWrapper is in the chain
                if 'FlattenObsWrapper' in wrapper_chain:
                    logger.info(f"   📦 FlattenObsWrapper detected - observation is already flattened!")
                    # Try to get original observation space
                    if hasattr(env, 'observation_space'):
                        logger.info(f"   📐 Current observation_space: {env.observation_space}")
                    # Try to access original obs space from unwrapped
                    if hasattr(unwrapped_env, 'observation_space'):
                        logger.info(f"   📐 Unwrapped observation_space: {unwrapped_env.observation_space}")
            except Exception as e:
                logger.debug(f"   Could not inspect wrapper chain: {e}")
            
            # Also log how FlattenObsWrapper transforms the observation
            try:
                # Find FlattenObsWrapper in the chain
                temp_env = env
                while hasattr(temp_env, 'env'):
                    if isinstance(temp_env, FlattenObsWrapper):
                        logger.info(f"   🔍 FlattenObsWrapper found!")
                        logger.info(f"      Original obs space: {temp_env._orig_obs_space}")
                        logger.info(f"      Flattened obs space: {temp_env.observation_space}")
                        # Try to see what the original observation would look like
                        # by checking if we can access the unwrapped env's last observation
                        unwrapped = temp_env.env
                        while hasattr(unwrapped, 'env'):
                            unwrapped = unwrapped.env
                        if hasattr(unwrapped, 'last_observation') or hasattr(unwrapped, '_last_obs'):
                            raw_obs = getattr(unwrapped, 'last_observation', None) or getattr(unwrapped, '_last_obs', None)
                            if raw_obs is not None:
                                logger.info(f"      Raw observation from unwrapped env: {type(raw_obs)}")
                                if isinstance(raw_obs, dict):
                                    logger.info(f"         Raw obs keys: {list(raw_obs.keys())}")
                        break
                    temp_env = temp_env.env
            except Exception as e:
                logger.debug(f"   Could not inspect FlattenObsWrapper: {e}")
        
        cumulative_reward += float(reward)
        rewards_list.append(float(reward))
        
        # Log observation change
        obs_change = np.abs(obs - current_obs).mean() if hasattr(obs, '__sub__') and hasattr(current_obs, '__sub__') else 0.0
        logger.info(f"✅ Step {step_idx + 1}/{num_steps} complete - Reward: {reward:.10f}, Cumulative: {cumulative_reward:.10f}, Obs change: {obs_change:.10f}")
        
        # Save environment data for each step
        saved_path = save_environment_data(
            obs=obs,
            reward=reward,
            step=episode_step,
            terminated=terminated,
            truncated=truncated,
            info=info
        )
        saved_paths.append(saved_path)
        
        # Try to unflatten observation for visualization
        raw_obs = None
        if flatten_wrapper is not None and isinstance(obs, np.ndarray) and obs.ndim == 1:
            try:
                # Unflatten using the original observation space
                raw_obs = unflatten(flatten_wrapper._orig_obs_space, obs)
                if step_idx == 0:
                    logger.info(f"   🔄 Successfully unflattened observation: {type(raw_obs)}")
                    if isinstance(raw_obs, dict):
                        logger.info(f"      Unflattened keys: {list(raw_obs.keys())}")
            except Exception as e:
                logger.debug(f"   Could not unflatten observation: {e}")
        
        # Update current observation
        current_obs = obs
        final_obs = obs
        final_info = info
        final_raw_obs = raw_obs
        final_action = action
        episode_step += 1
        
        # If episode ended, reset automatically
        if terminated or truncated:
            logger.info(f"🔄 Episode ended (terminated={terminated}, truncated={truncated}). Resetting environment...")
            current_obs, _ = env.reset()
            episode_step = 0
            logger.info("✅ Environment reset")
            break
    
    logger.info(f"💾 Saved {len(saved_paths)} data files. Total cumulative reward: {cumulative_reward:.6f}")
    
    # Convert to serializable format using helper function
    serializable_obs = serialize_for_json(final_obs)
    serializable_info = serialize_for_json(final_info)
    serializable_raw_obs = serialize_for_json(final_raw_obs) if final_raw_obs is not None else None
    
    # Serialize action
    serializable_action = None
    if final_action is not None:
        try:
            # Try to unflatten action if needed
            temp_env = env
            action_wrapper = None
            while isinstance(temp_env, gym.Wrapper):
                if isinstance(temp_env, FlattenActionWrapper):
                    action_wrapper = temp_env
                    break
                temp_env = temp_env.env
            
            if action_wrapper is not None and action_wrapper._needs_unflatten:
                unflattened_action = unflatten(action_wrapper._orig_action_space, final_action)
                serializable_action = serialize_for_json(unflattened_action)
            else:
                if isinstance(final_action, np.ndarray):
                    serializable_action = final_action.tolist()
                else:
                    serializable_action = serialize_for_json(final_action)
        except Exception as e:
            logger.debug(f"Could not unflatten action: {e}")
            if isinstance(final_action, np.ndarray):
                serializable_action = final_action.tolist()
            else:
                serializable_action = serialize_for_json(final_action)
    
    return StepResponse(
        observation=serializable_obs,
        observation_raw=serializable_raw_obs,
        reward=cumulative_reward,
        terminated=bool(terminated),
        truncated=bool(truncated),
        info=serializable_info,
        episode_step=episode_step,
        saved_path=saved_paths[-1] if saved_paths else None,  # Return last saved path
        num_steps=len(rewards_list),
        rewards=rewards_list if len(rewards_list) > 1 else None,  # Only include if multiple steps
        action=serializable_action
    )


@app.post("/run_episodes", response_model=EpisodesResponse)
async def run_episodes(request: Request, num_episodes: int = 10, max_steps_per_episode: int = 1, deterministic: bool = True):
    """
    Run multiple episodes and log each step.
    
    Args:
        num_episodes: Number of episodes to run (default: 10)
        max_steps_per_episode: Maximum steps per episode (default: 151)
        deterministic: If True, use deterministic policy (default: True)
    
    Returns:
        EpisodesResponse with detailed episode data
    """
    global current_obs, episode_step
    
    if model is None or env is None:
        raise HTTPException(status_code=500, detail="Model or environment not initialized")
    
    client_ip = request.client.host if request.client else "unknown"
    logger.info(f"🔵 RUN EPISODES REQUEST from {client_ip} - num_episodes={num_episodes}, max_steps={max_steps_per_episode}, deterministic={deterministic}")
    print(f"\n{'='*60}")
    print(f"Running {num_episodes} episodes (max {max_steps_per_episode} steps each, deterministic={deterministic})...")
    print(f"{'='*60}\n")
    
    episode_rewards = []
    episode_lengths = []
    episode_details = []
    
    for episode_idx in range(num_episodes):
        # Reset environment
        current_obs, info = env.reset()
        episode_step = 0
        episode_reward = 0.0
        step_rewards_this_episode = []
        terminated = False
        truncated = False
        
        print(f"\n--- Episode {episode_idx + 1}/{num_episodes} ---")
        logger.info(f"Starting episode {episode_idx + 1}/{num_episodes}")
        
        # Run episode
        for step_idx in range(max_steps_per_episode):
            # Get action from model
            obs_for_pred = current_obs
            if isinstance(current_obs, np.ndarray) and len(current_obs.shape) > 1:
                obs_for_pred = current_obs[0] if current_obs.shape[0] == 1 else current_obs.flatten()
            
            action, _ = model.predict(obs_for_pred, deterministic=deterministic)
            
            # Step environment
            obs, reward, terminated, truncated, info = env.step(action)
            
            # Handle potential batching
            if isinstance(obs, np.ndarray) and len(obs.shape) > 1 and obs.shape[0] == 1:
                obs = obs[0]
            if isinstance(reward, (list, np.ndarray)) and len(reward) > 0:
                reward = reward[0] if isinstance(reward, (list, np.ndarray)) else reward
            
            # Store step reward
            step_reward = float(reward)
            episode_reward += step_reward
            episode_step += 1
            step_rewards_this_episode.append(step_reward)
            current_obs = obs
            
            # Log step result
            print(f"  Step {episode_step:3d}: Reward = {step_reward:12.10f}, Episode Cumulative = {episode_reward:12.10f}", end="")
            logger.info(f"Episode {episode_idx + 1}, Step {episode_step}: Reward = {step_reward:.10f}, Cumulative = {episode_reward:.10f}")
            
            if terminated:
                print(" [TERMINATED]")
                logger.info(f"Episode {episode_idx + 1} terminated at step {episode_step}")
                break
            elif truncated:
                print(" [TRUNCATED]")
                logger.info(f"Episode {episode_idx + 1} truncated at step {episode_step}")
                break
            else:
                print()
        
        # Store episode data
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_step)
        
        # Create episode response
        episode_detail = EpisodeResponse(
            episode=episode_idx + 1,
            episode_return=episode_reward,
            episode_length=episode_step,
            step_rewards=step_rewards_this_episode,
            terminated=bool(terminated),
            truncated=bool(truncated)
        )
        episode_details.append(episode_detail)
        
        print(f"  → Episode {episode_idx + 1} complete: {episode_step} steps, Return = {episode_reward:.10f}")
        logger.info(f"Episode {episode_idx + 1} complete: {episode_step} steps, Return = {episode_reward:.10f}")
    
    # Summary
    mean_return = float(np.mean(episode_rewards))
    std_return = float(np.std(episode_rewards))
    mean_length = float(np.mean(episode_lengths))
    std_length = float(np.std(episode_lengths))
    
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Mean episodic return: {mean_return:.4f} ± {std_return:.4f}")
    print(f"Per-episode returns: {[f'{r:.4f}' for r in episode_rewards]}")
    print(f"Per-episode lengths: {episode_lengths}")
    print(f"Avg length: {mean_length:.1f} ± {std_length:.1f}")
    print(f"{'='*60}\n")
    
    logger.info(f"Completed {num_episodes} episodes. Mean return: {mean_return:.4f} ± {std_return:.4f}")
    
    return EpisodesResponse(
        total_episodes=num_episodes,
        episode_rewards=episode_rewards,
        episode_lengths=episode_lengths,
        mean_episode_return=mean_return,
        std_episode_return=std_return,
        mean_episode_length=mean_length,
        std_episode_length=std_length,
        episodes=episode_details
    )


@app.get("/saved_data")
async def list_saved_data():
    """List all saved environment data files."""
    files = sorted(saved_data_dir.glob("*.json"), key=lambda x: x.stat().st_mtime, reverse=True)
    return {
        "count": len(files),
        "files": [{"name": f.name, "path": str(f), "size": f.stat().st_size} for f in files[:50]]  # Last 50 files
    }


if __name__ == "__main__":
    import uvicorn
    import socket
    
    # Check if port 8000 is already in use
    def is_port_in_use(port):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            return s.connect_ex(('localhost', port)) == 0
    
    port = 8000
    if is_port_in_use(port):
        logger.warning(f"Port {port} is already in use. Attempting to use it anyway...")
        print(f"⚠️  Warning: Port {port} may be in use. If the server fails to start, kill the process using: lsof -ti:{port} | xargs kill -9")
    
    try:
        logger.info(f"Starting FastAPI server on http://0.0.0.0:{port}")
        print(f"🚀 Starting FastAPI server on http://0.0.0.0:{port}")
        uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
    except OSError as e:
        if "address already in use" in str(e).lower() or e.errno == 48:
            logger.error(f"Port {port} is already in use. Kill the process with: lsof -ti:{port} | xargs kill -9")
            print(f"❌ ERROR: Port {port} is already in use!")
            print(f"   Kill the process with: lsof -ti:{port} | xargs kill -9")
            raise
        else:
            raise
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
        print("\n👋 Server stopped by user")
    except Exception as e:
        logger.error(f"Server crashed: {e}", exc_info=True)
        print(f"❌ Server crashed: {e}")
        raise

