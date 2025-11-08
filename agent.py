import abc
from gymnasium import spaces # Assuming a gym-like space object

class Agent(abc.ABC):
    """
    Abstract base class for all agents.

    This class defines the minimal interface an agent needs to interact
    with an environment. Subclasses must implement the 'act' method.
    """

    def __init__(self, action_space: spaces.Space):
        """
        Initialize the agent with the given action space.

        Args:
            action_space: The environment's action space (e.g., from gymnasium).
        """
        self.action_space = action_space

    @abc.abstractmethod
    def act(self, observation) -> dict:
        """
        Compute the next action based on the current observation.

        Subclasses MUST implement this method.

        Args:
            observation: The current observation from the environment.

        Returns:
            dict: Action dictionary for the environment.
        """
        pass # The implementation is left to the child class
    
class RandomAgent(Agent):
    """Agent that produces random actions within the action space.
    
    This agent inherits the __init__ method from the base Agent class.
    """

    def act(self, observation) -> dict:
        """Compute a random next action.
        
        Args:
            observation: The current observation (unused by this agent).

        Returns:
            dict: A random action dictionary.
        """
        # The observation is ignored, but must be an argument
        # to match the base class's abstract method signature.
        _ = observation 
        
        action = self.action_space.sample()
        return action