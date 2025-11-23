from typing import List, Dict, Any

from rocket_league_rl.rlgym.api import DoneCondition, AgentID
from rocket_league_rl.rlgym.rocket_league.api import GameState


class NoopCondition(DoneCondition[AgentID, GameState]):
    """
    A DoneCondition that is never satisfied.
    """
    def reset(self, agents: List[AgentID], initial_state: GameState, shared_info: Dict[str, Any]) -> None:
        pass

    def is_done(self, agents: List[AgentID], state: GameState, shared_info: Dict[str, Any]) -> Dict[AgentID, bool]:
        return {agent: False for agent in agents}
