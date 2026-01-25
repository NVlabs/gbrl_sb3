from gymnasium.envs.registration import register


def register_minigrid_tests():
    register(
        id="MiniGrid-SpuriousFetch-8x8-N3-v0",
        entry_point="env.minigrid:SpuriousFetchEnv",
        kwargs={"size": 8, "numObjs": 3, "use_box": True, "mission_based": True, "randomize": False},
    )
    register(
        id="MiniGrid-SpuriousFetch-8x8-N3-v1",
        entry_point="env.minigrid:SpuriousFetchEnv",
        kwargs={"size": 8, "numObjs": 3, "use_box": True, "mission_based": False, "randomize": False},
    )
    register(
        id="MiniGrid-SpuriousFetch-8x8-N3-v2",
        entry_point="env.minigrid:SpuriousFetchEnv",
        kwargs={"size": 8, "numObjs": 3, "use_box": True, "mission_based": False, "randomize": True},
    )
    register(
        id="MiniGrid-SpuriousFetch-8x8-N3-v3",
        entry_point="env.minigrid:SpuriousFetchEnv",
        kwargs={"size": 8, "numObjs": 3, "use_box": False},
    )
    register(
        id="MiniGrid-SpuriousFetch-8x8-N3-v4",
        entry_point="env.minigrid:SpuriousFetchEnv",
        kwargs={"size": 8, "numObjs": 3, "use_box": False, "purple_ball": True},
    )
    register(
        id="MiniGrid-SpuriousFetch-8x8-N3-v5",
        entry_point="env.minigrid:SpuriousFetchEnv",
        kwargs={"size": 8, "numObjs": 3, "use_box": False, "purple_ball": True, 'grey_ball': True},
    )
    register(
        id="MiniGrid-SpuriousFetch-8x8-N3-v6",
        entry_point="env.minigrid:SpuriousFetchEnv",
        kwargs={"size": 8, "numObjs": 3, "use_box": True, "purple_box": True, 'randomize': True},
    )
    register(
        id="MiniGrid-SpuriousFetch-8x8-N3-v7",
        entry_point="env.minigrid:SpuriousFetchEnv",
        kwargs={"size": 8, "numObjs": 3, "use_box": False, "add_red_ball": True},
    )
    register(
        id="MiniGrid-ObstructedMazeCompliance_1Dlhb-v0",
        entry_point="env.minigrid:ObstructedMazeCompliance_1Dl",
        kwargs={"key_in_box": True, "blocked": True, "guided_reward": False},
    )
    register(
        id="MiniGrid-ObstructedMazeCompliance_1Dlh-v0",
        entry_point="env.minigrid:ObstructedMazeCompliance_1Dl",
        kwargs={"key_in_box": True, "blocked": False, "guided_reward": False},
    )
    register(
        id="MiniGrid-ObstructedMazeCompliance_1Dl-v0",
        entry_point="env.minigrid:ObstructedMazeCompliance_1Dl",
        kwargs={"key_in_box": False, "blocked": False, "guided_reward": False},
    )
    register(
        id="MiniGrid-ObstructedMazeCompliance_1Dlhb-v1",
        entry_point="env.minigrid:ObstructedMazeCompliance_1Dl",
        kwargs={"key_in_box": True, "blocked": True, "guided_reward": True},
    )
    register(
        id="MiniGrid-ObstructedMazeCompliance_1Dlh-v1",
        entry_point="env.minigrid:ObstructedMazeCompliance_1Dl",
        kwargs={"key_in_box": True, "blocked": False, "guided_reward": True},
    )
    register(
        id="MiniGrid-ObstructedMazeCompliance_1Dl-v1",
        entry_point="env.minigrid:ObstructedMazeCompliance_1Dl",
        kwargs={"key_in_box": False, "blocked": False, "guided_reward": True},
    )
    register(
        id="MiniGrid-ObstructedMazeCompliance_1Dlhb-v2r",
        entry_point="env.minigrid:ObstructedMazeCompliance_1Dl",
        kwargs={"key_in_box": True, "blocked": True, "guided_reward": True, "partial_guidance": 'before_goal'},
    )
    register(
        id="MiniGrid-ObstructedMazeCompliance_1Dlh-v2r",
        entry_point="env.minigrid:ObstructedMazeCompliance_1Dl",
        kwargs={"key_in_box": True, "blocked": False, "guided_reward": True, "partial_guidance": 'before_goal'},
    )
    register(
        id="MiniGrid-ObstructedMazeCompliance_1Dl-v2r",
        entry_point="env.minigrid:ObstructedMazeCompliance_1Dl",
        kwargs={"key_in_box": False, "blocked": False, "guided_reward": True, "partial_guidance": 'before_goal'},
    )
    register(
        id="MiniGrid-ObstructedMazeCompliance_1Dlhb-v2",
        entry_point="env.minigrid:ObstructedMazeCompliance_1Dl",
        kwargs={"key_in_box": True, "blocked": True, "guided_reward": False, "partial_guidance": 'before_goal'},
    )
    register(
        id="MiniGrid-ObstructedMazeCompliance_1Dlh-v2",
        entry_point="env.minigrid:ObstructedMazeCompliance_1Dl",
        kwargs={"key_in_box": True, "blocked": False, "guided_reward": False, "partial_guidance": 'before_goal'},
    )
    register(
        id="MiniGrid-ObstructedMazeCompliance_1Dl-v2",
        entry_point="env.minigrid:ObstructedMazeCompliance_1Dl",
        kwargs={"key_in_box": False, "blocked": False, "guided_reward": False, "partial_guidance": 'before_goal'},
    )
    register(
        id="MiniGrid-ObstructedMazeCompliance_1Dlhb-v3r",
        entry_point="env.minigrid:ObstructedMazeCompliance_1Dl",
        kwargs={"key_in_box": True, "blocked": True, "guided_reward": True, "partial_guidance": 'before_door'},
    )
    register(
        id="MiniGrid-ObstructedMazeCompliance_1Dlh-v3r",
        entry_point="env.minigrid:ObstructedMazeCompliance_1Dl",
        kwargs={"key_in_box": True, "blocked": False, "guided_reward": True, "partial_guidance": 'before_door'},
    )
    register(
        id="MiniGrid-ObstructedMazeCompliance_1Dl-v3r",
        entry_point="env.minigrid:ObstructedMazeCompliance_1Dl",
        kwargs={"key_in_box": False, "blocked": False, "guided_reward": True, "partial_guidance": 'before_door'},
    )
    register(
        id="MiniGrid-ObstructedMazeCompliance_1Dlhb-v3",
        entry_point="env.minigrid:ObstructedMazeCompliance_1Dl",
        kwargs={"key_in_box": True, "blocked": True, "guided_reward": False, "partial_guidance": 'before_door'},
    )
    register(
        id="MiniGrid-ObstructedMazeCompliance_1Dlh-v3",
        entry_point="env.minigrid:ObstructedMazeCompliance_1Dl",
        kwargs={"key_in_box": True, "blocked": False, "guided_reward": False, "partial_guidance": 'before_door'},
    )
    register(
        id="MiniGrid-ObstructedMazeCompliance_1Dl-v3",
        entry_point="env.minigrid:ObstructedMazeCompliance_1Dl",
        kwargs={"key_in_box": False, "blocked": False, "guided_reward": False, "partial_guidance": 'before_door'},
    )
    register(
        id="MiniGrid-GuidedLockedRoom-v0",
        entry_point="env.guidance.locked_room:GuidedLockedRoomEnv",
        kwargs={},
    )
    register(
        id="MiniGrid-DynamicCrossing-v0",
        entry_point="env.safety.dynamic_lava_room:DynamicCrossing",
        kwargs={},
    )
    register(
        id="MiniGrid-DynamicCrossing-v1",
        entry_point="env.safety.dynamic_lava_room:DynamicCrossing",
        kwargs={"reward_penalty": True},
    )
    register(
        id="MiniGrid-FragileCrossing-v0",
        entry_point="env.safety.fragile_carry:FragileCrossingEnv",
        kwargs={},
    )
    register(
        id="MiniGrid-FragileCrossing-v1",
        entry_point="env.safety.fragile_carry:FragileCrossingEnv",
        kwargs={"reward_penalty": True},
    )
