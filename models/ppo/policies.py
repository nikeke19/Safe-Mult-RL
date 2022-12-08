# This file is here just to define MlpPolicy/CnnPolicy
# that work for PPO
# from stable_baselines3.common.policies import (
#     ActorCriticCnnPolicy,
#     ActorCriticPolicy,
#     MultiInputActorCriticPolicy,
#     register_policy,
# )
#
# MlpPolicy = ActorCriticPolicy
# CnnPolicy = ActorCriticCnnPolicy
# MultiInputPolicy = MultiInputActorCriticPolicy
#
# register_policy("MlpPolicy", ActorCriticPolicy)
# register_policy("CnnPolicy", ActorCriticCnnPolicy)
# register_policy("MultiInputPolicy", MultiInputPolicy)

from models.common.policies import (
    CustomActorCriticPolicy,
    ActorCriticCnnPolicy,
    MultiInputActorCriticPolicy,
)

from stable_baselines3.common.policies import register_policy

MlpPolicy = CustomActorCriticPolicy
CnnPolicy = ActorCriticCnnPolicy
MultiInputPolicy = MultiInputActorCriticPolicy

register_policy("CustomMlpPolicy", CustomActorCriticPolicy)
register_policy("CustomCnnPolicy", ActorCriticCnnPolicy)
register_policy("CustomMultiInputPolicy", MultiInputPolicy)
