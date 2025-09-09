# main.py
import argparse, os, ray
from ray import air, tune
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.env.wrappers.pettingzoo_env import PettingZooEnv
from ray.tune.registry import register_env
from pettingzoo.utils import parallel_to_aec
from boxjump.box_env import BoxJumpEnvironment

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--timesteps", type=int, default=500_000)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--gpus", type=float, default=0.0)
    args = p.parse_args()

    env_cfg = dict(
        num_boxes=8, render_mode=None, include_time=True, include_highest=True,
        reward_mode="dense_height_stable_sq", physics_steps_per_action=6,
        physics_timestep_multiplier=2.0, termination_on_fall=True
    )

    env_name = "BoxJumpPZ-v1"
    register_env(env_name, lambda c: PettingZooEnv(parallel_to_aec(BoxJumpEnvironment(**c))))

    probe = BoxJumpEnvironment(**env_cfg)
    agent0 = probe.possible_agents[0]
    obs_space, act_space = probe.observation_space(agent0), probe.action_space(agent0)

    config = (
        PPOConfig()
        .environment(env=env_name, env_config=env_cfg)
        .framework("torch")
        .resources(num_gpus=args.gpus)
        .rollouts(num_rollout_workers=args.workers)
        .training(
            lr=3e-4, gamma=0.99, lambda_=0.95, clip_param=0.2,
            train_batch_size=16384, sgd_minibatch_size=2048, num_sgd_iter=10,
            vf_clip_param=200.0, grad_clip=0.5, model={"fcnet_hiddens":[256,256]}
        )
        .multi_agent(
            policies={"shared": (None, obs_space, act_space, {})},
            policy_mapping_fn=lambda *_: "shared",
            policies_to_train=["shared"],
        )
    ).to_dict()

    ray.init(ignore_reinit_error=True)
    tuner = tune.Tuner(
        "PPO",
        run_config=air.RunConfig(
            stop={"timesteps_total": args.timesteps},
            checkpoint_config=air.CheckpointConfig(checkpoint_at_end=True),
            local_dir=os.path.join(os.getcwd(), "rllib_results"),
            verbose=1,
        ),
        param_space=config,
    )
    result = tuner.fit()
    best = result.get_best_result(metric="episode_reward_mean", mode="max")
    if best.checkpoint: print("Best checkpoint:", best.checkpoint.to_directory())
    print("Best episode_reward_mean:", best.metrics.get("episode_reward_mean"))
