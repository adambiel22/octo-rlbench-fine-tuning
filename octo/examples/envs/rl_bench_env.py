import copy
from typing import List

import dlimp as dl
import gymnasium as gym
import jax.numpy as jnp
import numpy as np
from PIL import Image

from rlbench.gym import RLBenchEnv
from rlbench.utils import name_to_task_class
from rlbench.action_modes.action_mode import UR5ActionMode
from rlbench.action_modes.arm_action_modes import JointVelocity
from rlbench.action_modes.gripper_action_modes import Discrete


class RLBenchEnvAdapter(gym.Env):
    def __init__(
        self,
        rl_bench_env: RLBenchEnv,
        im_size: int = 256,
        seed: int = 1234,
    ):
        self._env = rl_bench_env
        self.observation_space = gym.spaces.Dict(
            {
                **{
                    f"image_{i}": gym.spaces.Box(
                        low=np.zeros((im_size, im_size, 3)),
                        high=255 * np.ones((im_size, im_size, 3)),
                        dtype=np.uint8,
                    )
                    for i in ["primary", "wrist"]
                },
                "proprio": gym.spaces.Box(
                        low=-np.infty * np.ones(6),
                        high=np.infty * np.ones(6),
                        dtype=np.float32,
                    )
            }
        )
        self.action_space = self._env.action_space
        self._im_size = im_size
        self._rng = np.random.default_rng(seed)

    def step(self, action):
        observation, reward, terminated, _, _ = self._env.step(action)
        obs = self._extract_obs(observation)

        # It assumes that reward == 1.0 means success
        if reward == 1.0:
            self._episode_is_success = 1

        rendered_frame = self.render(obs)

        return obs, reward, terminated, False, {"frame": rendered_frame}
    
    def render(self, obs = None):
        rendered_frame = self._env.render()

        if rendered_frame is not None:

            wrist_img = Image.fromarray(obs["image_wrist"].numpy())
            wrist_img = wrist_img.resize((360, 360), Image.Resampling.BILINEAR)
            resized_wrist_img = np.array(wrist_img)

            return np.concatenate([rendered_frame, resized_wrist_img], axis=1)

    def reset(self, **kwargs):
        obs, info = self._env.reset(**kwargs)

        obs = self._extract_obs(obs)
        self._episode_is_success = 0
        self.language_instruction = info["text_descriptions"]

        rendered_frame = self.render(obs)

        return obs, {"frame": rendered_frame}

    def _extract_obs(self, obs):
        curr_obs = {
            "image_primary": obs["front_rgb"],
            "image_wrist": obs["wrist_rgb"],
            "proprio": obs["joint_positions"]
        }
        curr_obs = dl.transforms.resize_images(
            curr_obs, match=curr_obs.keys(), size=(self._im_size, self._im_size)
        )

        return curr_obs

    def get_task(self):
        return {
            "language_instruction": [self.language_instruction],
        }

    def get_episode_metrics(self):
        return {
            "success_rate": self._episode_is_success,
        }


# register gym environments
gym.register(
    "place_shape_in_shape_sorter-vision-v0",
    entry_point=lambda: RLBenchEnvAdapter(
        RLBenchEnv(task_class=name_to_task_class("place_shape_in_shape_sorter"),
                   observation_mode='vision',
                   render_mode="rgb_array",
                   robot_setup="ur5",
                   headless=True,
                   action_mode=UR5ActionMode())
    ),
)
