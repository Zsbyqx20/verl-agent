# Copyright 2025 Nanyang Technological University (NTU), Singapore
# and the verl-agent (GiGPO) team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import List, Tuple, Dict, Union, Any
from collections import defaultdict
import torch
import numpy as np
from functools import partial
import os
import re
import json
from PIL import Image
from agent_system.environments.prompts import *
from agent_system.environments.base import EnvironmentManagerBase, to_numpy
from agent_system.memory import SimpleMemory, SearchMemory
from omegaconf import OmegaConf

def parse_gamefile(infos):
    gamefile = []
    for info in infos:
        if 'extra.gamefile' in info:
            gamefile.append(info['extra.gamefile'])
        else:
            gamefile.append(None)
    return gamefile

def set_gamefile(infos, gamefile):
    for i in range(len(infos)):
        if 'extra.gamefile' in infos[i]:
            infos[i]['extra.gamefile'] = gamefile[i]
        else:
            infos[i]['extra.gamefile'] = None
    return infos


class SearchEnvironmentManager(EnvironmentManagerBase):
    """
    EnvironmentManager for SearchEnv.
    """
    def __init__(self, envs, projection_f, config):
        self.memory = SearchMemory()
        super().__init__(envs, projection_f, config)

    def reset(self, kwargs) -> Tuple[Dict[str, Any], List[Dict]]:
        obs, infos = self.envs.reset(kwargs=kwargs)
        self.tasks = obs

        self.memory.reset(batch_size=len(obs))

        observations = {
            "text": self.build_text_obs(obs, init=True),
            "image": None,
            "anchor": obs.copy()
        }
        
        return observations, infos

    def step(self, text_actions: List[str]):
        actions, valids = self.projection_f(text_actions)
        next_obs, rewards, dones, infos = self.envs.step(actions)
        self.memory.store({
            "search": actions,
            "information": next_obs,
        })

        next_observations = {
            "text": self.build_text_obs(next_obs),
            "image": None,
            "anchor": next_obs.copy()
        }
        
        for i, info in enumerate(infos):
            info["is_action_valid"] = to_numpy(valids[i])

        rewards = to_numpy(rewards)
        dones = to_numpy(dones)

        return next_observations, rewards, dones, infos

    def build_text_obs(
        self,
        text_obs: List[str],
        init: bool = False
    ) -> List[str]:
        postprocess_text_obs: List[str] = []

        if not init and self.config.env.history_length > 0:
            memory_ctx, _ = self.memory.fetch(
                self.config.env.history_length,
                obs_key="information",
                action_key="search"
            )

        for i in range(len(text_obs)):
            if init or self.config.env.history_length <= 0:
                obs_i = SEARCH_TEMPLATE_NO_HIS.format(
                    task_description=self.tasks[i]
                )
            else:
                obs_i = SEARCH_TEMPLATE.format(
                    task_description=self.tasks[i],
                    memory_context=memory_ctx[i],
                    step_count=len(self.memory[i]),
                )
            postprocess_text_obs.append(obs_i)

        return postprocess_text_obs


    def _process_batch(self, batch_idx, total_batch_list, total_infos, success):
        # Find the last entry with active masks
        for i in reversed(range(len(total_batch_list[batch_idx]))):
            batch_item = total_batch_list[batch_idx][i]
            if batch_item['active_masks']:
                info = total_infos[batch_idx][i]
                won_value = float(info['won'])
                success['success_rate'].append(won_value)
                
                data_source = info.get("data_source")
                success[f"{data_source}_success_rate"].append(won_value)
                return  # Exit after finding the first active mask
            

class AlfWorldEnvironmentManager(EnvironmentManagerBase):
    def __init__(self, envs, projection_f, config):
        self.memory = SimpleMemory()
        super().__init__(envs, projection_f, config)
    
    def reset(self, kwargs):
        text_obs, image_obs, infos = self.envs.reset()
        self.gamefile = parse_gamefile(infos)
        # initialize the history buffer
        self.memory.reset(batch_size = len(text_obs))
        self.tasks = []
        self.pre_text_obs = text_obs
        self.extract_task(text_obs)

        full_text_obs = self.build_text_obs(text_obs, self.envs.get_admissible_commands, init=True)
        return {'text': full_text_obs, 'image': image_obs, 'anchor': text_obs}, infos
    
    def step(self, text_actions: List[str]):
        actions, valids = self.projection_f(text_actions, self.envs.get_admissible_commands)
        text_obs, image_obs, rewards, dones, infos = self.envs.step(actions)
        self.memory.store({'text_obs': self.pre_text_obs, 'action': actions})
        self.pre_text_obs = text_obs

        full_text_obs = self.build_text_obs(text_obs, self.envs.get_admissible_commands)
        if infos[0].get("extra.gamefile") is None:
            infos = set_gamefile(infos, self.gamefile)

        # add action_valid to infos
        for i, info in enumerate(infos):
            info['is_action_valid'] = to_numpy(valids[i])

        next_observations = {'text': full_text_obs, 'image': image_obs, 'anchor': text_obs}
        rewards = to_numpy(rewards)
        dones = to_numpy(dones)

        return next_observations, rewards, dones, infos
    
    def extract_task(self, text_obs: List[str]):
        for obs in text_obs:
            task_start = obs.find('Your task is to: ')
            
            if task_start != -1:
                self.tasks.append(obs[task_start + len('Your task is to: '):].strip())
            else:
                raise ValueError("Task description not found in text observation.")
        

    def build_text_obs(self, text_obs: List[str], admissible_actions: List[List[str]], init: bool = False) -> List[str]:
        """
        This function builds the text observation for the agent.
        """
        postprocess_text_obs = []
        if not init and self.config.env.history_length > 0:
            memory_contexts, valid_lens = self.memory.fetch(
                    self.config.env.history_length,
                    obs_key="text_obs",
                    action_key="action")
            
        for i in range(len(text_obs)):
            # exclude 'help' in admissible_actions[i]
            reformatted_admissible_actions = "\n ".join(f"'{s}'" for s in admissible_actions[i] if s != 'help')

            if init or self.config.env.history_length <= 0:
                obs = ALFWORLD_TEMPLATE_NO_HIS.format(
                    current_observation=text_obs[i],
                    admissible_actions=reformatted_admissible_actions
                )
            else:
                obs = ALFWORLD_TEMPLATE.format(
                    task_description=self.tasks[i],
                    step_count=len(self.memory[i]),
                    history_length=valid_lens[i],
                    action_history=memory_contexts[i],
                    current_step=len(self.memory[i]) + 1,
                    current_observation=text_obs[i],
                    admissible_actions=reformatted_admissible_actions
                )

            postprocess_text_obs.append(obs)
        return postprocess_text_obs

    def _process_batch(self, batch_idx, total_batch_list, total_infos, success):
        # Find the last entry with active masks
        for i in reversed(range(len(total_batch_list[batch_idx]))):
            batch_item = total_batch_list[batch_idx][i]
            if batch_item['active_masks']:
                info = total_infos[batch_idx][i]
                won_value = float(info['won'])
                success['success_rate'].append(won_value)
                
                # Process game file if it exists
                gamefile = info.get("extra.gamefile")
                if gamefile:
                    self._process_gamefile(gamefile, won_value, success)
                return  # Exit after finding the first active mask

    def _process_gamefile(self, gamefile, won_value, success):
        tasks = [
            "pick_and_place",
            "pick_two_obj_and_place",
            "look_at_obj_in_light",
            "pick_heat_then_place_in_recep",
            "pick_cool_then_place_in_recep",
            "pick_clean_then_place_in_recep",
        ]
        
        for task in tasks:
            if task in gamefile:
                success[f"{task}_success_rate"].append(won_value)
                break


class SokobanEnvironmentManager(EnvironmentManagerBase):
    ACTION_LOOKUP = {
        0: "Still",
        1: "Up",
        2: "Down",
        3: "Left",
        4: "Right",
    }
    def __init__(self, envs, projection_f, config):
        self.is_multi_modal = envs.mode == 'rgb_array'
        self.memory = SimpleMemory()
        super().__init__(envs, projection_f, config)

    def reset(self, kwargs):
        obs, infos = self.envs.reset()
        if self.is_multi_modal:
            obs = np.array(obs, obs[0].dtype)
            self.pre_text_obs = self.envs.render(mode='tiny_rgb_array')
            observations = {
                'text': self.build_text_obs(infos, init=True), 
                'image': obs,   
                'anchor': obs
            }
        else:
            self.pre_text_obs = obs
            observations = {
                'text': self.build_text_obs(infos, obs, init=True),
                'image': None,
                'anchor': obs
            }
        self.memory.reset(batch_size = len(infos))
        return observations, infos

    def step(self, text_actions: List[str]):
        actions, valids = self.projection_f(text_actions)

        next_obs, rewards, dones, infos = self.envs.step(actions)

        for i, info in enumerate(infos):
            info['is_action_valid'] = to_numpy(valids[i])

        self.memory.store({'text_obs': self.pre_text_obs, 'action': [self.ACTION_LOOKUP[act] for act in actions]})
        if self.is_multi_modal:
            next_obs = np.array(next_obs, next_obs[0].dtype)
            self.pre_text_obs = self.envs.render(mode='tiny_rgb_array')
            next_observations = {
                'text': self.build_text_obs(infos),  
                'image': next_obs,
                'anchor': next_obs 
            }
        else:
            self.pre_text_obs = next_obs
            next_observations = {
                'text': self.build_text_obs(infos, next_obs),  
                'image': None, 
                'anchor': next_obs 
            }

        rewards = to_numpy(rewards)
        dones = to_numpy(dones)

        return next_observations, rewards, dones, infos

    def build_text_obs(self, infos, text_obs: List[str]=None, init: bool = False) -> List[str]:
        """
        This function builds the text observation for the agent.
        """
        postprocess_text_obs = []

        if not init and self.config.env.history_length > 0:
            memory_contexts, valid_lens = self.memory.fetch(
                    self.config.env.history_length,
                    obs_key="text_obs",
                    action_key="action")
            
        for i in range(len(infos)):
            if init or self.config.env.history_length <= 0:
                obs = SOKOBAN_VISUAL_TEMPLATE if self.is_multi_modal \
                 else SOKOBAN_TEMPLATE_NO_HIS.format(
                    current_observation=text_obs[i],
                )
            else:
                if self.is_multi_modal:
                    obs = SOKOBAN_VISUAL_TEMPLATE
                else:
                    obs = SOKOBAN_TEMPLATE.format(
                        step_count=len(self.memory[i]),
                        history_length=valid_lens[i],
                        action_history=memory_contexts[i],
                        current_step=len(self.memory[i]) + 1,
                        current_observation=text_obs[i],
                    )
            postprocess_text_obs.append(obs)

        return postprocess_text_obs


class GymCardEnvironmentManager(EnvironmentManagerBase):
    def __init__(self, envs, projection_f, config):
        super().__init__(envs, projection_f, config)
    
    def reset(self, kwargs) -> Dict[str, Any]:
        obs, infos = self.envs.reset()
        # infos = [None] * self.envs.num_envs
        observations = {'text': self.build_text_obs(infos), 'image': obs, 'anchor': obs.copy()}
        
        return observations, infos

    def step(self, text_actions: List[str]):
        next_observations, rewards, dones, infos = super().step(text_actions)
        
        # add text observation to next_observations
        next_observations['text'] = self.build_text_obs(infos)
        next_observations['anchor'] = next_observations['image'].copy()

        return next_observations, rewards, dones, infos


    def build_text_obs(self, infos: Tuple[Dict]=None) -> List[str]:
        """
        This function builds the text observation for the agent.
        """
        postprocess_text_obs = []
        for i in range(len(infos)):
            if 'ezpoints' in self.config.env.env_name.lower():
                text_formula = ''.join(str(element) for element in infos[i]['Formula']) if infos[i] is not None else ''
                obs = GYM_CARDS_EZPOINTS_TEMPLATE.format(text_formula=text_formula)
            elif 'points24' in self.config.env.env_name.lower():
                text_formula = ''.join(str(element) for element in infos[i]['Formula']) if infos[i] is not None else ''
                obs = GYM_CARDS_POINTS24_TEMPLATE.format(text_formula=text_formula)
            elif 'numberline' in self.config.env.env_name.lower():
                obs = GYM_CARDS_NUMBERLINE_TEMPLATE
            elif "blackjack" in self.config.env.env_name.lower():
                obs = GYM_CARDS_BLACKJACK_TEMPLATE
            else:
                raise ValueError(f"Unsupported environment: {self.config.env.env_name}")
            postprocess_text_obs.append(obs)
        return postprocess_text_obs


class WebshopEnvironmentManager(EnvironmentManagerBase):
    def __init__(self, envs, projection_f, config):
        self.memory = SimpleMemory()
        super().__init__(envs, projection_f, config)
    
    def reset(self, kwargs) -> Dict[str, Any]:
        obs, infos = self.envs.reset()
        self.tasks = self.extract_task(obs)
        obs = self.format_obs(obs)
        # infos = [None] * self.envs.num_envs
        observations = {'text': self.build_text_obs(obs, infos, init=True), 
                        'image': None, 
                        'anchor': obs.copy()
                        }
        self.pre_text_obs = obs
        self.memory.reset(batch_size = len(infos))
        return observations, infos

    def step(self, text_actions: List[str]):
        actions, valids = self.projection_f(text_actions)
        next_obs, rewards, dones, infos = self.envs.step(actions)

        next_obs = self.format_obs(next_obs)

        self.memory.store({'text_obs': self.pre_text_obs, 'action': actions})
        self.pre_text_obs = next_obs

        next_observations = {
            'text': self.build_text_obs(next_obs, infos),
            'image': None,
            'anchor': next_obs.copy()
        }
        # add action_valid to infos
        for i, info in enumerate(infos):
            info['is_action_valid'] = to_numpy(valids[i])

        rewards = to_numpy(rewards)
        dones = to_numpy(dones)

        return next_observations, rewards, dones, infos

    def extract_task(self, text_obs: List[str]):
        tasks = []
        for obs in text_obs:
            parts = obs.split(" [SEP] ")
            assert parts[1]=='Instruction:'
            tasks.append(parts[2])
        return tasks
    
    def format_obs(self, text_obs):
        postprocess_text_obs = []
        for i in range(len(text_obs)):
            parts = text_obs[i].split(" [SEP] ")
            # the index of self.tasks[i] in parts
            try:
                index = parts.index(self.tasks[i])
                reformatted_obs = " [SEP] ".join(f"'{p}'" for p in parts[index+1:])
            except:
                reformatted_obs = text_obs[i]

            postprocess_text_obs.append(reformatted_obs)

        return postprocess_text_obs
    
    def format_avail_actions(self, avail):
        actions = []

        for key in avail.keys():
            if key not in ["has_search_bar", "clickables"]:
                raise ValueError(f"Unknown key in available actions: {key}")

        if avail["has_search_bar"]:
            actions.append("search[<your query>]")

        for txt in avail["clickables"]:
            actions.append(f"click[{txt}]")

        return actions
            
    def build_text_obs(self, text_obs: List[str], infos: List[List[str]], init: bool = False) -> List[str]:
        """
        This function builds the text observation for the agent.
        """
        postprocess_text_obs = []
        if not init and self.config.env.history_length > 0:
            memory_contexts, valid_lens = self.memory.fetch(
                    self.config.env.history_length,
                    obs_key="text_obs",
                    action_key="action")
            
        for i in range(len(text_obs)):
            
            available_actions = self.format_avail_actions(infos[i]['available_actions'])
            reformatted_available_actions = "\n".join(f"'{s}'," for s in available_actions)

            if init or self.config.env.history_length <= 0:
                obs = WEBSHOP_TEMPLATE_NO_HIS.format(
                    task_description=self.tasks[i],
                    current_observation=text_obs[i],
                    available_actions=reformatted_available_actions
                )
            else:
                obs = WEBSHOP_TEMPLATE.format(
                    task_description=self.tasks[i],
                    step_count=len(self.memory[i]),
                    history_length=valid_lens[i],
                    action_history=memory_contexts[i],
                    current_step=len(self.memory[i]) + 1,
                    current_observation=text_obs[i],
                    available_actions=reformatted_available_actions
                )
                if len(obs) > 13000:
                    print(f"Warning len(obs)={len(obs)} is too long")
                    obs = WEBSHOP_TEMPLATE_NO_HIS.format(
                        task_description=self.tasks[i],
                        current_observation=text_obs[i],
                        available_actions=reformatted_available_actions
                    )

            postprocess_text_obs.append(obs)

        return postprocess_text_obs

    def _process_batch(self, batch_idx, total_batch_list, total_infos, success):
        for i in reversed(range(len(total_batch_list[batch_idx]))):
            batch_item = total_batch_list[batch_idx][i]
            if batch_item['active_masks']:
                info = total_infos[batch_idx][i]
                won_value = float(info['won'])
                score_value = float(info['task_score'])
                success['success_rate'].append(won_value)
                success['webshop_task_score (not success_rate)'].append(score_value)
                return

class AppWorldEnvironmentManager(EnvironmentManagerBase):
    def __init__(self, envs, projection_f, config):
        self.memory = SimpleMemory()
        super().__init__(envs, projection_f, config)
    
    def reset(self, kwargs):
        text_obs, infos = self.envs.reset()
        
        self.supervisors = [info['supervisor'] for info in infos]
        self.memory.reset(batch_size = len(text_obs))
        self.tasks = text_obs.copy()
        self.pre_text_obs = text_obs

        full_text_obs = self.build_text_obs(text_obs, init=True)
        return {'text': full_text_obs, 'image': None, 'anchor': text_obs}, infos
    
    def step(self, text_actions: List[str]):
        actions, valids = self.projection_f(text_actions)

        text_obs, rewards, dones, infos = self.envs.step(actions)

        self.memory.store({'text_obs': text_obs, 'action': actions})
        self.pre_text_obs = text_obs

        full_text_obs = self.build_text_obs(text_obs)

        # add action_valid to infos
        for i, info in enumerate(infos):
            info['is_action_valid'] = to_numpy(valids[i])

        next_observations = {'text': full_text_obs, 'image': None, 'anchor': text_obs}
        rewards = to_numpy(rewards)
        dones = to_numpy(dones)

        return next_observations, rewards, dones, infos
    

    def build_text_obs(self, text_obs: List[str], init: bool = False) -> List[str]:
        """
        This function builds the text observation for the agent.
        """
        postprocess_text_obs = []
        if init and self.supervisors is not None:
            for i in range(len(text_obs)):
                obs = APPWORLD_TEMPLATE_NO_HIS.format(
                        supervisor_first_name=self.supervisors[i]['first_name'],
                        supervisor_last_name=self.supervisors[i]['last_name'],
                        supervisor_email=self.supervisors[i]['email'],
                        supervisor_phone_number=self.supervisors[i]['phone_number'],
                        task_description=self.tasks[i],
                    )
                postprocess_text_obs.append(obs)
        else:
            for i in range(len(text_obs)):
                # Get last `history_length` steps
                recent_history = self.memory[i][-self.config.env.history_length:]
                valid_history_length = len(recent_history)
                start_index = len(self.memory[i]) - valid_history_length
                action_history = ""
                for j, record in enumerate(recent_history):
                    step_number = start_index + j + 1
                    action = record["action"]
                    env_obs = record["text_obs"]
                    action_history += f"\nCode {step_number}: \n{action}\n\nResult {step_number}: \n{env_obs}\n"
                
                if len(action_history) > 10000:
                    action_history = "... " + action_history[-10000:]

                obs = APPWORLD_TEMPLATE.format(
                        supervisor_first_name=self.supervisors[i]['first_name'],
                        supervisor_last_name=self.supervisors[i]['last_name'],
                        supervisor_email=self.supervisors[i]['email'],
                        supervisor_phone_number=self.supervisors[i]['phone_number'],
                        task_description=self.tasks[i],
                        step_count=len(self.memory[i]),
                        history_length=valid_history_length,
                        action_history=action_history.strip(),
                        current_step=len(self.memory[i]) + 1,
                        current_observation=text_obs[i],
                    )
                postprocess_text_obs.append(obs)
        return postprocess_text_obs


RRG_LANG_NAME = {"en": "English", "zh": "Chinese", "es": "Spanish", "fr": "French",
                 "de": "German", "ja": "Japanese", "ko": "Korean", "pt": "Portuguese"}


class RRGEnvironmentManager(EnvironmentManagerBase):
    """Reverse-reasoning REPLAY manager (AMEX navigation toy).

    The policy generates the per-step REASONING (it is shown the goal, history of its OWN
    prior reasonings, the GT next action, and the marked screenshot). The action is replayed,
    so this manager's job each step is: (1) score the action-recovery MARGIN of the generated
    reasoning with the frozen 8B reader (the GiGPO step/micro reward); (2) apply the
    leakage/restatement VETO (reasoning that restates the GT coordinates -> invalid ->
    invalid_action_penalty); (3) thread the generated reasoning forward as own-notes history;
    (4) advance the replay. anchor = "task_id:step_idx" -> exact GiGPO step grouping.
    The trajectory/macro reward (completion judge) is computed post-rollout in the reward manager.
    """

    def __init__(self, envs, projection_f, config, tokenizer=None, processor=None):
        self.is_multi_modal = True
        self.memory = SimpleMemory()
        self.history = None  # per-slot list of generated reasonings (SFT 'Step k:' format)
        self.cur_frames = None

        rcfg = config.env.rrg
        self.rcfg = rcfg
        self._tokenizer = tokenizer
        self._processor = processor
        self._self_judge_wg = None  # set later by set_self_judge_wg() if self_judge=True
        from agent_system.environments.env_package.rrg.reward_client import RRGRewardClient, action_str
        self._action_str = action_str
        if not rcfg.get("self_judge", False):
            self.reward_client = RRGRewardClient(
                base_url=rcfg.reader_url, model_name=rcfg.reader_model,
                max_image_long=rcfg.get("max_image_long", 768),
                num_distractors=rcfg.get("num_distractors", 4),
                concurrency=rcfg.get("concurrency", 64),
                subtract_control=rcfg.get("subtract_control", False))
        else:
            self.reward_client = None  # created lazily in set_self_judge_wg()
        self.coord_tol = rcfg.get("coord_tol", 8)
        # Val answer-checker: only the val replay env (is_train=False) computes terminal
        # answer-recovery correctness for success_rate; train relies on the reward manager's
        # recall, so we skip the extra reader call there.
        self.data_kind = rcfg.get("data_kind", "amex")
        self.answer_check = (self.data_kind == "rrg") and (not getattr(envs, "is_train", True))
        self.answer_max_tokens = rcfg.get("answer_max_tokens", 2048)
        # The val answer-check may use a separate, stronger/slower reader (e.g. doubao via Ark)
        # for a more accurate success metric. Step margins ALWAYS stay on reward_client (the 8B):
        # the margin needs the logprobs path, which the doubao reasoning model does not expose.
        self.answer_client = self.reward_client
        if self.answer_check and rcfg.get("val_reader_url", None):
            self.answer_client = RRGRewardClient(
                base_url=rcfg.val_reader_url,
                model_name=rcfg.get("val_reader_model", None) or rcfg.reader_model,
                concurrency=rcfg.get("concurrency", 64),
                # key from env (RRG_VAL_READER_KEY) so it stays out of the logged/uploaded config.
                api_key=(os.environ.get("RRG_VAL_READER_KEY")
                         or rcfg.get("val_reader_key", None) or "sk-dummy"))
            self.answer_max_tokens = rcfg.get("val_answer_max_tokens", None) or self.answer_max_tokens

        self.system_prompt = ""
        spf = rcfg.get("system_prompt_file", None)
        if spf and os.path.isfile(spf):
            with open(spf, encoding="utf-8") as f:
                self.system_prompt = f.read().strip()

        # global hard-distractor pool: every GT action string across the loaded episodes
        self.action_pool = sorted({action_str(fr["action"])
                                   for ep in envs.episodes for fr in ep["frames"]})

        # Teacher-demo injection (EXPLORATION): seed the first `teacher_seed_k` slots of each group of
        # `group_n` with a verified teacher trajectory (task_id 'env-tid' -> [per-step reasoning]). The
        # rollout loop forces these into the response, so the (high-reward) teacher trajectory enters its
        # GiGPO group and pulls the policy toward note-taking it cannot produce on its own. TRAIN only,
        # off by default. Group size = config.env.rollout.n (matches the uid grouping in rollout_loop).
        self.teacher_store = {}
        tdp = rcfg.get("teacher_data_path", None)
        if tdp and os.path.isfile(tdp):
            with open(tdp, encoding="utf-8") as f:
                self.teacher_store = json.load(f)
        self.teacher_seed_k = int(rcfg.get("teacher_seed_k", 0))
        self.teacher_anneal_end_step = rcfg.get("teacher_anneal_end_step", None)
        self.group_n = int(getattr(config.env.rollout, "n", 0)) or int(getattr(envs, "group_n", 1))
        self.is_train_env = bool(getattr(envs, "is_train", True))
        self._reset_count = 0
        # Async MC pipeline: when self_judge is on and reward_client supports async submit,
        # step() submits the MC RPC and returns zeros; the rollout loop calls
        # flush_pending_mc() at the start of the next iteration to get the prior step's
        # margins and add them to episode_rewards. The first iteration bootstraps synchronously
        # so episode_rewards[0] = MC(reasonings[0]) is accurate.
        self._pending_mc = None
        self._step_count = 0
        # Async MC is enabled lazily: at __init__ time the reward_client is None for the
        # self-judge path (it gets injected later via set_self_judge_wg). Decide here
        # whether to *attempt* async and re-check at every step() call (cheap hasattr).
        self._async_mc_eligible = bool(rcfg.get("self_judge", False))
        if self.teacher_store and self.teacher_seed_k > 0 and self.is_train_env:
            print(f"[rrg-teacher] loaded {len(self.teacher_store)} teacher trajectories; seeding "
                  f"{self.teacher_seed_k}/{self.group_n} slots per group"
                  + (f"; anneal off after {self.teacher_anneal_end_step} steps"
                     if self.teacher_anneal_end_step else ""))
        super().__init__(envs, projection_f, config)

    def set_self_judge_wg(self, actor_rollout_wg):
        """Post-init injection: replace the HTTP reward client with a SelfJudgeClient
        that uses the policy's own vLLM engine (via actor_rollout_wg)."""
        self._self_judge_wg = actor_rollout_wg
        if self.rcfg.get("self_judge", False):
            from agent_system.environments.env_package.rrg.self_judge_client import SelfJudgeClient
            self.reward_client = SelfJudgeClient(
                tokenizer=self._tokenizer, processor=self._processor,
                actor_rollout_wg=actor_rollout_wg,
                config=self.rcfg)
            if self.answer_check and self.rcfg.get("val_reader_url", None) is None:
                self.answer_client = self.reward_client

    # ----- helpers ----- #
    def _load_img(self, path):
        return np.array(Image.open(path).convert("RGB"), dtype=np.uint8)

    def _anchor(self, fr):
        return f"{fr['task_id']}:{fr['step_idx']}"

    def _coord_leak(self, text, action):
        """Leakage veto: reasoning restates the GT click coordinate (within coord_tol px)."""
        if action.get("action") != "click":
            return False
        gx, gy = action["coordinate"]
        for m in re.finditer(r"\(?\s*(\d{2,4})\s*[,/x ]\s*(\d{2,4})\s*\)?", text or ""):
            x, y = int(m.group(1)), int(m.group(2))
            if abs(x - gx) <= self.coord_tol and abs(y - gy) <= self.coord_tol:
                return True
        return False

    def _forced_responses(self, frames):
        """Per-slot teacher text for teacher-seeded slots (TRAIN only), else None per slot. Returns None
        (whole list) when teacher injection is inactive, so the rollout-loop splice is a pure no-op."""
        if not (self.teacher_store and self.teacher_seed_k > 0 and self.is_train_env):
            return None
        if self.teacher_anneal_end_step is not None and self._reset_count >= self.teacher_anneal_end_step:
            return None
        out = []
        for i, fr in enumerate(frames):
            text = None
            if (i % self.group_n) < self.teacher_seed_k:           # first k slots of each uid group
                traj = self.teacher_store.get(str(fr["task_id"]))   # verified tasks only
                si = fr["step_idx"]
                if traj is not None and si < len(traj):
                    text = traj[si]
            out.append(text)
        return out

    def _obs(self, frames):
        obs = {
            'text': self.build_text_obs(frames),
            'image': [self._load_img(fr["image_path"]) for fr in frames],
            'anchor': [self._anchor(fr) for fr in frames],
        }
        forced = self._forced_responses(frames)
        if forced is not None:
            obs['forced_response'] = forced
        return obs

    # ----- gym-like API ----- #
    def reset(self, kwargs):
        obs_list, infos = self.envs.reset()
        self.cur_frames = obs_list
        self.history = [[] for _ in range(len(obs_list))]
        self.memory.reset(batch_size=len(obs_list))
        self._reset_count += 1  # ~one reset per training step; drives teacher anneal
        obs = self._obs(obs_list)
        if obs.get('forced_response'):
            n_teacher = sum(1 for x in obs['forced_response'] if x)
            print(f"[rrg-teacher] reset {self._reset_count}: {n_teacher} teacher-forced slots", flush=True)
        return obs, infos

    def flush_pending_mc(self):
        """Await the previously-submitted async MC scoring RPC and return its per-item
        margins, or None if nothing was pending. Called by the rollout loop at the
        start of each iteration so episode_rewards keeps correct per-step accounting
        while the MC RPC overlaps the policy rollout. Resets the pending slot."""
        if self._pending_mc is None:
            return None
        pending, self._pending_mc = self._pending_mc, None
        try:
            return pending.get()
        except Exception as e:
            print(f"[rrg-self-judge] pending MC .get() failed: {type(e).__name__}: {e}", flush=True)
            return [0.0] * len(self.cur_frames) if self.cur_frames else [0.0]

    def step(self, text_actions: List[str]):
        text_actions = list(text_actions)
        actions, valids = self.projection_f(text_actions)
        frames = self.cur_frames

        # (1) per-step action-recovery margin (reader SEES the current screenshot).
        # Async path: submit the MC RPC and return placeholder zeros so the rollout loop
        # can keep running. The next iteration's flush_pending_mc() retrieves the
        # actual margins. Bootstrap iteration (step_count==0) runs synchronously to
        # seed episode_rewards[0] correctly.
        items = [{"goal": fr["goal"], "image": fr["image_path"], "action": fr["action"],
                  "reasoning": text_actions[i]} for i, fr in enumerate(frames)]
        if self._async_mc_eligible and self._step_count > 0 and hasattr(self.reward_client, "submit_score_step_margins"):
            self._pending_mc = self.reward_client.submit_score_step_margins(items, self.action_pool)
            rewards = np.zeros(len(items), dtype=np.float32)
        else:
            rewards = np.asarray(self.reward_client.score_step_margins(items, self.action_pool),
                                 dtype=np.float32)
        self._step_count += 1

        # (2) leakage veto folded into validity
        for i, fr in enumerate(frames):
            leaked = self._coord_leak(text_actions[i], fr["action"])
            valids[i] = 1 if (valids[i] and not leaked) else 0

        # (3) own-notes history forward
        for i in range(len(frames)):
            self.history[i].append(text_actions[i].strip())

        # (4) advance the replay
        next_obs_list, _, dones, infos = self.envs.step(actions)
        self.cur_frames = next_obs_list
        for i, info in enumerate(infos):
            info['is_action_valid'] = to_numpy(valids[i])

        # (5) val answer-checker: on a finished episode, assemble the answer from the policy's
        # own reasonings (blind reader) and mark won iff it fully recovers gold. Train skips
        # this (success not used there); the reward manager handles recall for the macro reward.
        if self.answer_check and any(bool(d) for d in dones):
            done_idx = [i for i, d in enumerate(dones) if bool(d)]
            items = []
            for i in done_idx:
                fr = frames[i]  # frame just acted on carries this task's goal/gold/schema
                items.append({"goal": fr["goal"], "reasonings": list(self.history[i]),
                              "gold": fr.get("gold"), "schema": fr.get("schema")})
            scorable = [j for j, it in enumerate(items) if it["gold"] and it["schema"]]
            if scorable:
                scored = self.answer_client.score_traj_recovery(
                    [items[j] for j in scorable], max_tokens=self.answer_max_tokens)
                res_by_local = {j: r for j, r in zip(scorable, scored)}
                for local, i in enumerate(done_idx):
                    r = res_by_local.get(local)
                    if r is not None:
                        infos[i]['won'] = bool(r["correct"])

        return self._obs(next_obs_list), to_numpy(rewards), to_numpy(dones), infos

    def build_text_obs(self, frames, *args, **kwargs) -> List[str]:
        """Match the SFT prompt (prompts/system_amex.txt + amex_to_sft.build_user_text) so the
        SFT-initialized policy stays in-distribution. Own prior reasonings form the history."""
        out = []
        for i, fr in enumerate(frames):
            hist = self.history[i] if self.history else []
            hist_str = "\n".join(f"Step {k + 1}: {r}" for k, r in enumerate(hist)) \
                or "(none yet -- this is the first step)"
            lang = RRG_LANG_NAME.get(fr.get("lang", "en"), "English")
            user = (
                f"# Task goal\n{fr['goal']}\n\n"
                f"# Your reasoning in previous steps\n{hist_str}\n\n"
                f"# Ground-truth next action\n{json.dumps(fr['action'], ensure_ascii=False)}\n\n"
                f"# Output language\nWrite the reasoning chain in {lang}, to match the app.\n\n"
                "<image>"
            )
            out.append((self.system_prompt + "\n\n" + user) if self.system_prompt else user)
        return out


def make_envs(config, tokenizer=None, processor=None):
    """
    Create enviroments 
    """ 
    # check if config.env.rollout.n is an integer
    if not isinstance(config.env.rollout.n, int):
        raise ValueError("config.env.rollout.n should be an integer")
    group_n = config.env.rollout.n if config.env.rollout.n > 0 else 1
    resources_per_worker = OmegaConf.to_container(config.env.resources_per_worker, resolve=True)

    if "search" in config.env.env_name.lower():
        from agent_system.environments.env_package.search import build_search_envs, search_projection
        _envs = build_search_envs(seed=config.env.seed, env_num=config.data.train_batch_size, group_n=group_n, is_train=True, env_config=config.env)
        _val_envs = build_search_envs(seed=config.env.seed + 1000, env_num=config.data.val_batch_size, group_n=1, is_train=False, env_config=config.env)

        projection_f = partial(search_projection)
        envs = SearchEnvironmentManager(_envs, projection_f, config)
        val_envs = SearchEnvironmentManager(_val_envs, projection_f, config)
        return envs, val_envs
    elif "gym_cards" in config.env.env_name.lower():
        from agent_system.environments.env_package.gym_cards import build_gymcards_envs, gym_projection
        _envs = build_gymcards_envs(env_name=config.env.env_name, seed=config.env.seed, env_num=config.data.train_batch_size, group_n=group_n, is_train=True, resources_per_worker=resources_per_worker)
        _val_envs = build_gymcards_envs(env_name=config.env.env_name, seed=config.env.seed + 1000, env_num=config.data.val_batch_size, group_n=1, is_train=False, resources_per_worker=resources_per_worker)
        
        projection_f = partial(gym_projection, env_name=config.env.env_name)
        envs = GymCardEnvironmentManager(_envs, projection_f, config)
        val_envs = GymCardEnvironmentManager(_val_envs, projection_f, config)
        return envs, val_envs
    elif "alfworld" in config.env.env_name.lower():
        from agent_system.environments.env_package.alfworld import build_alfworld_envs, alfworld_projection
        if config.env.env_name == 'alfworld/AlfredThorEnv':
            alf_config_path = os.path.join(os.path.dirname(__file__), 'env_package/alfworld/configs/config_tw.yaml')
        elif config.env.env_name == 'alfworld/AlfredTWEnv':
            alf_config_path = os.path.join(os.path.dirname(__file__), 'env_package/alfworld/configs/config_tw.yaml')
        else:
            raise ValueError(f"Unsupported environment: {config.env.env_name}")

        env_kwargs = {
            'eval_dataset': config.env.alfworld.eval_dataset, # 'eval_in_distribution' or 'eval_out_of_distribution'
        }
        _envs = build_alfworld_envs(alf_config_path, config.env.seed, config.data.train_batch_size, group_n, is_train=True, env_kwargs=env_kwargs, resources_per_worker=resources_per_worker)
        _val_envs = build_alfworld_envs(alf_config_path, config.env.seed + 1000, config.data.val_batch_size, 1, is_train=False, env_kwargs=env_kwargs, resources_per_worker=resources_per_worker)
        
        projection_f = partial(alfworld_projection)
        envs = AlfWorldEnvironmentManager(_envs, projection_f, config)
        val_envs = AlfWorldEnvironmentManager(_val_envs, projection_f, config)
        return envs, val_envs
    elif "sokoban" in config.env.env_name.lower():
        from agent_system.environments.env_package.sokoban import build_sokoban_envs, sokoban_projection
        env_kwargs = {
            'dim_room': config.env.sokoban.dim_room,
            'num_boxes': config.env.sokoban.num_boxes,
            'max_steps': config.env.max_steps,
            'search_depth': config.env.sokoban.search_depth
        }
        _envs = build_sokoban_envs(config.env.seed, config.data.train_batch_size, group_n, mode=config.env.sokoban.mode, is_train=True, env_kwargs=env_kwargs, resources_per_worker=resources_per_worker)
        _val_envs = build_sokoban_envs(config.env.seed + 1000, config.data.val_batch_size, 1, mode=config.env.sokoban.mode, is_train=False, env_kwargs=env_kwargs, resources_per_worker=resources_per_worker)
        
        projection_f = partial(sokoban_projection)
        envs = SokobanEnvironmentManager(_envs, projection_f, config)
        val_envs = SokobanEnvironmentManager(_val_envs, projection_f, config)
        return envs, val_envs
    elif "webshop" in config.env.env_name.lower():
        from agent_system.environments.env_package.webshop import build_webshop_envs, webshop_projection
        if config.env.webshop.use_small:
            file_path = os.path.join(os.path.dirname(__file__), 'env_package/webshop/webshop/data/items_shuffle_1000.json')
            attr_path = os.path.join(os.path.dirname(__file__), 'env_package/webshop/webshop/data/items_ins_v2_1000.json')
        else:
            file_path = os.path.join(os.path.dirname(__file__), 'env_package/webshop/webshop/data/items_shuffle.json')
            attr_path = os.path.join(os.path.dirname(__file__), 'env_package/webshop/webshop/data/items_ins_v2.json')
        env_kwargs = {
                    'observation_mode': 'text', 
                    'num_products': None, 
                    'human_goals': config.env.webshop.human_goals,
                    'file_path': file_path,
                    'attr_path': attr_path
                    }
        _envs = build_webshop_envs(seed=config.env.seed, env_num=config.data.train_batch_size, group_n=group_n, is_train=True, env_kwargs=env_kwargs, resources_per_worker=resources_per_worker)
        _val_envs = build_webshop_envs(seed=config.env.seed + 1000, env_num=config.data.val_batch_size, group_n=1, is_train=False, env_kwargs=env_kwargs, resources_per_worker=resources_per_worker)

        projection_f = partial(webshop_projection)
        envs = WebshopEnvironmentManager(_envs, projection_f, config)
        val_envs = WebshopEnvironmentManager(_val_envs, projection_f, config)
        import time
        time.sleep((config.data.train_batch_size * group_n + config.data.val_batch_size) * 0.1) # wait for the envs to be ready
        return envs, val_envs
    elif "appworld" in config.env.env_name.lower():
        from agent_system.environments.env_package.appworld import build_appworld_envs, appworld_projection
        _envs = build_appworld_envs(dataset_name='train', seed=config.env.seed, env_num=config.data.train_batch_size, group_n=group_n, start_server_id=0, resources_per_worker=resources_per_worker)
        _val_envs = build_appworld_envs(dataset_name='test_normal', seed=config.env.seed + 1000, env_num=config.data.val_batch_size, group_n=1, start_server_id=config.data.train_batch_size*group_n, resources_per_worker=resources_per_worker)
        
        projection_f = partial(appworld_projection)
        envs = AppWorldEnvironmentManager(_envs, projection_f, config)
        val_envs = AppWorldEnvironmentManager(_val_envs, projection_f, config)
        return envs, val_envs
    elif "rrg" in config.env.env_name.lower():
        from agent_system.environments.env_package.rrg import build_rrg_envs, rrg_projection
        rcfg = config.env.rrg
        data_kind = rcfg.get('data_kind', 'amex')
        if data_kind == 'rrg':
            # Native RRG: answer-bearing tasks served from a task_root directory tree.
            env_kwargs = {
                'data_kind': 'rrg',
                'task_root': rcfg.train_task_root,
                'num_episodes': rcfg.get('num_episodes', None),
            }
            val_task_root = rcfg.get('val_task_root', None) or rcfg.train_task_root
            val_num_eps = rcfg.get('val_num_episodes', None)
            if val_num_eps is None:
                val_num_eps = rcfg.get('num_episodes', None)
            val_kwargs = {
                'data_kind': 'rrg',
                'task_root': val_task_root,
                'num_episodes': val_num_eps,
            }
        else:
            env_kwargs = {
                'data_kind': 'amex',
                'data_jsonl': rcfg.train_jsonl,
                'image_root': rcfg.train_image_root,
                'num_episodes': rcfg.get('num_episodes', None),
            }
            # NB: OmegaConf .get() returns the stored value even when it is null, so use explicit
            # None-fallbacks (val_* default to null in the yaml -> fall back to the train values).
            val_jsonl = rcfg.get('val_jsonl', None) or rcfg.train_jsonl
            val_image_root = rcfg.get('val_image_root', None) or rcfg.train_image_root
            val_num_eps = rcfg.get('val_num_episodes', None)
            if val_num_eps is None:
                val_num_eps = rcfg.get('num_episodes', None)
            val_kwargs = {
                'data_kind': 'amex',
                'data_jsonl': val_jsonl,
                'image_root': val_image_root,
                'num_episodes': val_num_eps,
            }
        _envs = build_rrg_envs(seed=config.env.seed, env_num=config.data.train_batch_size,
                               group_n=group_n, is_train=True, env_kwargs=env_kwargs)
        _val_envs = build_rrg_envs(seed=config.env.seed + 1000, env_num=config.data.val_batch_size,
                                   group_n=1, is_train=False, env_kwargs=val_kwargs)
        projection_f = partial(rrg_projection)
        envs = RRGEnvironmentManager(_envs, projection_f, config,
                                     tokenizer=tokenizer, processor=processor)
        val_envs = RRGEnvironmentManager(_val_envs, projection_f, config,
                                         tokenizer=tokenizer, processor=processor)
        return envs, val_envs
    else:
        print("Environment not supported")
        exit(1)