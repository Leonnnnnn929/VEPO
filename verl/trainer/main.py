# Copyright 2024 Bytedance Ltd. and/or its affiliates
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
"""
Note that we don't combine the main with ray_trainer as ray_trainer is used by other main.
"""

# Monkey-patch torch.library.wrap_triton for PyTorch < 2.6 compatibility.
# wrap_triton was introduced in PyTorch 2.6; in 2.5 it does not exist but
# some libraries (e.g. vLLM, flash-attn) may reference it.  We simply make
# it an identity function so the original triton kernel is returned as-is.
import torch
import torch.library
if not hasattr(torch.library, "wrap_triton"):
    torch.library.wrap_triton = lambda fn: fn

import json

import ray
from omegaconf import OmegaConf

from ..single_controller.ray import RayWorkerGroup
from ..utils.tokenizer import get_processor, get_tokenizer
from ..workers.fsdp_workers import FSDPWorker
from ..workers.reward import CustomRewardManager
from .config import PPOConfig
from .ray_trainer import RayPPOTrainer, ResourcePoolManager, Role


@ray.remote(num_cpus=1)
def main_task(config: PPOConfig):
    # please make sure main_task is not scheduled on head
    # print config
    config.deep_post_init()
    print(json.dumps(config.to_dict(), indent=2))

    # instantiate tokenizer
    tokenizer = get_tokenizer(
        config.worker.actor.model.model_path,
        trust_remote_code=config.worker.actor.model.trust_remote_code,
        use_fast=True,
    )
    processor = get_processor(
        config.worker.actor.model.model_path,
        trust_remote_code=config.worker.actor.model.trust_remote_code,
        use_fast=True,
    )

    # define worker classes
    ray_worker_group_cls = RayWorkerGroup
    role_worker_mapping = {
        Role.ActorRollout: ray.remote(FSDPWorker),
        Role.Critic: ray.remote(FSDPWorker),
        Role.RefPolicy: ray.remote(FSDPWorker),
    }
    global_pool_id = "global_pool"
    resource_pool_spec = {
        global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
    }
    mapping = {
        Role.ActorRollout: global_pool_id,
        Role.Critic: global_pool_id,
        Role.RefPolicy: global_pool_id,
    }
    resource_pool_manager = ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=mapping)

    reward_fn = CustomRewardManager(
        tokenizer=tokenizer, num_examine=1, compute_score=config.worker.reward.compute_score
    )
    val_reward_fn = CustomRewardManager(
        tokenizer=tokenizer, num_examine=1, compute_score="pure_math"
    )

    trainer = RayPPOTrainer(
        config=config,
        tokenizer=tokenizer,
        processor=processor,
        role_worker_mapping=role_worker_mapping,
        resource_pool_manager=resource_pool_manager,
        ray_worker_group_cls=ray_worker_group_cls,
        reward_fn=reward_fn,
        val_reward_fn=val_reward_fn,
    )
    trainer.init_workers()
    trainer.fit()


def main():
    cli_args = OmegaConf.from_cli()
    file_config = OmegaConf.load(getattr(cli_args, "config"))
    cli_args.pop("config", None)

    default_config = OmegaConf.structured(PPOConfig())
    ppo_config = OmegaConf.merge(default_config, file_config, cli_args)
    ppo_config = OmegaConf.to_object(ppo_config)

    if not ray.is_initialized():
        # Multi-node: connect to a pre-started Ray cluster (chief runs `ray start --head`,
        # workers run `ray start --address=<chief>:6379`) by setting RAY_ADDRESS=auto in env
        # before launching this script. Single-node falls back to a fresh local cluster.
        import os as _os
        _ray_addr = _os.environ.get("RAY_ADDRESS")
        _init_kwargs = dict(runtime_env={"env_vars": {"TOKENIZERS_PARALLELISM": "true", "NCCL_DEBUG": "WARN"}})
        if _ray_addr:
            # Connecting to an existing cluster — don't pass cluster-shape args here.
            _init_kwargs["address"] = _ray_addr
        else:
            # Local single-node bootstrap. On big boxes (e.g. 384-core hosts) Ray's
            # default num_cpus = nproc triggers a prestart storm of ~nproc python
            # workers; when the conda env lives on a shared filesystem
            # (/apdcephfs_jn/...) that storm chokes on torch/vLLM imports and
            # `ray.init` hangs forever in the worker registration step. Cap CPUs
            # so Ray only prestarts a manageable pool. Override via RAY_NUM_CPUS.
            _local_cpus = int(_os.environ.get("RAY_NUM_CPUS", "64"))
            _init_kwargs["num_cpus"] = _local_cpus
        ray.init(**_init_kwargs)

    ray.get(main_task.remote(ppo_config))


if __name__ == "__main__":
    main()
