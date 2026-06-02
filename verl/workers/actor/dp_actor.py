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
Implement Actor
"""

import os
from collections import defaultdict
from typing import Any, Dict, Optional, Tuple

import torch
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from tqdm import tqdm

from ...protocol import DataProto
from ...trainer import core_algos
from ...utils import torch_functional as VF
from ...utils.py_functional import append_to_dict
from ...utils.ulysses import gather_outpus_and_unpad, ulysses_pad_and_slice_inputs
from .base import BasePPOActor
from .config import ActorConfig


try:
    from flash_attn.bert_padding import index_first_axis, pad_input, rearrange, unpad_input
except ImportError:
    pass


__all__ = ["DataParallelPPOActor"]


class DataParallelPPOActor(BasePPOActor):
    def __init__(
        self,
        config: ActorConfig,
        actor_module: nn.Module,
        actor_optimizer: Optional[torch.optim.Optimizer] = None,
    ):
        """
        When optimizer is None, it is Reference Policy
        """
        super().__init__(config)
        self.rank = int(os.getenv("RANK", "0"))
        self.actor_module = actor_module
        self.actor_optimizer = actor_optimizer
        self.compute_entropy_from_logits = torch.compile(VF.entropy_from_logits, dynamic=True)

    def _forward_micro_batch(
        self, micro_batch: Dict[str, torch.Tensor], temperature: float
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            entropy: # (bs, response_len)
            log_probs: # (bs, response_len)
        """
        input_ids = micro_batch["input_ids"]
        batch_size, seqlen = input_ids.shape
        attention_mask = micro_batch["attention_mask"]
        position_ids = micro_batch["position_ids"]
        responses = micro_batch["responses"]
        response_length = responses.size(-1)
        if position_ids.dim() == 3:  # qwen2vl mrope
            position_ids = position_ids.transpose(0, 1)  # (bsz, 3, seqlen) -> (3, bsz, seqlen)

        multi_modal_inputs = {}
        if "multi_modal_inputs" in micro_batch:
            for key in micro_batch["multi_modal_inputs"][0].keys():
                multi_modal_inputs[key] = torch.cat(
                    [inputs[key] for inputs in micro_batch["multi_modal_inputs"]], dim=0
                )

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            if self.config.padding_free:
                input_ids_rmpad, indices, *_ = unpad_input(
                    input_ids.unsqueeze(-1), attention_mask
                )  # input_ids_rmpad (total_nnz, ...)
                input_ids_rmpad = input_ids_rmpad.transpose(0, 1)  # (1, total_nnz)

                # unpad the position_ids to align the rotary
                if position_ids.dim() == 3:
                    position_ids_rmpad = (
                        index_first_axis(rearrange(position_ids, "c b s ... -> (b s) c ..."), indices)
                        .transpose(0, 1)
                        .unsqueeze(1)
                    )  # (3, bsz, seqlen) -> (3, 1, bsz * seqlen)
                else:
                    position_ids_rmpad = index_first_axis(
                        rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."), indices
                    ).transpose(0, 1)

                # for compute the log_prob
                input_ids_rmpad_rolled = torch.roll(input_ids_rmpad, shifts=-1, dims=1)  # (1, total_nnz)

                # pad and slice the inputs if sp > 1
                if self.config.ulysses_sequence_parallel_size > 1:
                    input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad_and_slice_inputs(
                        input_ids_rmpad, position_ids_rmpad, sp_size=self.config.ulysses_sequence_parallel_size
                    )
                    input_ids_rmpad_rolled, _, _ = ulysses_pad_and_slice_inputs(
                        input_ids_rmpad_rolled, None, self.config.ulysses_sequence_parallel_size
                    )

                input_ids_rmpad_rolled = input_ids_rmpad_rolled.squeeze(0)  # ((total_nnz / sp) + pad)

                # only pass input_ids and position_ids to enable flash_attn_varlen
                output = self.actor_module(
                    input_ids=input_ids_rmpad,
                    attention_mask=None,
                    position_ids=position_ids_rmpad,
                    **multi_modal_inputs,
                    use_cache=False,
                )  # prevent model thinks we are generating
                logits_rmpad = output.logits.squeeze(0)  # (total_nnz, vocab_size)

                logits_rmpad.div_(temperature)

                # compute entropy
                entropy_rmpad = self.compute_entropy_from_logits(logits_rmpad)  # ((total_nnz / sp) + pad)

                # if use_sp: ((total_nnz / sp) + pad) ; if not use_sp: (batch, seqlen)
                log_probs = VF.logprobs_from_logits(logits=logits_rmpad, labels=input_ids_rmpad_rolled)

                # gather log_prob if sp > 1
                if self.config.ulysses_sequence_parallel_size > 1:
                    # gather and unpad for the ulysses sp
                    log_probs = gather_outpus_and_unpad(log_probs, gather_dim=0, unpad_dim=0, padding_size=pad_size)
                    entropy_rmpad = gather_outpus_and_unpad(
                        entropy_rmpad, gather_dim=0, unpad_dim=0, padding_size=pad_size
                    )
                # pad back to (bsz, seqlen)
                full_entropy = pad_input(
                    hidden_states=entropy_rmpad.unsqueeze(-1), indices=indices, batch=batch_size, seqlen=seqlen
                )
                full_log_probs = pad_input(
                    hidden_states=log_probs.unsqueeze(-1), indices=indices, batch=batch_size, seqlen=seqlen
                )

                # only return response part
                entropy = full_entropy.squeeze(-1)[:, -response_length - 1 : -1]  # (bsz, response_length)
                log_probs = full_log_probs.squeeze(-1)[:, -response_length - 1 : -1]  # (bsz, response_length)
            else:
                output = self.actor_module(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    **multi_modal_inputs,
                    use_cache=False,
                )
                logits: torch.Tensor = output.logits
                logits.div_(temperature)
                logits = logits[:, -response_length - 1 : -1, :]  # (bsz, response_length, vocab_size)
                log_probs = VF.logprobs_from_logits(logits, responses)  # (bsz, response_length)
                entropy = VF.entropy_from_logits(logits)  # (bsz, response_length)

        return entropy, log_probs

    def _forward_micro_batch_logits(
        self, micro_batch: Dict[str, torch.Tensor], temperature: float
    ) -> torch.Tensor:
        """
        Forward pass that returns the full logits for the response part.

        Memory-efficient implementation: instead of padding logits back to
        (bsz, seqlen, vocab_size) which requires ~37 GiB for large batches,
        we directly extract response-part logits from the unpadded tensor.

        Returns:
            logits: (bs, response_length, vocab_size)
        """
        input_ids = micro_batch["input_ids"]
        batch_size, seqlen = input_ids.shape
        attention_mask = micro_batch["attention_mask"]
        position_ids = micro_batch["position_ids"]
        responses = micro_batch["responses"]
        response_length = responses.size(-1)
        if position_ids.dim() == 3:  # qwen2vl mrope
            position_ids = position_ids.transpose(0, 1)

        multi_modal_inputs = {}
        if "multi_modal_inputs" in micro_batch:
            for key in micro_batch["multi_modal_inputs"][0].keys():
                multi_modal_inputs[key] = torch.cat(
                    [inputs[key] for inputs in micro_batch["multi_modal_inputs"]], dim=0
                )

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            if self.config.padding_free:
                input_ids_rmpad, indices, *_ = unpad_input(
                    input_ids.unsqueeze(-1), attention_mask
                )
                input_ids_rmpad = input_ids_rmpad.transpose(0, 1)

                if position_ids.dim() == 3:
                    position_ids_rmpad = (
                        index_first_axis(rearrange(position_ids, "c b s ... -> (b s) c ..."), indices)
                        .transpose(0, 1)
                        .unsqueeze(1)
                    )
                else:
                    position_ids_rmpad = index_first_axis(
                        rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."), indices
                    ).transpose(0, 1)

                if self.config.ulysses_sequence_parallel_size > 1:
                    input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad_and_slice_inputs(
                        input_ids_rmpad, position_ids_rmpad, sp_size=self.config.ulysses_sequence_parallel_size
                    )

                output = self.actor_module(
                    input_ids=input_ids_rmpad,
                    attention_mask=None,
                    position_ids=position_ids_rmpad,
                    **multi_modal_inputs,
                    use_cache=False,
                )
                logits_rmpad = output.logits.squeeze(0)  # (total_nnz, vocab_size)
                logits_rmpad.div_(temperature)
                vocab_size = logits_rmpad.shape[-1]

                if self.config.ulysses_sequence_parallel_size > 1:
                    logits_rmpad = gather_outpus_and_unpad(logits_rmpad, gather_dim=0, unpad_dim=0, padding_size=pad_size)

                # Memory-efficient: extract response logits directly from unpadded tensor
                # without allocating the full (bsz, seqlen, vocab_size) padded tensor.
                #
                # Original logic (OOM-prone):
                #   full_logits = pad_input(logits_rmpad, indices, batch_size, seqlen)  # (bsz, seqlen, V)
                #   logits = full_logits[:, -response_length-1:-1, :]                  # (bsz, response_length, V)
                #
                # The slice [-response_length-1:-1] corresponds to positions
                # [seqlen-response_length-1, seqlen-1) in each sample's row.
                # We find which unpadded tokens fall in this range using `indices`.

                # indices contains flattened positions in (bsz * seqlen) for each valid token.
                # For sample i, the response logits region is:
                #   [i*seqlen + (seqlen - response_length - 1), i*seqlen + (seqlen - 1))
                # i.e., [i*seqlen + seqlen - response_length - 1, i*seqlen + seqlen - 1)

                # Pre-allocate output with zeros (matching pad_input behavior for missing positions)
                logits = torch.zeros(batch_size, response_length, vocab_size,
                                     dtype=logits_rmpad.dtype, device=logits_rmpad.device)

                for i in range(batch_size):
                    # Response logits region in flattened (bsz*seqlen) space
                    region_start = i * seqlen + (seqlen - response_length - 1)
                    region_end = i * seqlen + (seqlen - 1)

                    # Find which unpadded tokens fall in [region_start, region_end)
                    # indices is sorted per sample, so we can use searchsorted
                    left = torch.searchsorted(indices, region_start)
                    right = torch.searchsorted(indices, region_end)

                    if right > left:
                        # Get the positions within the response_length window
                        token_positions = indices[left:right] - region_start  # relative positions in [0, response_length)
                        logits[i, token_positions] = logits_rmpad[left:right]

                del logits_rmpad
            else:
                output = self.actor_module(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    **multi_modal_inputs,
                    use_cache=False,
                )
                logits = output.logits
                logits.div_(temperature)
                logits = logits[:, -response_length - 1 : -1, :]  # (bsz, response_length, vocab_size)

        return logits

    @torch.no_grad()
    def compute_jsd_metrics(self, data: DataProto) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute TRUE token-level JSD and entropy from full vocabulary distributions.

        This method receives interleaved (normal, noisy) pairs:
            data[0] = normal_0, data[1] = noisy_0, data[2] = normal_1, ...

        Both normal and noisy samples share the SAME response tokens (generated
        under the normal/real image). The noisy sample only differs in its visual
        input (perturbed or black image). This ensures that JSD and entropy gap
        measure the pure effect of visual information at each token position,
        without confounding from different generation trajectories.

        For each pair, it performs forward passes to get full logits, then computes:
            - JSD_t: true JSD between P(·|I, s_t) and P(·|I', s_t) over full vocab
            - H_normal_t: true Shannon entropy of P(·|I, s_t) over full vocab
            - H_noisy_t: true Shannon entropy of P(·|I', s_t) over full vocab

        The entropy gap is defined as:
            ΔH_t = H_noisy_t - H_normal_t
        which captures per-token visual dependency: positive means removing the
        image increases uncertainty at that position.

        Args:
            data: DataProto containing interleaved (normal, noisy) pairs.
                  Even indices are normal, odd indices are noisy.
                  Both share the same response tokens.

        Returns:
            jsd_t: (num_pairs, response_length) per-token JSD
            h_normal_t: (num_pairs, response_length) per-token entropy with real image
            h_noisy_t: (num_pairs, response_length) per-token entropy with noisy/null image
        """
        from ...utils.jsd_masking import compute_jsd_and_entropy_from_logits

        self.actor_module.eval()

        temperature = data.meta_info["temperature"]
        score_type = data.meta_info.get("score_type", "D")
        need_kl = (score_type == "H")

        select_keys = ["responses", "input_ids", "attention_mask", "position_ids"]
        if "multi_modal_inputs" in data.non_tensor_batch.keys():
            non_tensor_select_keys = ["multi_modal_inputs"]
        else:
            non_tensor_select_keys = []

        total_samples = len(data.batch)
        num_pairs = total_samples // 2
        response_length = data.batch["responses"].size(-1)

        jsd_all = torch.zeros(num_pairs, response_length, device="cpu")
        entropy_normal_all = torch.zeros(num_pairs, response_length, device="cpu")
        entropy_noisy_all = torch.zeros(num_pairs, response_length, device="cpu")
        kl_all = torch.zeros(num_pairs, response_length, device="cpu") if need_kl else None

        # Process pairs in micro-batches
        micro_batch_size = self.config.micro_batch_size_per_device_for_experience
        selected_data = data.select(select_keys, non_tensor_select_keys)

        for start in tqdm(range(0, num_pairs, micro_batch_size),
                          desc="Compute JSD metrics", disable=(self.rank != 0)):
            end = min(start + micro_batch_size, num_pairs)

            # Even indices = normal, odd indices = noisy
            normal_idx = list(range(start * 2, end * 2, 2))
            noisy_idx = list(range(start * 2 + 1, end * 2 + 1, 2))

            normal_data = selected_data[normal_idx]
            noisy_data = selected_data[noisy_idx]

            normal_inputs = {**normal_data.batch, **normal_data.non_tensor_batch}
            noisy_inputs = {**noisy_data.batch, **noisy_data.non_tensor_batch}

            # Forward pass to get full logits
            logits_normal = self._forward_micro_batch_logits(normal_inputs, temperature=temperature)
            logits_noisy = self._forward_micro_batch_logits(noisy_inputs, temperature=temperature)

            # Compute true token-level JSD and entropy from full distributions
            jsd_batch, entropy_normal_batch, entropy_noisy_batch, kl_pq_batch = compute_jsd_and_entropy_from_logits(
                logits_normal=logits_normal,
                logits_noisy=logits_noisy,
                compute_kl=need_kl,
            )

            jsd_all[start:end] = jsd_batch.cpu()
            entropy_normal_all[start:end] = entropy_normal_batch.cpu()
            entropy_noisy_all[start:end] = entropy_noisy_batch.cpu()
            if need_kl:
                kl_all[start:end] = kl_pq_batch.cpu()

            # Free GPU memory
            del logits_normal, logits_noisy, jsd_batch, entropy_normal_batch, entropy_noisy_batch, kl_pq_batch

        return jsd_all, entropy_normal_all, entropy_noisy_all, kl_all

    def _optimizer_step(self) -> torch.Tensor:
        if isinstance(self.actor_module, FSDP):
            grad_norm = self.actor_module.clip_grad_norm_(self.config.max_grad_norm)
        else:
            grad_norm = nn.utils.clip_grad_norm_(self.actor_module.parameters(), max_norm=self.config.max_grad_norm)

        self.actor_optimizer.step()
        return grad_norm

    @torch.no_grad()
    def compute_log_prob(self, data: DataProto) -> torch.Tensor:
        """Compute the log probability of the responses given input_ids, attention_mask and position_ids

        Args:
            data (DataProto): a DataProto containing keys

                ``input_ids``: tensor of shape [batch_size, sequence_length]. torch.int64. Note that input_ids is the
                concatenation of prompt and response. Note that ``sequence_length = prompt_length + response_length``.

                ``attention_mask``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``position_ids``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``responses``:  tensor of shape [batch_size, response_length]. torch.int64.

        Returns:
            torch.Tensor: the log_prob tensor
        """
        self.actor_module.eval()

        temperature = data.meta_info["temperature"]
        select_keys = ["responses", "input_ids", "attention_mask", "position_ids"]
        if "multi_modal_inputs" in data.non_tensor_batch.keys():
            non_tensor_select_keys = ["multi_modal_inputs"]
        else:
            non_tensor_select_keys = []

        micro_batches = data.select(select_keys, non_tensor_select_keys).split(
            self.config.micro_batch_size_per_device_for_experience
        )
        log_probs_lst = []
        for micro_batch in tqdm(micro_batches, desc="Compute log probs", disable=(self.rank != 0)):
            model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch}
            _, log_probs = self._forward_micro_batch(model_inputs, temperature=temperature)
            log_probs_lst.append(log_probs)

        log_probs = torch.concat(log_probs_lst, dim=0)
        return log_probs

    def update_policy(self, data: DataProto) -> Dict[str, Any]:
        self.actor_module.train()

        temperature = data.meta_info["temperature"]  # temperature must be in the data.meta_info to avoid slient error
        select_keys = ["responses", "input_ids", "attention_mask", "position_ids", "old_log_probs", "advantages"]
        if self.config.use_kl_loss:
            select_keys.append("ref_log_prob")
        # Include JSD mask if available
        if self.config.jsd_mask and "jsd_mask" in data.batch.keys():
            select_keys.append("jsd_mask")

        if "multi_modal_inputs" in data.non_tensor_batch.keys():
            non_tensor_select_keys = ["multi_modal_inputs"]
        else:
            non_tensor_select_keys = []

        # Split to make minibatch iterator for updating the actor
        # See PPO paper for details. https://arxiv.org/abs/1707.06347
        mini_batches = data.select(select_keys, non_tensor_select_keys).split(self.config.global_batch_size_per_device)

        metrics = defaultdict(list)
        n = len(mini_batches)
        for _ in range(self.config.ppo_epochs):
            for i, mini_batch in enumerate(mini_batches):
                gradient_accumulation = (
                    self.config.global_batch_size_per_device // self.config.micro_batch_size_per_device_for_update
                )
                micro_batches = mini_batch.split(self.config.micro_batch_size_per_device_for_update)

                self.actor_optimizer.zero_grad()
                for micro_batch in tqdm(micro_batches, desc=f"Update policy [{i + 1}/{n}]", disable=(self.rank != 0)):
                    model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch}
                    responses = model_inputs["responses"]
                    response_length = responses.size(1)
                    attention_mask = model_inputs["attention_mask"]
                    response_mask = attention_mask[:, -response_length:]
                    old_log_prob = model_inputs["old_log_probs"]
                    advantages = model_inputs["advantages"]

                    clip_ratio = self.config.clip_ratio
                    entropy_coeff = self.config.entropy_coeff

                    # Get JSD mask if available
                    token_jsd_mask = None
                    if self.config.jsd_mask and "jsd_mask" in model_inputs:
                        token_jsd_mask = model_inputs["jsd_mask"]

                    # all return: (bsz, response_length)
                    entropy, log_prob = self._forward_micro_batch(model_inputs, temperature=temperature)

                    pg_loss, pg_clipfrac, ppo_kl = core_algos.compute_policy_loss(
                        old_log_prob=old_log_prob,
                        log_prob=log_prob,
                        advantages=advantages,
                        eos_mask=response_mask,
                        cliprange=clip_ratio,
                        jsd_mask=token_jsd_mask,
                        jsd_mask_mode=self.config.jsd_mask_mode,
                        jsd_mask_lambda=self.config.jsd_mask_lambda,
                    )
                    # compute entropy loss from entropy
                    entropy_loss = VF.masked_mean(entropy, response_mask)

                    # compute policy loss
                    policy_loss = pg_loss - entropy_loss * entropy_coeff

                    if self.config.use_kl_loss:
                        ref_log_prob = model_inputs["ref_log_prob"]
                        # compute kl loss
                        kld = core_algos.kl_penalty(
                            logprob=log_prob,
                            ref_logprob=ref_log_prob,
                            kl_penalty=self.config.kl_loss_type,
                        )
                        kl_loss = VF.masked_mean(kld, response_mask)
                        policy_loss = policy_loss + kl_loss * self.config.kl_loss_coef
                        metrics["actor/kl_loss"] = kl_loss.detach().item()
                        metrics["actor/kl_coef"] = self.config.kl_loss_coef

                    loss = policy_loss / gradient_accumulation
                    loss.backward()

                    batch_metrics = {
                        "actor/entropy_loss": entropy_loss.detach().item(),
                        "actor/pg_loss": pg_loss.detach().item(),
                        "actor/pg_clipfrac": pg_clipfrac.detach().item(),
                        "actor/ppo_kl": ppo_kl.detach().item(),
                    }
                    append_to_dict(metrics, batch_metrics)

                grad_norm = self._optimizer_step()
                append_to_dict(metrics, {"actor/grad_norm": grad_norm.detach().item()})

        self.actor_optimizer.zero_grad()
        return metrics
