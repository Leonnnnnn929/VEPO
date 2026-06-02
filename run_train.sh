
set -x

source activate YOUR_ENV_PATH

export VLLM_ATTENTION_BACKEND=XFORMERS
export VLLM_USE_V1=0

export WANDB_BASE_URL=https://api.wandb.ai
export WANDB_API_KEY="YOUR_WANDB_API_KEY"



MODEL_PATH=Qwen/Qwen2.5-VL-7B-Instruct   
EXPERIMENT_NAME=vepo_qwen2_5_vl_7b
PROJECT_NAME=VEPO
CHECKPOINT_DIR="checkpoints/${PROJECT_NAME}/${EXPERIMENT_NAME}"

# Noisy image mode: "gaussian" (add Gaussian noise) or "no_image" (zero out pixels)
NOISY_IMAGE_MODE=gaussian


# JSD mask parameters 
JSD_MASK_TOP_P=0.2          # Fraction of tokens to select for training (top-p)
JSD_MASK_SCORE_TYPE=D       # Score formula: D = (1-(1-ĵ)^α·(1-|ΔĤ|)^(1-α))·ĥ
JSD_MASK_ALPHA=0.7          # α: trade-off between JSD and |ΔH| (0=all ΔH, 1=all JSD)
JSD_MASK_MODE=mask          
JSD_MASK_LAMBDA=0.5         

DATA_SEED=42

SYSTEM_PROMPT="""You FIRST think about the reasoning process as an internal monologue and then provide the final answer.
 The reasoning process MUST BE enclosed within <think> </think> tags. The final answer MUST BE put in \boxed{}.."""


SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

python3 -m verl.trainer.main \
    config=training_scripts/config.yaml \
    data.train_files=YOUR_TRAIN_DATA_PATH \
    data.val_files=xyliu6/k12-freeform@test \
    data.system_prompt="${SYSTEM_PROMPT}" \
    data.max_response_length=2048 \
    data.max_pixels=1000000 \
    data.rollout_batch_size=512 \
    worker.actor.global_batch_size=128 \
    worker.actor.micro_batch_size_per_device_for_update=16 \
    worker.actor.micro_batch_size_per_device_for_experience=4 \
    worker.actor.optim.lr=1e-6 \
    worker.actor.model.freeze_vision_tower=true \
    worker.actor.model.model_path=${MODEL_PATH} \
    worker.actor.use_kl_loss=false \
    worker.actor.offload.offload_params=true \
    worker.actor.offload.offload_optimizer=true \
    worker.reward.compute_score=math \
    worker.rollout.gpu_memory_utilization=0.35 \
    worker.rollout.tensor_parallel_size=4 \
    worker.rollout.n=12 \
    worker.rollout.enable_chunked_prefill=false \
    trainer.experiment_name=${EXPERIMENT_NAME} \
    trainer.project_name=${PROJECT_NAME} \
    trainer.n_gpus_per_node=${GPUS_PER_NODE} \
    trainer.nnodes=${NNODES} \
    trainer.save_freq=10 \
    trainer.val_before_train=false \
    worker.actor.is_noisy=true \
    worker.actor.noisy_image_mode=${NOISY_IMAGE_MODE} \
    worker.actor.aug_type=gaussian \
    worker.actor.gaussian_noise_step=500 \
    worker.actor.decay_mode=sigmoid \
    worker.actor.decay_coef=30 \
    worker.actor.decay_sig_mid_step=40 \
    worker.actor.jsd_mask=true \
    worker.actor.jsd_mask_top_p=${JSD_MASK_TOP_P} \
    worker.actor.jsd_mask_score_type=${JSD_MASK_SCORE_TYPE} \
    worker.actor.jsd_mask_alpha=${JSD_MASK_ALPHA} \
    worker.actor.jsd_mask_mode=${JSD_MASK_MODE} \
    trainer.save_checkpoint_path="${CHECKPOINT_DIR}" \
    data.image_key="images" \
    data.val_image_key="images" \
    trainer.total_episodes=20 \
    data.seed=${DATA_SEED}

echo "Training complete. Checkpoints saved to: ${CHECKPOINT_DIR}"
