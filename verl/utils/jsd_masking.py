"""
JSD-based token scoring and masking utilities for VEPO.

Token-level visual dependency:
    Let I be the visual input and I' be a non-informative, perturbed version.
    At a given state s_t = (q, o_{<t}), the visual dependency at step t is the
    JSD between the policy's full output distributions conditioned on I and I':

        JSD_t = 0.5 * KL(P_t || M_t) + 0.5 * KL(Q_t || M_t)

    where P_t = softmax(logits_normal_t), Q_t = softmax(logits_noisy_t),
    M_t = 0.5 * (P_t + Q_t).

Token-level entropy:
    H_t = -sum_v P_t(v) * log P_t(v)

These utilities compute token-level JSD / entropy under normal and noisy visual
conditions, then derive per-token VEPO scores and top-p masks from them.
"""

import torch
import torch.nn.functional as F
from typing import Optional, Tuple


@torch.no_grad()
def compute_jsd_and_entropy_from_logits(
    logits_normal: torch.Tensor,
    logits_noisy: torch.Tensor,
    compute_kl: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """
    Compute token-level JSD and entropy for both distributions in a single pass.

    Both normal and noisy forward passes use the SAME response tokens (generated
    under the real image). They only differ in visual input. This ensures that
    JSD and entropy gap measure the pure effect of visual information.

    Entropy gap is defined as:
        ΔH_t = H_noisy_t - H_normal_t
    Positive ΔH means removing the image increases uncertainty → token depends on image.

    Args:
        logits_normal: (batch_size, response_length, vocab_size) logits with normal image
        logits_noisy: (batch_size, response_length, vocab_size) logits with noisy/null image
        compute_kl: if True, also compute KL(P||Q) for score_type H (default False)

    Returns:
        jsd: (batch_size, response_length) per-token JSD
        entropy_normal: (batch_size, response_length) per-token entropy of normal distribution
        entropy_noisy: (batch_size, response_length) per-token entropy of noisy distribution
        kl_pq: (batch_size, response_length) per-token KL(P||Q), or None if compute_kl=False
    """
    eps = 1e-8

    # Compute log probs and probs for both distributions
    log_p = F.log_softmax(logits_normal.float(), dim=-1)  # (bs, resp_len, vocab)
    log_q = F.log_softmax(logits_noisy.float(), dim=-1)   # (bs, resp_len, vocab)
    p = torch.exp(log_p)  # (bs, resp_len, vocab)
    q = torch.exp(log_q)  # (bs, resp_len, vocab)

    # Entropy of normal distribution: H_normal_t = -sum_v P(v) * log P(v)
    entropy_normal = -(p * log_p).sum(dim=-1)  # (bs, resp_len)

    # Entropy of noisy distribution: H_noisy_t = -sum_v Q(v) * log Q(v)
    entropy_noisy = -(q * log_q).sum(dim=-1)  # (bs, resp_len)

    # JSD: midpoint M = 0.5 * (P + Q)
    m = 0.5 * (p + q)
    log_m = torch.log(m + eps)

    kl_p_m = (p * (log_p - log_m)).sum(dim=-1)  # (bs, resp_len)
    kl_q_m = (q * (log_q - log_m)).sum(dim=-1)  # (bs, resp_len)
    jsd = (0.5 * kl_p_m + 0.5 * kl_q_m).clamp(min=0.0)  # (bs, resp_len)

    # KL(P || Q) = DKL(πθ(·|st, I) ∥ πθ(·|st, I'))
    # Only computed when needed (score_type H) to avoid unnecessary overhead
    kl_pq = None
    if compute_kl:
        kl_pq = (p * (log_p - log_q)).sum(dim=-1).clamp(min=0.0)  # (bs, resp_len)

    return jsd, entropy_normal, entropy_noisy, kl_pq


@torch.no_grad()
def compute_jsd_mask_scores(
    jsd_t: torch.Tensor,
    h_t: torch.Tensor,
    response_mask: torch.Tensor,
    score_type: str = "A",
    alpha: float = 0.5,
    h_noisy_t: Optional[torch.Tensor] = None,
    kl_t: Optional[torch.Tensor] = None,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Compute per-token importance scores for JSD-based token masking.

    Entropy gap (δ_t) is defined as the difference between noisy and normal entropy:
        ΔH_t = H_noisy_t - H_normal_t
    Positive ΔH means removing the image increases uncertainty at position t,
    indicating that the token depends on visual information.

    Four scoring formulas are supported:
        Formula A: s_t = 1 - (1 - ĵ_t)(1 - ĥ_t)(1 - δ̂_t)
            Tokens with high JSD OR high entropy OR positive entropy gap get high scores.

        Formula B: s_t = (1 - (1 - ĵ_t)(1 - ĥ_t)) · δ̂_t
            Tokens need BOTH (high JSD or high entropy) AND positive entropy gap.

        Formula C: s_t = (1 - (1 - ĵ_t)^α · (1 - δ̂_t)^(1-α)) · ĥ_t
            Uses max(ΔH_t, 0) as the entropy gap signal (only positive direction).
            α controls the trade-off between JSD and δ contributions.
            Gated by normalized entropy ĥ_t to focus on uncertain tokens.
            δ_t = max(H_noisy_t - H_normal_t, 0).

        Formula D: s_t = (1 - (1 - ĵ_t)^α · (1 - |ΔĤ_t|)^(1-α)) · ĥ_t
            Uses absolute entropy gap |ΔH_t| = |H_noisy_t - H_normal_t| to capture
            both directions of entropy change as signals of visual dependency.
            α controls the trade-off between JSD and |ΔH| contributions.

    All inputs (jsd_t, h_t, ΔH_t) are min-max normalized per response to [0, 1]
    before computing scores, ensuring comparable scales.

    Args:
        jsd_t: (batch_size, response_length) per-token JSD
        h_t: (batch_size, response_length) per-token entropy (normal/real image)
        response_mask: (batch_size, response_length) mask for valid response tokens
        score_type: "A", "B", "C", or "D", selects the scoring formula
        alpha: trade-off parameter for score_type C/D (default 0.5)
        h_noisy_t: (batch_size, response_length) per-token entropy (noisy/null image).
                   If None, falls back to temporal difference ΔH_t = H_t - H_{t-1}.
        eps: small constant for numerical stability

    Returns:
        scores: (batch_size, response_length) per-token importance scores in [0, 1]
    """
    # Compute entropy gap ΔH_t
    if h_noisy_t is not None:
        # ΔH_t = H_noisy_t - H_normal_t (per-token, same position)
        # Positive means removing image increases uncertainty → visual dependency
        delta_h = h_noisy_t - h_t
    else:
        # Fallback: temporal difference ΔH_t = H_t - H_{t-1}
        delta_h = torch.zeros_like(h_t)
        delta_h[:, 1:] = h_t[:, 1:] - h_t[:, :-1]

    delta_h_pos = torch.clamp(delta_h, min=0.0)  # max(ΔH_t, 0)
    delta_h_abs = torch.abs(delta_h)  # |ΔH_t|

    # ---- Per-response min-max normalization to [0, 1] ----
    def _minmax_normalize(x, mask):
        """Normalize x to [0, 1] per response, only over valid tokens."""
        # Set invalid positions to +inf for min, -inf for max
        x_masked = x.clone()
        x_masked[~mask.bool()] = float('inf')
        x_min = x_masked.min(dim=-1, keepdim=True).values  # (bs, 1)
        x_masked[~mask.bool()] = float('-inf')
        x_max = x_masked.max(dim=-1, keepdim=True).values  # (bs, 1)
        x_range = (x_max - x_min).clamp(min=eps)
        x_norm = (x - x_min) / x_range
        x_norm = x_norm * mask  # zero out invalid positions
        return x_norm.clamp(0.0, 1.0)

    jsd_norm = _minmax_normalize(jsd_t, response_mask)
    h_norm = _minmax_normalize(h_t, response_mask)
    delta_h_norm = _minmax_normalize(delta_h_pos, response_mask)

    if score_type == "A":
        # s_t = 1 - (1 - ĵ_t)(1 - ĥ_t)(1 - δ̂_t)
        scores = 1.0 - (1.0 - jsd_norm) * (1.0 - h_norm) * (1.0 - delta_h_norm)
    elif score_type == "B":
        # s_t = (1 - (1 - ĵ_t)(1 - ĥ_t)) · δ̂_t
        scores = (1.0 - (1.0 - jsd_norm) * (1.0 - h_norm)) * delta_h_norm
    elif score_type == "C":
        # s_t = (1 - (1 - ĵ_t)^α · (1 - δ̂_t)^(1-α)) · ĥ_t
        # Uses max(ΔH_t, 0) as entropy change signal (only positive direction)
        # α controls trade-off: higher α emphasizes JSD, lower α emphasizes δ
        # Gated by normalized entropy ĥ_t to focus on uncertain tokens
        geometric_signal = 1.0 - (1.0 - jsd_norm).pow(alpha) * (1.0 - delta_h_norm).pow(1.0 - alpha)
        entropy_gate = h_norm  # ĥ_t
        scores = geometric_signal * entropy_gate
    elif score_type == "D":
        # s_t = (1 - (1 - ĵ_t)^α · (1 - |ΔĤ_t|)^(1-α)) · ĥ_t
        # Uses |ΔH_t| instead of max(ΔH_t, 0) to capture both directions of entropy change
        # α controls trade-off: higher α emphasizes JSD, lower α emphasizes |ΔH|
        delta_h_abs_norm = _minmax_normalize(delta_h_abs, response_mask)
        geometric_signal = 1.0 - (1.0 - jsd_norm).pow(alpha) * (1.0 - delta_h_abs_norm).pow(1.0 - alpha)
        entropy_gate = h_norm  # ĥ_t
        scores = geometric_signal * entropy_gate
    elif score_type == "E":
        # s_t = 1 - (1 - ĵ_t)^α · (1 - |ΔĤ_t|)^(1-α)
        # Same as D but WITHOUT entropy gating (ablation: removes ĥ_t multiplier)
        delta_h_abs_norm = _minmax_normalize(delta_h_abs, response_mask)
        scores = 1.0 - (1.0 - jsd_norm).pow(alpha) * (1.0 - delta_h_abs_norm).pow(1.0 - alpha)
    elif score_type == "F":
        # s_t = (1 - ((1 - ĵ_t)·α + (1 - |ΔĤ_t|)·(1-α))) · ĥ_t
        # Linear combination instead of geometric (exponential) form
        delta_h_abs_norm = _minmax_normalize(delta_h_abs, response_mask)
        linear_signal = 1.0 - ((1.0 - jsd_norm) * alpha + (1.0 - delta_h_abs_norm) * (1.0 - alpha))
        entropy_gate = h_norm  # ĥ_t
        scores = linear_signal * entropy_gate
    elif score_type == "G":
        # s_t = ĵ_t · |ΔĤ_t| · ĥ_t
        # Simple product of all three normalized signals (ablation)
        delta_h_abs_norm = _minmax_normalize(delta_h_abs, response_mask)
        scores = jsd_norm * delta_h_abs_norm * h_norm
    elif score_type == "H":
        # s_t = (1 - (1 - k̂l_t)^α · (1 - |ΔĤ_t|)^(1-α)) · ĥ_t
        # Same as D but replaces JSD with KL(πθ(·|st, I) ∥ πθ(·|st, I'))
        # kl_t is passed via the kl_t parameter
        if kl_t is None:
            raise ValueError("score_type 'H' requires kl_t (KL divergence) to be provided.")
        kl_norm = _minmax_normalize(kl_t, response_mask)
        delta_h_abs_norm = _minmax_normalize(delta_h_abs, response_mask)
        geometric_signal = 1.0 - (1.0 - kl_norm).pow(alpha) * (1.0 - delta_h_abs_norm).pow(1.0 - alpha)
        entropy_gate = h_norm  # ĥ_t
        scores = geometric_signal * entropy_gate
    else:
        raise ValueError(f"Unknown score_type: {score_type}. Must be 'A', 'B', 'C', 'D', 'E', 'F', 'G', or 'H'.")

    scores = scores * response_mask
    return scores


@torch.no_grad()
def compute_jsd_topk_mask(
    jsd_t: torch.Tensor,
    h_t: torch.Tensor,
    response_mask: torch.Tensor,
    top_p: float = 0.3,
    score_type: str = "A",
    alpha: float = 0.5,
    h_noisy_t: Optional[torch.Tensor] = None,
    bottom: bool = False,
    kl_t: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, dict]:
    """
    Compute a binary TopK mask selecting the top-p fraction of tokens per response
    based on JSD-based importance scores.

    Only tokens within the top-p fraction (by score) are selected (mask=1).
    The mask is applied on top of response_mask (invalid tokens are always 0).

    If bottom=True, selects the BOTTOM-p fraction (least important tokens) instead.

    Args:
        jsd_t: (batch_size, response_length) per-token JSD
        h_t: (batch_size, response_length) per-token entropy (normal/real image)
        response_mask: (batch_size, response_length) mask for valid response tokens
        top_p: fraction of tokens to select per response (0.0, 1.0]
        score_type: "A", "B", "C", or "D", selects the scoring formula
        alpha: trade-off parameter for score_type C/D (default 0.5)
        h_noisy_t: (batch_size, response_length) per-token entropy (noisy/null image).
                   If provided, entropy gap = H_noisy - H_normal (visual dependency).
        bottom: if True, select the bottom-p tokens (lowest scores) instead of top-p
        kl_t: (batch_size, response_length) per-token KL divergence. Required for score_type H.

    Returns:
        topk_mask: (batch_size, response_length) binary mask, 1 for selected tokens
        metrics: dict with diagnostic metrics
    """
    scores = compute_jsd_mask_scores(
        jsd_t=jsd_t,
        h_t=h_t,
        response_mask=response_mask,
        score_type=score_type,
        alpha=alpha,
        h_noisy_t=h_noisy_t,
        kl_t=kl_t,
    )

    bs, resp_len = scores.shape
    valid_lengths = response_mask.sum(dim=-1)  # (bs,)

    # Number of tokens to select per response: k = ceil(p * T_valid)
    k_per_response = torch.ceil(top_p * valid_lengths).long().clamp(min=1)  # (bs,)

    # Set invalid positions to -inf (top) or +inf (bottom) so they are never selected
    scores_for_topk = scores.clone()
    if bottom:
        scores_for_topk[~response_mask.bool()] = float('inf')
    else:
        scores_for_topk[~response_mask.bool()] = float('-inf')

    # Sort scores: descending for top-p, ascending for bottom-p
    sorted_scores, sorted_indices = scores_for_topk.sort(dim=-1, descending=(not bottom))

    # Create position indices: [0, 1, 2, ..., resp_len-1] for each sample
    position_indices = torch.arange(resp_len, device=scores.device).unsqueeze(0).expand(bs, -1)

    # Mask: position < k_per_response[i]
    topk_mask_sorted = (position_indices < k_per_response.unsqueeze(-1)).float()

    # Scatter back to original positions
    topk_mask = torch.zeros_like(scores)
    topk_mask.scatter_(1, sorted_indices, topk_mask_sorted)

    # Ensure invalid positions are always 0
    topk_mask = topk_mask * response_mask

    # ---- Compute ΔH for metrics ----
    if h_noisy_t is not None:
        delta_h = h_noisy_t - h_t  # entropy gap: H_noisy - H_normal
    else:
        delta_h = torch.zeros_like(h_t)
        delta_h[:, 1:] = h_t[:, 1:] - h_t[:, :-1]
    delta_h_pos = torch.clamp(delta_h, min=0.0)
    indicator = (delta_h > 0).float()

    # Diagnostic metrics
    total_valid = response_mask.sum().clamp(min=1)
    total_selected = topk_mask.sum().clamp(min=1)
    actual_select_ratio = topk_mask.sum(dim=-1) / valid_lengths.clamp(min=1)

    valid_scores = scores[response_mask.bool()]
    selected_scores = scores[topk_mask.bool()] if topk_mask.sum() > 0 else torch.tensor([0.0])

    metrics = {
        "jsd_mask/jsd_mean": (jsd_t * response_mask).sum() / total_valid,
        "jsd_mask/entropy_mean": (h_t * response_mask).sum() / total_valid,
        "jsd_mask/delta_h_positive_ratio": (indicator * response_mask).sum() / total_valid,
        "jsd_mask/score_mean": valid_scores.mean() if valid_scores.numel() > 0 else torch.tensor(0.0),
        "jsd_mask/score_std": valid_scores.std() if valid_scores.numel() > 1 else torch.tensor(0.0),
        "jsd_mask/selected_score_mean": selected_scores.mean(),
        "jsd_mask/selected_score_std": selected_scores.std() if selected_scores.numel() > 1 else torch.tensor(0.0),
        "jsd_mask/actual_select_ratio_mean": actual_select_ratio.mean(),
        "jsd_mask/actual_select_ratio_std": actual_select_ratio.std() if actual_select_ratio.numel() > 1 else torch.tensor(0.0),
        "jsd_mask/total_selected_tokens": total_selected,
        "jsd_mask/total_valid_tokens": total_valid,
        "jsd_mask/jsd_selected_mean": (jsd_t * topk_mask).sum() / total_selected,
        "jsd_mask/entropy_selected_mean": (h_t * topk_mask).sum() / total_selected,
        "jsd_mask/delta_h_pos_selected_mean": (delta_h_pos * topk_mask).sum() / total_selected,
    }

    return topk_mask, metrics


@torch.no_grad()
def visualize_jsd_mask_tokens(
    response_ids: torch.Tensor,
    topk_mask: torch.Tensor,
    scores: torch.Tensor,
    jsd_t: torch.Tensor,
    h_t: torch.Tensor,
    response_mask: torch.Tensor,
    tokenizer,
    num_samples: int = 3,
    max_tokens: int = 512,
    h_noisy_t: Optional[torch.Tensor] = None,
) -> list:
    """
    Generate HTML visualizations of JSD mask token selection.

    For each sample, tokens are rendered as colored spans:
      - Selected tokens (mask=1): green background, opacity proportional to score
      - Masked-out tokens (mask=0): light gray background
    A legend and per-sample statistics are included.

    Args:
        response_ids: (batch_size, response_length) token IDs of the response
        topk_mask: (batch_size, response_length) binary mask (1=selected)
        scores: (batch_size, response_length) importance scores in [0, 1]
        jsd_t: (batch_size, response_length) per-token JSD values
        h_t: (batch_size, response_length) per-token entropy values
        response_mask: (batch_size, response_length) valid token mask
        tokenizer: HuggingFace tokenizer for decoding token IDs
        num_samples: number of samples to visualize
        max_tokens: max tokens to display per sample (truncate for readability)
        h_noisy_t: (batch_size, response_length) per-token entropy with noisy/null image.
                   If provided, ΔH = H_noisy - H_normal.

    Returns:
        html_list: list of HTML strings, one per sample
    """
    import html as html_lib

    bs = response_ids.size(0)
    num_samples = min(num_samples, bs)

    # Compute ΔH for display
    if h_noisy_t is not None:
        delta_h = h_noisy_t - h_t  # H_noisy - H_normal
    else:
        delta_h = torch.zeros_like(h_t)
        delta_h[:, 1:] = h_t[:, 1:] - h_t[:, :-1]

    html_list = []

    for i in range(num_samples):
        valid_len = int(response_mask[i].sum().item())
        display_len = min(valid_len, max_tokens)

        ids = response_ids[i, :display_len].tolist()
        mask_vals = topk_mask[i, :display_len].tolist()
        score_vals = scores[i, :display_len].tolist()
        jsd_vals = jsd_t[i, :display_len].tolist()
        h_vals = h_t[i, :display_len].tolist()
        dh_vals = delta_h[i, :display_len].tolist()

        total_selected = int(topk_mask[i, :valid_len].sum().item())
        select_ratio = total_selected / max(valid_len, 1)

        # Build HTML
        html_parts = []
        html_parts.append(
            '<div style="font-family: monospace; font-size: 13px; line-height: 1.8; '
            'padding: 12px; background: #1e1e1e; color: #d4d4d4; border-radius: 8px; '
            'margin-bottom: 16px; white-space: pre-wrap; word-wrap: break-word;">'
        )

        # Header with stats
        html_parts.append(
            f'<div style="margin-bottom: 10px; padding: 8px; background: #2d2d2d; '
            f'border-radius: 4px; font-size: 12px; color: #9cdcfe;">'
            f'<b>Sample {i+1}</b> | '
            f'Total tokens: {valid_len} | '
            f'Selected: {total_selected} ({select_ratio:.1%}) | '
            f'Truncated: {"Yes" if valid_len > max_tokens else "No"}'
            f'</div>'
        )

        # Legend
        html_parts.append(
            '<div style="margin-bottom: 8px; font-size: 11px; color: #808080;">'
            '<span style="background: rgba(76,175,80,0.6); padding: 1px 4px; border-radius: 2px; '
            'color: white;">■ Selected (trained)</span> '
            '<span style="background: rgba(100,100,100,0.3); padding: 1px 4px; border-radius: 2px; '
            'color: #888;">■ Masked (skipped)</span> '
            '| Hover for details (JSD, entropy, ΔH, score)'
            '</div>'
        )

        # Render tokens
        for j, token_id in enumerate(ids):
            token_str = tokenizer.decode([token_id])
            token_display = html_lib.escape(token_str)
            # Replace spaces/newlines for visibility
            if token_str == ' ':
                token_display = '·'
            elif token_str == '\n':
                token_display = '↵\n'
            elif token_str == '\t':
                token_display = '→'

            is_selected = mask_vals[j] > 0.5
            score = score_vals[j]
            jsd_val = jsd_vals[j]
            h_val = h_vals[j]
            dh_val = dh_vals[j]

            tooltip = (
                f"token: {html_lib.escape(repr(token_str))} | "
                f"score: {score:.4f} | "
                f"JSD: {jsd_val:.4f} | "
                f"H: {h_val:.4f} | "
                f"ΔH: {dh_val:.4f}"
            )

            if is_selected:
                # Green with opacity proportional to score
                opacity = 0.3 + 0.7 * score  # range [0.3, 1.0]
                bg_color = f"rgba(76,175,80,{opacity:.2f})"
                text_color = "white"
                border = "1px solid rgba(76,175,80,0.8)"
            else:
                bg_color = "rgba(100,100,100,0.15)"
                text_color = "#666"
                border = "1px solid rgba(100,100,100,0.2)"

            html_parts.append(
                f'<span title="{tooltip}" style="'
                f'background: {bg_color}; '
                f'color: {text_color}; '
                f'padding: 1px 2px; '
                f'margin: 1px; '
                f'border-radius: 3px; '
                f'border: {border}; '
                f'display: inline-block; '
                f'cursor: pointer; '
                f'font-size: 12px;">'
                f'{token_display}</span>'
            )

        if valid_len > max_tokens:
            html_parts.append(
                f'<span style="color: #ff9800; font-style: italic;"> '
                f'... ({valid_len - max_tokens} more tokens)</span>'
            )

        html_parts.append('</div>')
        html_list.append(''.join(html_parts))

    return html_list
