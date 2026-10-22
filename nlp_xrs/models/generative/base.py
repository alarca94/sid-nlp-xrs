import torch

from nlp_xrs.utils.constants import Key
from nlp_xrs.utils.in_out import plot_mask


def generate_square_subsequent_mask(total_len):
    mask = torch.tril(torch.ones(total_len, total_len))  # (total_len, total_len), lower triangle -> 1.; others 0.
    mask = mask == 0  # lower -> False; others True
    return mask


def generate_peter_mask(src_len, tgt_len):
    total_len = src_len + tgt_len
    mask = generate_square_subsequent_mask(total_len)
    mask[0, 1] = False  # allow to attend for user and item
    return mask


def generate_peter_sid_mask(ui_len: int, src_len: int, tgt_len: int) -> torch.Tensor:
    """Generate PETER-style mask adapted for multi-token semantic item IDs.

    All item code positions (0..i_len-1) are fully visible to each other
    (bidirectional intra-item attention) and can also attend to the user
    position (i_len).  The rest of the sequence (features + explanation)
    remains causal.

    Design rationale: allowing all codes – not only the last one – to attend
    to the user enables early codebook tokens to incorporate user context when
    building their contribution to the hidden state.  This mirrors the
    observation in the PETER paper where the item and user positions mutually
    attend to improve context prediction.
    """
    total_len = src_len + tgt_len
    mask = generate_square_subsequent_mask(total_len)
    mask[: ui_len - 1, :ui_len] = False  # item codes attend to each other and to the user
    return mask


def peter_like_mask(mask, lengths):
    hist_len = lengths[Key.ITEM] - 1
    mask[hist_len, hist_len + 1] = False  # allow last item to attend to the user
    return mask


def bert_like_mask(mask, lengths):
    ui_len = lengths[Key.USER] + lengths[Key.ITEM]
    mask[:ui_len, :ui_len] = False
    return mask


def sawer_like_mask(mask, lengths):
    mask[: lengths[Key.ITEM], lengths[Key.ITEM]] = False
    return mask


def generate_sequer_mask(src_len, tgt_len, mask_type="default", plot=False, lengths=None):
    total_len = src_len + tgt_len
    if "mlm" in mask_type:
        # Candidate item ID will be masked and added after the User Token
        total_len += 1

    mask = generate_square_subsequent_mask(total_len)

    if mask_type == "peter":
        mask = peter_like_mask(mask, lengths)
    elif mask_type.startswith("bert"):
        mask = bert_like_mask(mask, lengths)
    elif mask_type.startswith("sawer"):
        mask = sawer_like_mask(mask, lengths)

    if plot:
        plot_mask(mask)
    return mask
