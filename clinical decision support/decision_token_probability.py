"""Shared yes/no decision-token probability for malignancy classification."""

from typing import Tuple

import torch


def _single_token_id(tokenizer, answer: str) -> int:
    token_ids = tokenizer.encode(answer, add_special_tokens=False)
    if len(token_ids) != 1:
        raise RuntimeError(
            f"Expected {answer!r} to map to exactly one decision token, got {token_ids}."
        )
    return int(token_ids[0])


@torch.no_grad()
def infer_yesno_probability(
    model,
    processor,
    image,
    prompt_text: str,
) -> Tuple[float, str, float, float]:
    """Return P(yes) from a two-way softmax over next-token yes/no logits."""
    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": prompt_text},
        ],
    }]
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )
    inputs.pop("token_type_ids", None)

    model_dtype = getattr(model, "dtype", None)
    for key, value in inputs.items():
        if not hasattr(value, "to"):
            continue
        value = value.to(model.device)
        if (
            torch.is_tensor(value)
            and value.is_floating_point()
            and isinstance(model_dtype, torch.dtype)
        ):
            value = value.to(dtype=model_dtype)
        inputs[key] = value

    outputs = model(**inputs)
    if not hasattr(outputs, "logits") or outputs.logits is None:
        raise RuntimeError("Model outputs has no logits for yes/no classification.")

    yes_id = _single_token_id(processor.tokenizer, "yes")
    no_id = _single_token_id(processor.tokenizer, "no")
    decision_logits = outputs.logits[0, -1]
    yes_logit = decision_logits[yes_id].float()
    no_logit = decision_logits[no_id].float()
    _, p_yes = torch.softmax(torch.stack((no_logit, yes_logit)), dim=0)

    p_yes_value = float(p_yes.item())
    prediction = "yes" if p_yes_value >= 0.5 else "no"
    return (
        p_yes_value,
        prediction,
        float(yes_logit.item()),
        float(no_logit.item()),
    )
