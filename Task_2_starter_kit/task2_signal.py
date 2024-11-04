"""
You may modify the signal generation pipeline as you wish.

We use an LLM to generate a sentiment score according to the prompt below. 

You can improve the sentiment analysis here or generate your own signal.
"""

import re
import torch

SAMPLE_PROMPT = """Task: Analyze the following news headline about a stock and provide a sentiment score between -{signal_strengh} and {signal_strengh}, where:
- -{signal_strengh} means very negative sentiment
- -{threshold} means neutral negative sentiment
- 0 means neutral sentiment
- {threshold} indicates neutral positive sentiment
- {signal_strengh} means very positive sentiment

Do not provide any explanations. Output only a single number in the range of -{signal_strengh} to {signal_strengh} based on the sentiment of the news. 

News headline: "{news}"

Price Data: "{prices}"

Generate only a single integer value for the sentiment score after the colon. Sentiment score:
"""


def _generate_signal(tokenizer, model, device, news, prices, signal_strengh, threshold):
    """
    Using model forward pass to do backprop
    """
    prompt = SAMPLE_PROMPT.format(
        signal_strengh=signal_strengh,
        threshold=threshold,
        news=news,
        prices=prices
    )
    print(prompt)
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    print(inputs)
    print(inputs['input_ids'].shape)
    generated_ids = inputs["input_ids"].to(device)

    log_probs = []
    max_new_tokens = 5

    for _ in range(max_new_tokens):
        outputs = model(input_ids=generated_ids)
        logits = outputs.logits  # shape: [batch_size, seq_length, vocab_size]

        next_token_logits = logits[:, -1, :]

        # Apply numerical stability fix by subtracting the max logits
        next_token_logits = next_token_logits - torch.max(next_token_logits, dim=-1, keepdim=True)[0]

        next_token_probs = torch.softmax(next_token_logits, dim=-1)

        # Replace NaNs and Infs with zeros
        next_token_probs = torch.nan_to_num(next_token_probs, nan=0.0, posinf=0.0, neginf=0.0)

        # Ensure the probabilities sum to 1
        prob_sum = next_token_probs.sum(dim=-1, keepdim=True)
        # Avoid division by zero
        epsilon = 1e-8
        prob_sum = prob_sum + epsilon
        next_token_probs = next_token_probs / prob_sum

        # In case the sum is still zero (all probabilities are zero), replace with uniform distribution
        zero_sum_mask = (prob_sum.squeeze() == epsilon)
        if zero_sum_mask.any():
            vocab_size = next_token_probs.size(-1)
            next_token_probs[zero_sum_mask] = 1.0 / vocab_size

        next_token_id = torch.multinomial(next_token_probs, num_samples=1)

        token_log_prob = torch.log(next_token_probs[0, next_token_id[0, 0]])
        log_probs.append(token_log_prob)

        generated_ids = torch.cat([generated_ids, next_token_id], dim=1)

    total_log_prob = torch.stack(log_probs).sum()

    output_string = tokenizer.decode(generated_ids[0], skip_special_tokens=True).strip()

    match = re.search(r"Sentiment score:\s*(-?\d+(?:\.\d+)?)", output_string)
    signal_strength = float(match.group(1)) if match else 0

    return signal_strength, total_log_prob


def _generate_eval_signal(tokenizer, model, device, news, prices, signal_strengh, threshold):
    prompt = SAMPLE_PROMPT.format(signal_strengh=signal_strengh, threshold=threshold, news=news, prices=prices)
    print(prompt)

    # using news signals, prompt model for a scaled sentiment scorea
    input = tokenizer(prompt, return_tensors="pt").to(device)
    print(input)
    print(input['input_ids'].shape)
    outputs = model.generate(**input, max_new_tokens=5, pad_token_id=tokenizer.eos_token_id)
    output_string = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
    match = re.search(r"Sentiment score:\s*(-?\d+(?:\.\d+)?)", output_string)
    return float(match.group(1)) if match else 0


def generate_eval_signal(tokenizer, model, device, news, prices, signal_strengh, threshold):
    return _generate_eval_signal(tokenizer, model, device, news, prices, signal_strengh, threshold)


def generate_signal(tokenizer, model, device, news, prices, signal_strengh, threshold):
    return _generate_signal(tokenizer, model, device, news, prices, signal_strengh, threshold)
