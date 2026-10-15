# vLLM with text watermarking

This is the `watermark-all` branch of [KristianMin/vllm](https://github.com/KristianMin/vllm): vLLM with 12 text watermarking algorithms built into the engine. The watermark is applied on the GPU while tokens are sampled, and every algorithm comes with a detector that tests token IDs for the watermark without needing the model.

`gumbel` and `dual_key_gumbel` are upstream vLLM features, `red_green` is proposed upstream in [#60738](https://github.com/vllm-project/vllm/pull/60738), and the other algorithms exist only in this fork for now.

This page gets you started. The [watermarking guide](https://github.com/KristianMin/vllm/blob/watermark-all/docs/features/watermarking.md) describes every algorithm, its parameters and its detection statistics in full.

## Install

The fork changes only Python and Triton code, so it installs on top of the precompiled binaries of its upstream base commit, without a CUDA build:

```bash
git clone --branch watermark-all https://github.com/KristianMin/vllm.git
cd vllm
VLLM_USE_PRECOMPILED=1 pip install --editable .
```

You need an NVIDIA GPU and vLLM's usual requirements. The fork is tested on H200 and A100 GPUs.

## Generate watermarked text

Choose an algorithm and a secret key when you start the engine. Every request is then watermarked unless it opts out:

```bash
vllm serve Qwen/Qwen3-0.6B \
  --watermark-config '{"algorithm": "red_green", "key": 42}'
```

A request opts out with `"watermarking": false` in its body; with the OpenAI Python client, pass `extra_body={"watermarking": False}`.

The same in Python:

```python
from vllm import LLM, SamplingParams

llm = LLM("Qwen/Qwen3-0.6B", watermark_config={"algorithm": "red_green", "key": 42})
outputs = llm.generate(
    "Tell me a story about an explorer who discovers a mysterious island.",
    SamplingParams(temperature=1.0, max_tokens=256),
)
token_ids = list(outputs[0].outputs[0].token_ids)
```

In Python, a request opts out with `SamplingParams(watermarking=False)`. Greedy decoding (`temperature=0`) and beam search are never watermarked.

## Detect the watermark

Detection takes token IDs, the key and the settings that shape the watermark, but not the model:

```python
from vllm.v1.watermarking import RedGreenWatermarkDetector

result = RedGreenWatermarkDetector(key=42, gamma=0.25).detect(token_ids)
print(result.p_value, result.is_watermarked)
```

`is_watermarked` is true when the p-value is at most `p_value_threshold`, 0.01 unless you pass another value to the detector. For text you received rather than generated, tokenize it with the generating model's tokenizer:

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
token_ids = tokenizer.encode(text, add_special_tokens=False)
```

The detector must use the same tokenizer, `key` and `context_width` as generation, plus the settings listed for its algorithm below. Pass them as keyword arguments. `examples/basic/online_serving/watermark_detection_server.py` serves Gumbel-max detection over HTTP.

## Algorithms

Set `algorithm` in the watermark config; `gumbel` is the default. Every algorithm except `unigram` reads the last `context_width` tokens (default 4) as its context.

| `algorithm` | How it watermarks | Defaults | Detector | Detector settings beyond `key` and `context_width` |
| --- | --- | --- | --- | --- |
| `gumbel` | Gumbel-max sampling with keyed noise ([Aaronson](https://simons.berkeley.edu/sites/default/files/2024-10/LLM24-2%20Slides%20-%20Scott%20Aaronson.pdf)); tokens still follow the model's distribution | | `GumbelWatermarkDetector` | |
| `dual_key_gumbel` | Gumbel-max with two keys derived from `key`, the second used with probability `alpha` | `alpha` 0.1 | `DualKeyGumbelWatermarkDetector` | `alpha` |
| `lookahead_gumbel` | Gumbel-max that keeps every emitted token watermarked under speculative decoding | `aggregator` `tsum` | `GumbelWatermarkDetector` | |
| `red_green` | Raises the logits of a keyed green list by `delta` ([Kirchenbauer et al., 2023](https://arxiv.org/abs/2301.10226)) | `delta` 2.0, `gamma` 0.25 | `RedGreenWatermarkDetector` | `gamma` |
| `unigram` | Red-green with one green list for every position, without context ([Zhao et al., 2023](https://arxiv.org/abs/2306.17439)) | `delta` 2.0, `gamma` 0.5 | `UnigramWatermarkDetector` | `gamma` (no `context_width`) |
| `multibit_red_green` | Red-green that embeds a `message_bits`-bit integer `message` | `delta` 0.5, `message_bits` 16 | `MultibitRedGreenWatermarkDetector`, which also decodes the message | `message_bits` |
| `chi_squared` | Samples p·w, with weights w from keyed binomial scores ([Gloaguen et al., 2026](https://arxiv.org/abs/2602.06754)) | `delta` 1.0 | `BinomialScoreWatermarkDetector` | |
| `ppl_hard` | Highest expected score within a perplexity budget of `epsilon` nats per token (same paper) | `epsilon` 0 | `BinomialScoreWatermarkDetector` | |
| `ppl_soft` | The same budget, met by picking the token that maximizes score + β·log p | `epsilon` 0 | `BinomialScoreWatermarkDetector` | |
| `synthid` | [SynthID-Text](https://www.nature.com/articles/s41586-024-08025-4) tournament sampling | `depth` 30, `num_leaves` 2 | `SynthIDWatermarkDetector` | `depth` |
| `textgrain` | Keyed block coupling that removes a fraction `entropy_budget` of the sampling entropy (Li et al., 2026) | `entropy_budget` 0.2, `num_blocks` 16, `num_columns` 16 | `TextGrainWatermarkDetector` | `num_blocks`, `num_columns` |
| `gaussian_copula` | Gumbel-max whose noise is tied to the key through a Gaussian copula ([AutoMark](https://arxiv.org/abs/2609.37310)) | `correlation` 0.98551; AutoMark uses `context_width` 2 | `GaussianCopulaWatermarkDetector` | |

All detectors are in `vllm.v1.watermarking`.

## Speculative decoding

- `dual_key_gumbel`, `lookahead_gumbel` and `textgrain` support speculative decoding.
- `gumbel`, `red_green` and `unigram` need `"allow_target_only_watermarking": true`, which leaves draft tokens unwatermarked. For `gumbel`, that weakens the watermark in proportion to the share of accepted drafts. `red_green` and `unigram` verify drafts against their watermarked logits, so they keep their full strength at a lower acceptance rate.
- `multibit_red_green`, `chi_squared`, `ppl_hard`, `ppl_soft`, `synthid` and `gaussian_copula` don't support it.

The drafter must be an autoregressive model (`dspark`, `eagle`, `eagle3` or `mtp`) with `"draft_sample_method": "probabilistic"`. The [guide](https://github.com/KristianMin/vllm/blob/watermark-all/docs/features/watermarking.md#speculative-decoding) has the details.

## Before you rely on it

- Keep the key secret. Anyone who has it can detect the watermark, and also forge it or remove it.
- The p-value assumes independent scores. Measure the false-positive rate on your own unwatermarked text, with your key, before acting on `is_watermarked`.
- Detection settings must match generation exactly. If you served several configurations, test the text against each one and correct for multiple testing.
- Watermarking needs Model Runner V2, vLLM's default model runner. Models that bring their own sampler can't be watermarked.

## Differences from the papers' code

- All algorithms use vLLM's Philox function for their keyed randomness, and vLLM applies temperature, top-k and top-p before the watermark. The papers' implementations generate their randomness differently, so their detectors can't detect this fork's text, and the other way around. Philox is not cryptographic.
- `chi_squared` samples p·w, the distribution of the paper's Theorem D.6. The reference code also applies the temperature to w, so the two differ when the temperature isn't 1.
- `ppl_soft` draws its scores for the 4096 most probable tokens only, and selects among those.
- `chi_squared`, the two perplexity watermarks and `gaussian_copula` leave an output's first `context_width` tokens unwatermarked, as their references do. `textgrain` watermarks them with a padded context, as vLLM's Gumbel-max does.

## Tests

`tests/watermarking` holds unit tests, recorded outputs that pin every algorithm's tokens and detection results, and an end-to-end test that generates and detects text with 10 of the algorithms. The kernel and end-to-end tests need a GPU:

```bash
pytest tests/watermarking
```
