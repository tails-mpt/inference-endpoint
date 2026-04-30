# ADR 0008: Speculative decoding from HF only — no GCS drafts

**Status:** Accepted

## Context

The starting point for this tool was baby-shark's
`examples/jobs/sglang_serving_qwen2.5-1.5B`, which:

1. Downloaded a custom-trained EAGLE3 draft model from a private GCS
   bucket (`gs://speculative-decoding-models/runs/...`).
2. Patched `eagle_aux_hidden_state_layer_ids` in the draft's
   `config.json` to work around a sglang layer-indexing quirk.
3. Copied tokenizer files from the target model into the draft directory
   because EAGLE3 drafts ship without tokenizers.

Steps (2) and (3) only matter for the GCS path. Public HF-published
drafts (e.g. `RedHatAI/Qwen3-14B-speculator.eagle3`) ship correctly
configured and with tokenizers; both sglang and vllm accept HF IDs
directly for `--speculative-draft-model-path` / the
`--speculative-config` JSON.

## Decision

v0.1 supports speculative decoding **only with HF-published drafts**.
`SPEC_DRAFT_MODEL` takes an HF model ID. No download step, no config
patching, no tokenizer copy.

## Consequences

- **Code stays small.** No `prep_model.py`, no GCS auth code, no
  tokenizer-copy logic. ~50 LOC saved.
- **Custom-trained drafts in private GCS are out of scope.** Teams that
  train their own EAGLE3 drafts must either (a) push them to HF (private
  or public), or (b) wait for a v0.2 that adds the GCS path back.
- **HF-draft availability is growing fast.** As of writing, RedHatAI
  publishes EAGLE3 speculators for most popular base models. The "must
  use private GCS draft" use case is shrinking.
- **Adding the GCS path back is straightforward.** Add a
  `prep_model.py` step gated by `SPEC_DRAFT_MODEL` starting with `gs://`.
  The split would be: HF path = no prep; GCS path = full prep. No
  reorganization of the rest of the tool.
