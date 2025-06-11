from trl.trainer import GRPOTrainer
from accelerate.utils import is_peft_model
import torch


class AudioGRPOTrainer(GRPOTrainer):
    def __init__(self, model, *args, **kwargs):
        # force vllm off, kill reference‑model allocation
        kwargs.setdefault("args", None)  # let parent build a GRPOConfig
        super().__init__(model=model, *args, **kwargs)
        self.use_vllm = False  # safety: parent can’t sneak it on
        self.ref_model = None  # we’ll always .disable_adapter()

        # no tokenizer; use a trivial pass‑through
        class _DummyTok:
            pad_token_id = 0
            eos_token_id = 1
            bos_token_id = 1

            def __call__(self, **kw):
                raise RuntimeError("tokenizer unused")

            def batch_decode(self, ids, **kw):
                return ["<unused>"] * len(ids)

        self.processing_class = _DummyTok()

    # ------- generation & scoring ------- #
    def _generate_and_score_completions(self, inputs):
        """
        Overhauled for audio codec models.

        1. `inputs` is a list of dicts with a pre‑tokenised `prompt` tensor.
        2. We call `self.model.sample_audio(prompt, max_len=…)`
           …or whatever API you expose.
        3. We return the exact same dict keys the parent expects, but built
           from audio tokens.
        """
        device = self.accelerator.device
        prompts = [x["prompt"].to(device) for x in inputs]  # already tensors
        prompt_ids = torch.nn.utils.rnn.pad_sequence(
            prompts, batch_first=True, padding_value=0
        )
        prompt_mask = (prompt_ids != 0).int()

        # >>> YOUR OWN SAMPLER <<<
        completion_ids, completion_mask = self.model.sample_audio(
            prompt_ids,
            prompt_mask,
            max_new_tokens=self.max_completion_length,
            temperature=self.temperature,
            top_p=self.top_p,
        )

        # advantages & rewards identical; fall back to parent helper
        fake_inputs = [{"prompt": p} for p in prompts]  # minimal shim
        for d, c in zip(fake_inputs, completion_ids):
            d["completion_ids"] = c
        return super()._generate_and_score_completions(fake_inputs)

    # ------- log‑prob computation ------- #
    def _get_per_token_logps(
        self, model, input_ids, attention_mask, logits_to_keep, batch_size=None
    ):
        """
        Same math as parent but routed through your audio forward().
        Assume model returns logits over codebook tokens like text models.
        """
        with torch.no_grad():
            logits = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                logits_to_keep=logits_to_keep + 1,
            ).logits[:, :-1]  # drop last token
        target = input_ids[:, -logits_to_keep:]
        logps = torch.log_softmax(logits / self.temperature, dim=-1)
        return logps.gather(-1, target.unsqueeze(-1)).squeeze(-1)

    # no tokenizer decode, so strip any place parent tries; easiest is:
    # batch_decode = lambda *a, **k: [""] * len(a[1])
