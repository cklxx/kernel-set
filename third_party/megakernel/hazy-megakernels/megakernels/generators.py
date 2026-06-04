import torch
from torch import Tensor

from megakernels.llama import LlamaForCausalLM
from megakernels.mk import MK_Interpreter
from megakernels.model_types import BatchState
from megakernels.python_vm import PyVM_Interpreter
from megakernels.scheduler import Schedule


class Generator:
    def generate(
        self,
        output_tokens: Tensor,
        prompt_len: int,
        ntok: int,
        ntok_already_generated: int = 1,
    ):
        raise NotImplementedError

    def generate_with_eos(
        self,
        output_tokens: Tensor,
        prompt_len: int,
        ntok: int,
        eos_token_check_interval: int,
        eos_token_ids: list[int],
    ):
        """
        Return pos id with first eos token, and total num tokens generated
        """
        assert output_tokens.shape[0] == 1, "batch size must be 1"

        for ntok_already_generated in range(
            1,
            ntok,
            eos_token_check_interval,
        ):
            ntok_for_chunk = min(
                eos_token_check_interval, ntok - ntok_already_generated
            )
            self.generate(
                output_tokens,
                prompt_len=prompt_len,
                ntok=ntok_for_chunk,
                ntok_already_generated=ntok_already_generated,
            )

            start_out_idx = ntok_already_generated
            end_out_idx = ntok_already_generated + ntok_for_chunk

            to_cpu = output_tokens[0, start_out_idx:end_out_idx].cpu()
            for j, token in enumerate(to_cpu):
                if token in eos_token_ids:
                    # -1 because we didn't generate the first token
                    return start_out_idx + j, end_out_idx - 1

        return ntok, ntok - 1


class PyTorchGenerator(Generator):
    def __init__(
        self,
        model: LlamaForCausalLM,
    ):
        self.model = model

    def generate(
        self,
        output_tokens: Tensor,
        prompt_len: int,
        ntok: int,
        ntok_already_generated: int = 1,
    ):
        bs = output_tokens.shape[0]
        starting_seq_len = prompt_len + ntok_already_generated
        start_position_ids = torch.ones(
            bs, 1, dtype=torch.long, device=self.model.device
        ) * (starting_seq_len - 1)

        for i in range(ntok):
            position_ids = start_position_ids + i
            input_token_pos = i + ntok_already_generated - 1
            decode_inp = BatchState(
                input_ids=output_tokens[:, input_token_pos : input_token_pos + 1],
                position_ids=position_ids,
                seq_len=starting_seq_len + i + 1,
            )
            decode_output: BatchState = self.model(decode_inp)
            assert decode_output.output_ids is not None
            output_pos = input_token_pos + 1
            output_tokens[:, output_pos] = decode_output.output_ids.squeeze(-1)


class MK_Generator(Generator):
    def __init__(
        self,
        model: LlamaForCausalLM,
        interpreter: MK_Interpreter,
        schedule: Schedule,
        barrier_fill_val: int = 0,
        skip_mk: bool = False,
        skip_rest: bool = False,
    ):
        self.model = model
        self.interpreter = interpreter
        self.schedule = schedule

        self.barrier_fill_val = barrier_fill_val
        self.skip_mk = skip_mk
        self.skip_rest = skip_rest

        self.fill()

    def fill(self):
        self.schedule.globs.barriers.fill_(self.barrier_fill_val)

    def replace_with_noops(self):
        self.schedule.globs.instructions.zero_()

    def run(self, input_ids: Tensor, pos_id: int):
        if not self.skip_rest:
            batch_state = BatchState(
                input_ids=input_ids,
            )

            post_embedding: BatchState = self.model.model.embed_tokens(batch_state)
            hiddens = post_embedding.hidden_states
            assert hiddens is not None
            self.schedule.globs.hidden_states[:] = hiddens.squeeze(1)

        self.fill()
        self.schedule.globs.pos_id = pos_id
        if not self.skip_mk:
            self.interpreter.interpret(self.schedule.globs)

        if self.skip_rest:
            return input_ids

        logits = self.schedule.globs.logits
        output_ids = torch.argmax(logits, dim=-1)

        return output_ids

    def generate(
        self,
        output_tokens: Tensor,
        prompt_len: int,
        ntok: int,
        ntok_already_generated: int = 1,
    ):
        """
        Return num tokens until stop seq, and total num tokens generated
        """
        for i in range(ntok):
            input_token_pos = ntok_already_generated + i - 1
            output_token_pos = input_token_pos + 1

            input_ids = output_tokens[:, input_token_pos : input_token_pos + 1]

            pos_id = prompt_len + ntok_already_generated + i - 1
            output_ids = self.run(input_ids, pos_id=pos_id)
            output_tokens[:, output_token_pos] = output_ids.squeeze(-1)


class PyVM_Generator(MK_Generator):
    def __init__(
        self,
        model: LlamaForCausalLM,
        interpreter: PyVM_Interpreter,
        schedule: Schedule,
    ):
        self.model = model
        self.interpreter = interpreter
        self.schedule = schedule

        self.instructions = self.schedule.get_linear_instructions()

    def run(self, input_ids: Tensor, pos_id: int):
        batch_state = BatchState(
            input_ids=input_ids,
        )

        post_embedding: BatchState = self.model.model.embed_tokens(batch_state)
        hiddens = post_embedding.hidden_states
        assert hiddens is not None
        self.schedule.globs.hidden_states[:] = hiddens
        self.schedule.globs.barriers.zero_()
        self.schedule.globs.pos_id = pos_id

        self.interpreter.interpret(self.schedule.globs, self.instructions)

        output_hiddens = self.schedule.globs.hidden_states

        post_embedding.hidden_states = output_hiddens

        post_lm_head: BatchState = self.model.lm_head(post_embedding)

        output_ids = post_lm_head.output_ids
        assert output_ids is not None
        return output_ids
