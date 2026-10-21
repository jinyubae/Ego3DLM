import random
import re
from typing import List, Optional, Tuple, Union

import torch
from torch import Tensor, nn
from transformers import AutoTokenizer, GPT2LMHeadModel

# Prompts of the validation / evaluation tasks
SCENE_QA_PROMPT = ('Answer the question with <3D_Scene_Placeholder> scene and the corresponding <Video_Placeholder> '
                   'video frames: <Question_Placeholder>')
MOTION_PROMPT = ('Perform human motion tracking and prediction. First, analyze the spatial constraints (front, left, '
                 'right) and the optimal direction from the scene. Then, output current and future human motion token '
                 'sequences, and describe current and future human motion conditioned on the given scene, observed '
                 'video CLIP embeddings, and observed three-points features. Input scene: <3D_Scene_Placeholder>. '
                 'Input video CLIP embeddings: <Video_Placeholder>. Input three-points features: <TP_Placeholder>')


class MotionLanguageModel(nn.Module):
    """GPT-2 over text and motion tokens, with the 3D scene, video and three-point embeddings injected
    at their placeholders in the prompt."""

    def __init__(
        self,
        model_path: str,
        model_type: str = "gpt2_custom_motion_3d_feature",
        motion_codebook_size: int = 512,
        motion_codebook_num: int = 2,
        down_t: int = 4,
        predict_ratio: float = 1.0,
        max_length: int = 256,
    ) -> None:
        super().__init__()
        if model_type != "gpt2_custom_motion_3d_feature":
            raise ValueError(f"Unsupported model_type: {model_type}")

        self.m_codebook_size = motion_codebook_size
        self.m_codebook_num = motion_codebook_num
        self.max_length = max_length
        self.down_t = down_t
        self.predict_ratio = predict_ratio

        self.tokenizer = AutoTokenizer.from_pretrained(model_path, legacy=True)
        self.language_model = GPT2LMHeadModel.from_pretrained(model_path)

        # Motion tokens (one vocabulary shared by the codebooks) and the scene / video delimiters
        motion_tokens = [f'<motion_id_{i}>' for i in range(int(self.m_codebook_size // self.m_codebook_num))]
        motion_tokens += ['<motion_start>', '<motion_end>', '<motion_mask>']
        self.tokenizer.add_tokens(motion_tokens)
        self.tokenizer.add_special_tokens({"additional_special_tokens": ['<scene>', '<scene/>']})
        self.tokenizer.add_special_tokens({"additional_special_tokens": ['<video>', '<video/>']})
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.language_model.resize_token_embeddings(len(self.tokenizer))

    @property
    def device(self):
        return self.language_model.device

    def _add_special_tokens(self, tokens):
        self.tokenizer.add_special_tokens({"additional_special_tokens": tokens})
        self.language_model.resize_token_embeddings(len(self.tokenizer))

    # Special tokens added from Stage II on (see the train / validate scripts for the order)
    def init_tp_special_token(self):
        self._add_special_tokens(['<tp>', '<tp/>'])

    def init_text_special_token(self):
        self._add_special_tokens(['<CUR_MOTION_DESC>', '<CUR_MOTION_DESC/>', '<FUT_MOTION_DESC>', '<FUT_MOTION_DESC/>'])

    def init_cot_special_token(self):
        self._add_special_tokens(['<SCENE_REASONING>', '<SCENE_REASONING/>'])

    def forward(self,
                texts: List[str],
                motion_tokens: Tensor,
                lengths: List[int],
                tasks: List[dict],
                pc_token_embeds=None,
                video_token_embeds=None,
                video_lengths=None,
                scene_q: Optional[List[str]] = None,
                scene_a: Optional[List[str]] = None,
                future_motion_strings: Optional[List[str]] = None,
                future_texts: Optional[List[str]] = None,
                three_points_embeds=None,
                three_points_lengths: Optional[List[int]] = None,
                obstacle_cot_answer: Optional[List[str]] = None):
        """Teacher-forced LM loss on the task outputs (Stages I and II)."""
        self.tokenizer.padding_side = "right"

        motion_strings = self.motion_token_to_string(motion_tokens, lengths)
        batch_size = len(lengths)
        if future_motion_strings is None:
            future_motion_strings = ["" for _ in range(batch_size)]
        if future_texts is None:
            future_texts = ["" for _ in range(batch_size)]

        inputs, outputs = self.template_fulfill(
            tasks,
            motion_strings,
            texts,
            scene_questions=scene_q,
            scene_answers=scene_a,
            future_motion_strings=future_motion_strings,
            future_texts=future_texts,
            obstacle_cot_answers=obstacle_cot_answer,
        )
        prompts, output_prompts = [], []
        for i in range(len(inputs)):
            # current and future motion form one motion span in the output
            outputs[i] = outputs[i].replace('<motion_end><motion_start>', '')
            prompts.append(inputs[i] + ' \n ')
            output_prompts.append(outputs[i] + self.tokenizer.eos_token)

        inputs_embeds, attention_mask, labels, _ = self.build_batch_with_scene_and_tp_embedding(
            prompts=prompts,
            outputs_text=output_prompts,
            pc_token_embeds=pc_token_embeds,
            three_points_token_embeds=three_points_embeds,
            three_points_lens=three_points_lengths,
            video_token_embeds=video_token_embeds,
            video_lengths=video_lengths,
            device=motion_tokens.device)
        return self.language_model(inputs_embeds=inputs_embeds, attention_mask=attention_mask, labels=labels)

    def generate_direct(self,
                        prompts: List[str],
                        max_new_tokens: int,
                        keep_special_tokens: bool,
                        pc_token_embeds=None,
                        video_token_embeds=None,
                        video_lengths=None,
                        three_points_embed=None,
                        three_points_lengths: Optional[List[int]] = None,
                        num_return_sequences: int = 1) -> List[str]:
        """Sample a continuation of every prompt; returns the decoded strings."""
        self.tokenizer.padding_side = 'left'
        prompts = [prompt + " \n " for prompt in prompts]
        inputs_embeds, attention_mask, _, _ = self.build_batch_with_scene_and_tp_embedding(
            prompts=prompts,
            outputs_text=['' for _ in range(len(prompts))],
            pc_token_embeds=pc_token_embeds,
            video_token_embeds=video_token_embeds,
            video_lengths=video_lengths,
            three_points_token_embeds=three_points_embed,
            three_points_lens=three_points_lengths,
            device=self.device,
        )
        try:
            outputs = self.language_model.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                pad_token_id=self.tokenizer.pad_token_id,
                do_sample=True,
                max_new_tokens=max_new_tokens,
                num_return_sequences=num_return_sequences,
            )
        finally:
            torch.cuda.empty_cache()
        return self.tokenizer.batch_decode(outputs, skip_special_tokens=not keep_special_tokens)

    def answer_scene_questions(self, scene_questions, pc_token_embeds, video_token_embeds, video_lengths):
        """Stage I validation: answer a question about the 3D scene (semantic or spatial)."""
        prompts = [self.placeholder_fulfill(SCENE_QA_PROMPT, scene_question=q) for q in scene_questions]
        return self.generate_direct(prompts,
                                    max_new_tokens=128,
                                    keep_special_tokens=False,
                                    pc_token_embeds=pc_token_embeds,
                                    video_token_embeds=video_token_embeds,
                                    video_lengths=video_lengths)

    def generate_motion_and_text(self, batch_size, pc_token_embeds, video_token_embeds, video_lengths,
                                 three_points_embeds, three_points_lengths, num_samples):
        """Stage II / III validation: scene reasoning, current + future motion and their descriptions,
        `num_samples` generations per sample. Returns the motion tokens and the raw generated texts."""
        prompts = [self.placeholder_fulfill(MOTION_PROMPT)] * batch_size
        outputs_string = self.generate_direct(prompts,
                                              max_new_tokens=256,
                                              keep_special_tokens=True,
                                              pc_token_embeds=pc_token_embeds,
                                              video_token_embeds=video_token_embeds,
                                              video_lengths=video_lengths,
                                              three_points_embed=three_points_embeds,
                                              three_points_lengths=three_points_lengths,
                                              num_return_sequences=num_samples)
        return self.motion_string_to_token(outputs_string), outputs_string

    @staticmethod
    def _normalize_grpo_prompts(prompts: List[str]) -> List[str]:
        # Keep sampling and scoring contexts identical in GRPO.
        return [("" if p is None else str(p)).rstrip() + " \n " for p in prompts]

    def sample_group(self,
                     prompts: List[str],
                     pc_token_embeds=None,
                     video_token_embeds=None,
                     video_lengths=None,
                     three_points_embed=None,
                     three_points_lengths: Optional[List[int]] = None,
                     num_samples: int = 6,
                     max_new_tokens: int = 128,
                     temperature: float = 1.0,
                     top_p: float = 0.9,
                     top_k: int = 50) -> List[str]:
        """GRPO: sample `num_samples` outputs per prompt."""
        self.tokenizer.padding_side = 'left'
        prompts = self._normalize_grpo_prompts(prompts)
        inputs_embeds, attention_mask, _, _ = self.build_batch_with_scene_and_tp_embedding(
            prompts=prompts,
            outputs_text=['' for _ in range(len(prompts))],
            pc_token_embeds=pc_token_embeds,
            video_token_embeds=video_token_embeds,
            video_lengths=video_lengths,
            three_points_token_embeds=three_points_embed,
            three_points_lens=three_points_lengths,
            device=self.device,
        )
        try:
            gen_out = self.language_model.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                pad_token_id=self.tokenizer.pad_token_id,
                do_sample=True,
                max_new_tokens=max_new_tokens,
                num_return_sequences=num_samples,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
            )
        finally:
            # release the generation cache before the log-prob forward passes
            torch.cuda.empty_cache()
        return self.tokenizer.batch_decode(gen_out, skip_special_tokens=False)

    def compute_logprobs(self,
                         prompts: List[str],
                         outputs_text: List[str],
                         pc_token_embeds=None,
                         video_token_embeds=None,
                         video_lengths=None,
                         three_points_embed=None,
                         three_points_lengths: Optional[List[int]] = None):
        """GRPO: length-normalized log-probability of each output given its prompt."""
        prompts = self._normalize_grpo_prompts(prompts)
        self.tokenizer.padding_side = "right"
        inputs_embeds, attention_mask, labels, _ = self.build_batch_with_scene_and_tp_embedding(
            prompts=prompts,
            outputs_text=outputs_text,
            pc_token_embeds=pc_token_embeds,
            video_token_embeds=video_token_embeds,
            video_lengths=video_lengths,
            three_points_token_embeds=three_points_embed,
            three_points_lens=three_points_lengths,
            device=self.device,
        )
        logits = self.language_model(inputs_embeds=inputs_embeds, attention_mask=attention_mask,
                                     return_dict=True).logits  # [B, T, V]
        V = logits.size(-1)
        shift_logits = logits[..., :-1, :].contiguous()  # [B, T-1, V]
        shift_labels = labels[..., 1:].contiguous()  # [B, T-1]
        mask = shift_labels != -100

        loss_fct = torch.nn.CrossEntropyLoss(reduction='none', ignore_index=-100)
        neg_logprobs = loss_fct(shift_logits.view(-1, V), shift_labels.view(-1))
        token_log_probs = -neg_logprobs.view(labels.size(0), -1) * mask

        valid_lens = mask.sum(dim=1).clamp(min=1)
        return token_log_probs.sum(dim=1) / valid_lens

    def motion_token_to_string(self, motion_token: Tensor, lengths: List[int]):
        """
        motion_token: B tensors (T, C) of code indices, C = number of codebooks, each index in [0, K)
        lengths: valid number of time steps per sample
        returns: B strings "<motion_start><motion_id_..>...<motion_end>"
        """
        C = int(self.m_codebook_num)
        K = int(self.m_codebook_size // C)
        out_strings = []
        for i in range(len(motion_token)):
            seq = motion_token[i][:int(lengths[i])]
            body_tokens = []
            for t in range(seq.size(0)):
                for cb in range(C):
                    idx = int(seq[t, cb].item())
                    if not (0 <= idx < K):
                        raise ValueError(f"motion_token[{i},{t},{cb}]={idx} out of range [0,{K})")
                    body_tokens.append(f"<motion_id_{idx}>")
            out_strings.append("<motion_start>" + "".join(body_tokens) + "<motion_end>")
        return out_strings

    def motion_string_to_token(self, motion_string: List[str]):
        """Parse the motion span of every string into a (T, C) tensor of code indices."""
        motion_tokens = []
        C = int(self.m_codebook_num)
        for s in motion_string:
            s = s.replace('<|endoftext|>', '')
            middle = self.get_middle_str(s, "<motion_start>", "<motion_end>")
            matches = re.findall(r"<motion_id_(\d+)>", middle)
            T = len(matches) // C
            tokens = torch.zeros((T, C), dtype=torch.long, device=self.device)
            for t in range(T):
                for cb in range(C):
                    tokens[t, int(cb)] = int(matches[t * C + cb])
            motion_tokens.append(tokens)
        return motion_tokens

    @staticmethod
    def placeholder_fulfill(prompt: str, motion_string: str = "", text: str = "", scene_question: str = "",
                            scene_answer: str = "", future_motion_string: str = "", future_text: str = "",
                            obstacle_cot_answer: str = ""):
        return (
            prompt.replace("<Caption_Placeholder>", text)
                .replace("<Motion_Placeholder>", motion_string)
                .replace("<3D_Scene_Placeholder>", "<scene>" + "<3D_Scene_Placeholder>" + "<scene/>")
                .replace("<Video_Placeholder>", "<video>" + "<Video_Placeholder>" + "<video/>")
                .replace("<TP_Placeholder>", "<tp>" + "<TP_Placeholder>" + "<tp/>")
                .replace("<Question_Placeholder>", scene_question)
                .replace("<Answer_Placeholder>", scene_answer)
                .replace("<Future_Motion_Placeholder>", future_motion_string)
                .replace("<Future_Text_Placeholder>", future_text)
                .replace("<Scene_CoT_Placeholder>", obstacle_cot_answer)
        )

    def template_fulfill(self,
                         tasks,
                         motion_strings,
                         texts,
                         scene_questions: Optional[List[str]] = None,
                         scene_answers: Optional[List[str]] = None,
                         future_motion_strings: Optional[List[str]] = None,
                         future_texts: Optional[List[str]] = None,
                         obstacle_cot_answers: Optional[List[str]] = None):
        """Fill the input / output templates of the training tasks."""
        inputs = []
        outputs = []
        for i in range(len(tasks)):
            input_template = random.choice(tasks[i]['input'])
            output_template = random.choice(tasks[i]['output'])
            future_motion_str = future_motion_strings[i] if future_motion_strings else ""
            future_text_str = future_texts[i] if future_texts else ""
            scene_question = ""
            scene_answer = ""
            obstacle_cot_answer = ""
            if obstacle_cot_answers is not None and i < len(obstacle_cot_answers) and obstacle_cot_answers[i] is not None:
                obstacle_cot_answer = obstacle_cot_answers[i]
            if scene_questions is not None and i < len(scene_questions) and scene_questions[i] is not None:
                scene_question = scene_questions[i]
            if scene_answers is not None and i < len(scene_answers) and scene_answers[i] is not None:
                scene_answer = scene_answers[i]
            inputs.append(
                self.placeholder_fulfill(input_template, motion_strings[i], texts[i], scene_question, scene_answer,
                                         future_motion_str, future_text_str))
            outputs.append(
                self.placeholder_fulfill(output_template, motion_strings[i], texts[i], scene_question, scene_answer,
                                         future_motion_str, future_text_str, obstacle_cot_answer))
        return inputs, outputs

    def build_batch_with_scene_and_tp_embedding(
        self,
        prompts: List[str],
        outputs_text: List[str],
        pc_token_embeds: Optional[Union[torch.Tensor, List[Optional[torch.Tensor]]]],
        video_token_embeds: Optional[Union[torch.Tensor, List[Optional[torch.Tensor]]]],
        video_lengths: Optional[Union[torch.Tensor, List[Optional[torch.Tensor]]]],
        three_points_token_embeds: Optional[Union[torch.Tensor, List[Optional[torch.Tensor]]]] = None,
        three_points_lens: Optional[List[Optional[int]]] = None,
        device: torch.device = torch.device("cuda"),
    ):
        """
        Embed prompt + output, inserting the scene / video / three-point embeddings at their placeholders.
        return:
            inputs_embeds: (B, Lmax, H)
            attention_mask: (B, Lmax)  (valid token=1, padding=0)
            labels: (B, Lmax)          (prompt/placeholder = -100)
            prompt_lens: list[int]     (position where output starts in each sample)
        """
        tokenizer = self.tokenizer
        batch_size = len(prompts)

        def normalize_list(x, B):
            if isinstance(x, torch.Tensor):
                return [x[i] for i in range(x.size(0))]
            return x if x is not None else [None] * B

        pc_list = normalize_list(pc_token_embeds, batch_size)
        vi_list = normalize_list(video_token_embeds, batch_size)
        tp_list = normalize_list(three_points_token_embeds, batch_size)

        if len(outputs_text) != batch_size:
            raise ValueError("outputs_text length must match prompts length")
        if len(pc_list) != batch_size:
            raise ValueError("pc_token_embeds length must match batch size")
        if tp_list is not None and len(tp_list) != batch_size:
            raise ValueError("three_points_token_embeds length must match batch size")
        if three_points_lens is not None and len(three_points_lens) != batch_size:
            raise ValueError("three_points_lens length must match batch size")

        input_embeddings = self.language_model.get_input_embeddings()
        embed_dim = input_embeddings.weight.size(1)

        SC_PLACEHOLDER = "<3D_Scene_Placeholder>"
        VI_PLACEHOLDER = "<Video_Placeholder>"
        TP_PLACEHOLDER = "<TP_Placeholder>"

        seq_embeds, seq_labels, seq_attn, prompt_lens = [], [], [], []

        def wte(ids: List[int]) -> torch.Tensor:
            if not ids:
                return torch.empty(0, embed_dim, device=device)
            ids_t = torch.tensor(ids, dtype=torch.long, device=device)
            return input_embeddings(ids_t)  # (T, H)

        def parse_with_placeholders(text: str) -> List[Tuple[str, Optional[str]]]:
            """
            return : list[(segment_text, None | 'SC' | 'VI' | 'TP')]
            * segment_text is the text between placeholders; the tag is the placeholder that immediately follows
            example : 'A <SC> B <TP> C <VI> D' -> [('A ', 'SC'), (' B ', 'TP'), (' C ', 'VI'), (' D', None)]
            """
            marks = []
            i_sc = text.find(SC_PLACEHOLDER)
            i_vi = text.find(VI_PLACEHOLDER)
            i_tp = text.find(TP_PLACEHOLDER)
            if i_sc != -1:
                marks.append((i_sc, 'SC', SC_PLACEHOLDER))
            if i_vi != -1:
                marks.append((i_vi, 'VI', VI_PLACEHOLDER))
            if i_tp != -1:
                marks.append((i_tp, 'TP', TP_PLACEHOLDER))
            marks.sort(key=lambda x: x[0])

            out = []
            cur = 0
            for pos, tag, token in marks:
                out.append((text[cur:pos], tag))
                cur = pos + len(token)
            out.append((text[cur:], None))
            return out

        pad_emb = wte([self.tokenizer.pad_token_id])
        for i in range(batch_size):
            prompt_segments = []
            for seg_text, next_ph in parse_with_placeholders(prompts[i]):
                seg_emb = wte(tokenizer(seg_text, add_special_tokens=False)["input_ids"])
                if seg_emb.size(0) > 0:
                    prompt_segments.append(seg_emb)

                if next_ph == 'SC':
                    pc_emb_i = pc_list[i]
                    if pc_emb_i is None:
                        raise ValueError("3D scene placeholder present but pc embedding missing")
                    pc_emb_i = pc_emb_i.to(device)
                    if pc_emb_i.dim() != 2 or pc_emb_i.size(-1) != embed_dim:
                        raise ValueError(f"pc embedding must be (T,H) with H=={embed_dim}")
                    if pc_emb_i.size(0) > 0:
                        prompt_segments.append(pc_emb_i)
                elif next_ph == 'VI':
                    video_emb_i = vi_list[i]
                    if video_emb_i is None:
                        raise ValueError("video placeholder present but video embedding missing")
                    video_emb_i = video_emb_i.to(device)
                    if video_emb_i.dim() != 2 or video_emb_i.size(-1) != embed_dim:
                        raise ValueError(f"video embedding must be (T,H) with H=={embed_dim}")
                    video_emb_i = video_emb_i[:video_lengths[i]]
                    if video_emb_i.size(0) > 0:
                        prompt_segments.append(video_emb_i)
                elif next_ph == 'TP':
                    tp_emb_i = tp_list[i] if tp_list is not None else None
                    if tp_emb_i is None:
                        raise ValueError("TP placeholder present but three_points embedding missing")
                    tp_emb_i = tp_emb_i.to(device)
                    if tp_emb_i.dim() != 2 or tp_emb_i.size(-1) != embed_dim:
                        raise ValueError(f"three_points embedding must be (T,H) with H=={embed_dim}")
                    tgt_len = three_points_lens[i] if (three_points_lens is not None and three_points_lens[i] is not None) else tp_emb_i.size(0)
                    tgt_len = max(0, min(int(tgt_len), tp_emb_i.size(0)))
                    tp_emb_i = tp_emb_i[:tgt_len]
                    if tp_emb_i.size(0) > 0:
                        prompt_segments.append(tp_emb_i)

            prompt_emb = torch.cat(prompt_segments, dim=0) if len(prompt_segments) > 0 \
                        else torch.empty(0, embed_dim, device=device)

            out_ids = tokenizer(outputs_text[i], add_special_tokens=False)["input_ids"]
            if len(out_ids) > self.max_length:
                out_ids = out_ids[:self.max_length]
            out_emb = wte(out_ids)

            inp_components = [prompt_emb]
            if out_emb.size(0) > 0:
                inp_components.append(out_emb)
            inp_emb_i = torch.cat(inp_components, dim=0)
            # keep the last max_length positions (the output is never truncated here)
            if inp_emb_i.shape[0] >= self.max_length:
                inp_emb_i = inp_emb_i[-self.max_length:, ...]

            L_out = out_emb.size(0)
            L_all = min(prompt_emb.size(0) + L_out, self.max_length)
            L_prompt = L_all - L_out

            labels_i = torch.full((L_all,), -100, dtype=torch.long, device=device)
            if L_out > 0:
                labels_i[L_prompt:] = torch.tensor(out_ids, dtype=torch.long, device=device)

            seq_embeds.append(inp_emb_i)
            seq_labels.append(labels_i)
            seq_attn.append(torch.ones(L_all, dtype=torch.long, device=device))
            prompt_lens.append(L_prompt)

        max_len = self.max_length
        inputs_embeds = pad_emb.expand(batch_size, max_len, -1).clone().to(device)
        attention_mask = torch.zeros(batch_size, max_len, dtype=torch.long, device=device)
        labels = torch.full((batch_size, max_len), -100, dtype=torch.long, device=device)

        for i in range(batch_size):
            L = seq_embeds[i].size(0)
            if self.tokenizer.padding_side == "right":
                inputs_embeds[i, :L] = seq_embeds[i]
                attention_mask[i, :L] = seq_attn[i]
                labels[i, :L] = seq_labels[i]
            else:
                inputs_embeds[i, -L:] = seq_embeds[i]
                attention_mask[i, -L:] = seq_attn[i]
                labels[i, -L:] = seq_labels[i]

        return inputs_embeds, attention_mask, labels, prompt_lens

    @staticmethod
    def get_middle_str(content, startStr, endStr):
        """Motion span of a generated string; a single zero token if it has none."""
        try:
            startIndex = content.index(startStr)
            if startIndex >= 0:
                startIndex += len(startStr)
            endIndex = content.index(endStr)
        except ValueError:
            return '<motion_start><motion_id_0><motion_id_0><motion_end>'
        return '<motion_start>' + content[startIndex:endIndex] + '<motion_end>'
