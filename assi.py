from abc import ABC, abstractmethod

import torch
from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer, BitsAndBytesConfig, pipeline, Conversation

default_device_map = {
    "transformer.word_embeddings": 0,
    "transformer.word_embeddings_layernorm": 0,
    "lm_head": "cpu",
    "transformer.h": 0,
    "transformer.ln_f": 0,
    "model": "cpu",
}


class ChatBot(ABC):
    @abstractmethod
    def reset(self) -> None:
        pass

    def set_initial_messages(self, messages) -> None:
        self.initial_messages = messages

    @abstractmethod
    def prompt(self, prompt: str) -> str:
        pass


def create_quantized_model(model_name=None, device_map=None):
    if model_name is None:
        model_name = "mistralai/Mistral-7B-Instruct-v0.2"
    if device_map is None:
        device_map = default_device_map
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        llm_int8_enable_fp32_cpu_offload=True,
    )
    print("Loading: ", model_name)
    return AutoAWQForCausalLM.from_pretrained(
        model_name,
        device_map=device_map,
        quantization_config=bnb_config,
    )


def get_quantized_model(name=None) -> ChatBot:
    if name is None:
        name = "mistralai/Mistral-7B-Instruct-v0.2"
    model = create_quantized_model(name)
    tokenizer = AutoTokenizer.from_pretrained(name)
    return QuantizedAssi(model, tokenizer)


class QuantizedAssi(ChatBot):
    def __init__(self, model, tokenizer, initial_messages=None):
        if initial_messages is None:
            initial_messages = []
        self.initial_messages = initial_messages
        self.chat = initial_messages
        self.model = model
        self.tokenizer = tokenizer

    def reset(self) -> None:
        self.chat = self.initial_messages

    def prompt(self, prompt):
        self.chat.append({"role": "user", "content": prompt})
        model_inputs = self.tokenizer.apply_chat_template(self.chat, return_tensors="pt")
        generated_ids = self.model.generate(model_inputs, max_new_tokens=500, do_sample=True)
        decoded = self.tokenizer.batch_decode(generated_ids)
        answer = decoded[0].split("[/INST]")[-1].strip()[:-4]
        self.chat.append({"role": "assistant", "content": answer})
        return answer


class PipelinedAssi(ChatBot):
    def __init__(self, botname=None, initial_messages=None):
        if botname is None:
            self.botname = "mistralai/Mistral-7B-Instruct-v0.2"
        else:
            self.botname = botname
        if initial_messages is None:
            initial_messages = []
        self.initial_messages = initial_messages
        self.chat = initial_messages
        self.conversation = Conversation()
        self.tokenizer = AutoTokenizer.from_pretrained(self.botname)
        self.pipeline = pipeline("conversational",
                                 self.botname,
                                 device_map=default_device_map,
                                 tokenizer=self.tokenizer,
                                 )

    def reset(self):
        self.chat = self.initial_messages
        self.conversation = Conversation()
        for message in self.chat:
            self.conversation.add_message(message)

    def prompt(self, prompt):
        self.conversation.add_message({"role": "user", "content": prompt})
        self.conversation = self.pipeline(self.conversation, max_new_tokens=100)
        return self.conversation.messages[-1]['content']
