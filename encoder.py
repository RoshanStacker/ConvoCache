import numpy as np
import torch
from peft import PeftModel, PeftConfig, PeftModelForCausalLM
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaTokenizerFast, AutoModel


class AngleEncoder:
    def __init__(
        self,
        model_id: str = "SeanLee97/angle-llama-7b-nli-20231027",
        device: str = "auto",
    ):
        self.model_id = model_id
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Load model
        self.model, self.tokenizer = self._init_model(self.model_id, self.device)
        # # Type hints
        self.model: PeftModelForCausalLM
        self.tokenizer: LlamaTokenizerFast

        # Set the embedding size
        self.embedding_size = self.model.config.hidden_size

    @staticmethod
    def _init_model(angle_model_id: str, device: str):
        """
        Initialize the model and tokenizer.
        'SeanLee97/angle-llama-7b-nli-20231027' uses ~14GB of GPU memory
        :param angle_model_id: model id from huggingface
        :param device: 'cuda' or 'cpu'
        :return:
        """
        if device not in ["cuda", "cpu"]:
            raise ValueError(f"Device {device} not supported. Use 'cuda' or 'cpu'")
        config = PeftConfig.from_pretrained(angle_model_id)
        tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)
        if "cuda" in device:
            angle_model = (
                AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path)
                .bfloat16()
                .cuda()
            )
            angle_model = PeftModel.from_pretrained(angle_model, angle_model_id).cuda()
        else:
            angle_model = AutoModelForCausalLM.from_pretrained(
                config.base_model_name_or_path
            ).bfloat16()
            angle_model = PeftModel.from_pretrained(angle_model, angle_model_id)
        return angle_model, tokenizer

    @staticmethod
    def _decorate_text(text: str):
        """Prompt decoration provided by AnglE authors"""
        return f'Summarize sentence "{text}" in one word."'

    def encode(self, input: list) -> np.ndarray:
        """
        Encode the input using the AnglE model.
        :param input: Text input, dialogue history or string
        :return: embedding with shape (1, self.embedding_size)
        """
        if isinstance(input, list):
            input = input[-1]  # Use the last utterance
        tok = self.tokenizer([self._decorate_text(input)], return_tensors="pt")
        for k, v in tok.items():
            tok[k] = v.cuda() if "cuda" in self.device else v
        embedding = (
            self.model(output_hidden_states=True, **tok)
            .hidden_states[-1][:, -1]
            .float()
            .detach()
            .cpu()
            .numpy()
        )
        return embedding

    def to_cpu(self):
        self.model = self.model.cpu()
        self.device = "cpu"

    def to_cuda(self):
        if torch.cuda.is_available():
            self.model = self.model.cuda()
            self.device = "cuda"

    def destroy(self):
        self.model = None
        torch.cuda.empty_cache()


class SimCSEEncoder:
    def __init__(
        self,
        model_id: str = "princeton-nlp/sup-simcse-roberta-large",
        device: str = "auto",
    ):
        self.model_id = model_id
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Load Model
        self.model, self.tokenizer = self._init_model(self.model_id, self.device)

        # Set the embedding size
        self.embedding_size = self.model.config.hidden_size

    @staticmethod
    def _init_model(simcse_model_id: str, device: str):
        """
        Initialize the model and tokenizer.
        :param simcse_model_id: model id from huggingface
        :return:
        """
        if device not in ["cuda", "cpu"]:
            raise ValueError(f"Device {device} not supported. Use 'cuda' or 'cpu'")
        tokenizer = AutoTokenizer.from_pretrained(simcse_model_id)
        model = AutoModel.from_pretrained(simcse_model_id)
        model.to(device)

        return model, tokenizer

    def encode(self, input: list) -> np.ndarray:
        """
        Encode the input using the SimCSE model.
        :param input: Text input, dialogue history or string
        :return: embedding with shape (1, self.embedding_size)
        """
        if isinstance(input, list):
            input = input[-1]  # Use the last utterance
        tok = self.tokenizer(input, padding=True, truncation=True, return_tensors='pt')
        tok.to(self.device)
        with torch.no_grad():
            outputs = self.model(**tok)
            embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        return embedding

    def destroy(self):
        self.model = None
        torch.cuda.empty_cache()

class NoneEncoder:
    def __init__(self, embedding_size: int = 768):
        self.embedding_size = embedding_size

    def encode(self, input: list) -> np.ndarray:
        """
        Encode the input using the SimCSE model.
        :param input: Text input, dialogue history or string
        :return: embedding with shape (1, self.embedding_size)
        """
        return np.zeros((1, self.embedding_size))

    def destroy(self):
        pass

class WeightEncoder:

    def __init__(self, encoder, weights_rate=0.75, num_points=10, weights_override=None):
        """
        Encoder that computes the weighted sum of the embeddings of the dialogue history.
        A simple Conversation Encoder.
        :param encoder: AngleEncoder or SimCSEEncoder. Instance of the encoder to use.
        :param weights_rate: rate of exponential decay
        :param num_points: number of points to generate
        """
        self.encoder = encoder
        if weights_override:
            self.weights = weights_override
        else:
            self.weights = self.exponential_drop_off(r=weights_rate, num_points=num_points)
        self.embedding_size = encoder.embedding_size

    def encode(self, dialogue_history: list[str]) -> np.ndarray:
        """
        Encode the dialogue history using the weighted sum of the embeddings.
        :param dialogue_history: List of strings for the dialogue history.
        :return: embedding with shape (1, self.embedding_size)
        """
        sum = np.zeros((1, self.embedding_size))
        weights = self.weights
        if len(dialogue_history) < len(self.weights):
            range = len(dialogue_history)
            # Set new weights for the shorter dialogue history
            weights = weights[:range]
            # Normalize the weights
            weights_sum = np.sum(weights)
            weights = [weight / weights_sum for weight in weights]

        for i, weight in enumerate(weights):
            embedding = self.encoder.encode(dialogue_history[-(i+1)])
            sum += weight * embedding
        sum = sum.astype(np.float32)
        assert sum.shape == (1, self.embedding_size), f"Bad shape: {sum.shape}"
        return sum

    def destroy(self):
        self.encoder.destroy()

    @staticmethod
    def exponential_drop_off(r=1, num_points=10) -> np.ndarray:
        """
        Generate exponential drop off weights.
        :param r: rate of exponential decay
        :param num_points: number of points to generate
        :return: array of weights that sum to 1
        """
        points = np.exp(-r * np.arange(1, num_points + 1))

        # Normalize points so that their sum adds up to 1
        points /= np.sum(points)
        return points