from abc import abstractmethod


class AbstractEvaluationReferenceFree:
    """
    Abstract class for evaluation methods that do not require a reference.
    """

    # def __init__(self, device: str = "auto"):
    #     """
    #     Initialize the evaluation.
    #
    #     :param device: The device to use for the evaluation. Either "cuda" or "cpu".
    #     """
    #     # SAMPLE FOR TORCH DEVICE
    #     if device not in ["cuda", "cpu"]:  # auto
    #         device = "cuda" if torch.cuda.is_available() else "cpu"
    #     self.device = device

    @abstractmethod
    def evaluate(self, dialogue_history: list[str] or str, response: str) -> float or dict:
        """
        Evaluate the model on the data.
        Implementations may have more arguments, but using only these will evaluate with defaults.

        :param dialogue_history: The dialogue history. A list of strings. with alternating speaker.
        :param response: The response to evaluate for dialogue quality.
        :return: The evaluation score. Or a dictionary of scores depending on
                 the model and settings.
        """
        pass