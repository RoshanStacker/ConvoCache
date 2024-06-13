from datasets import load_dataset
import pandas as pd


def convert_conversations_to_prompt_response_pairs(conversations: list[list[str]]) -> pd.DataFrame:
    """
    Converts a list of conversations into a dataframe of prompt response pairs

    :return: a dataframe with columns "dialogue_history" and "reference_response"
             "dialogue_history" is a list of strings for alternating speakers
                                before the reference response
             "reference_response" is a string for the response
    """

    prompts = []
    responses = []
    for conversation in conversations:
        for i in range(1, len(conversation)):
            prompts.append(conversation[:i])
            responses.append(conversation[i])
    prompt_response_pairs = pd.DataFrame(
        {
            "dialogue_history": prompts,
            "reference_response": responses,
        }
    )
    return prompt_response_pairs


def get_huggingface_data(name: str):
    dataset = load_dataset(name)
    train = dataset["train"]
    test = dataset["test"]

    if "dialog" in train.column_names:
        train = train["dialog"]
    if "dialog" in test.column_names:
        test = test["dialog"]

    assert isinstance(train[0], list) and isinstance(
        test[0], list
    ), "Data is not in the expected format"

    train = convert_conversations_to_prompt_response_pairs(train)
    test = convert_conversations_to_prompt_response_pairs(test)

    return train, test