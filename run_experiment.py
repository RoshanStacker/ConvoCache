import argparse
import os
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from tqdm import tqdm
# enable progress_apply
tqdm.pandas()

from cache import Cache, ConvoCache
from data_collector import get_huggingface_data
from evaluation.UniEval.uniEval import UniEvalDialog

PARTIAL_UTTERANCE_SPLITS = [
    0.3,
    0.6,
    0.8,
    0.9,
]  # Splits for partial utterance experiment
# Splits are Rounded to 2 decimal places for the column names. (must avoid duplicates)


def baseline_experiment(data: pd.DataFrame, convo_cache: ConvoCache):
    """
    Run the baseline experiment which completes the dialogues using the cache
    and collects the top 5 candidate results (with scores) for each dialogue
    :param data: dataframe with columns "dialogue_history" and "reference_response"
    :param convo_cache: cache with an encoder, capable of completing dialogues
    :return: dataframe with columns: "dialogue_history", "reference_response",
             "response_1", "response_2", "response_3", "response_4", "response_5",
             "distance_1", "distance_2", "distance_3", "distance_4", "distance_5"
    """

    def complete_dialogue(row):
        responses, distances = convo_cache.complete_dialogue(row["dialogue_history"])
        return pd.Series(
            {
                "response_1": responses[0],
                "response_2": responses[1],
                "response_3": responses[2],
                "response_4": responses[3],
                "response_5": responses[4],
                "distance_1": distances[0],
                "distance_2": distances[1],
                "distance_3": distances[2],
                "distance_4": distances[3],
                "distance_5": distances[4],
            }
        )

    print("Completing dialogues...")
    results = data.progress_apply(complete_dialogue, axis=1)
    return pd.concat([data, results], axis=1)


def split_utterance(dialogue_history: list[str], split: float) -> list[str]:
    """"""
    last_utterance: str = dialogue_history[-1]
    dialogue_history = dialogue_history[:-1]  # pop

    split_index = int(len(last_utterance) * split)
    while split_index > 0 and last_utterance[split_index] != " ":
        # find the closest space ' ' to the left
        split_index -= 1

    dialogue_history.append(str(last_utterance[:split_index]))
    return dialogue_history


def prefetch_experiment(data: pd.DataFrame, convo_cache: ConvoCache, split: float):
    def complete_prefetch_dialogue(row):
        partial_dialogue_history = split_utterance(row["dialogue_history"], split)
        responses, distances = convo_cache.complete_dialogue(partial_dialogue_history)

        return pd.Series(
            {
                f"partial_split": split,
                f"split_prompt": partial_dialogue_history[-1],
                "response_1": responses[0],
                "response_2": responses[1],
                "response_3": responses[2],
                "response_4": responses[3],
                "response_5": responses[4],
                "distance_1": distances[0],
                "distance_2": distances[1],
                "distance_3": distances[2],
                "distance_4": distances[3],
                "distance_5": distances[4],
            }
        )

    results = data.progress_apply(complete_prefetch_dialogue, axis=1)

    return pd.concat([data, results], axis=1)


def one_response_partial_utterance_experiment(
    data: pd.DataFrame, convo_cache: ConvoCache
):
    """Complete dialogues using a % of the last utterance"""
    splits = PARTIAL_UTTERANCE_SPLITS

    def complete_prefetch_dialogue(row, split):
        partial_dialogue_history = split_utterance(row["dialogue_history"], split)
        responses, distances = convo_cache.complete_dialogue(partial_dialogue_history)
        return pd.Series(
            {
                f"split_prompt_{split:.2f}": partial_dialogue_history[-1],
                f"response_{split:.2f}": responses[0],
                f"distance_{split:.2f}": distances[0],
            }
        )

    print(
        f"Completing dialogues with partial utterances (will repeat {len(splits)} times)..."
    )
    results = pd.concat(
        [
            data.progress_apply(complete_prefetch_dialogue, split=split, axis=1)
            for split in splits
        ],
        axis=1,
    )
    return pd.concat([data, results], axis=1)


def random_experiment(data: pd.DataFrame, convo_cache: ConvoCache):
    """Generate 5 random responses for each dialogue"""
    responses = convo_cache.cache.response_db
    random_responses = np.random.choice(responses, (len(data), 5))
    return pd.concat(
        [
            data,
            pd.DataFrame(
                {
                    "random_1": random_responses[:, 0],
                    "random_2": random_responses[:, 1],
                    "random_3": random_responses[:, 2],
                    "random_4": random_responses[:, 3],
                    "random_5": random_responses[:, 4],
                }
            ),
        ],
        axis=1,
    )


def gpt_benchmark_experiment(data: pd.DataFrame, model: str, key: str):
    """
    Use GPT to respond to the dialogues. Similar to ref and random.
    """
    from openai_chat_completion import respond_to_dialogue, setup_openai_key

    setup_openai_key(key)

    def complete_dialogue(row):
        response = respond_to_dialogue(row["dialogue_history"], model)
        return response

    responses = data.progress_apply(complete_dialogue, axis=1)

    return pd.concat([data, responses.rename(f"{model}_gpt_response")], axis=1)


def save_experiment_results(data: pd.DataFrame, output_file: Path):
    """
    Save the experiment results to a file
    :param data: dataframe with columns "dialogue_history", "reference_response",
                 "response_1", "response_2", "response_3", "response_4", "response_5",
                 "distance_1", "distance_2", "distance_3", "distance_4", "distance_5"
    :param output_file: path to the output file
    """
    data.to_csv(output_file, index=False)


def main(args):
    # 1. Load the data
    # train and test are dataframes with columns "dialogue_history" and "reference_response"
    train, test = get_huggingface_data(args.hf_dataset)

    # 2. Load the encoder
    if args.encoder == "angle":
        from encoder import AngleEncoder

        encoder = AngleEncoder()
        print("Angle encoder loaded")
    elif args.encoder == "simcse":
        from encoder import SimCSEEncoder

        encoder = SimCSEEncoder()
        print("SimCSE encoder loaded")
    elif args.encoder == "none":
        from encoder import NoneEncoder

        encoder = NoneEncoder()
        print("None encoder loaded")
    else:
        raise ValueError(f"Unknown encoder {args.encoder}")

    if args.weights_rate is not None or args.weights:
        from encoder import WeightEncoder

        if args.weights:
            encoder = WeightEncoder(encoder, weights_override=args.weights)
            print("Encoder with weights loaded %s" % args.weights)
        elif args.num_weights is None:
            encoder = WeightEncoder(encoder, args.weights_rate)  # Default to 10 weights
        else:
            encoder = WeightEncoder(encoder, args.weights_rate, args.num_weights)
        print("Encoder with weights loaded")

    # 3. Encode the data
    # This will take a while
    if args.encoder != "none":
        train_embeddings: Optional[np.ndarray] = None
        response_db: list = []
        if args.embeddings_file:
            # Load the embeddings from a file
            try:
                train_embeddings = np.load(args.embeddings_file)
                response_db = train["reference_response"].tolist()
            except FileNotFoundError:
                train_embeddings = None  # Compute the embeddings
        if train_embeddings is None or len(train_embeddings) == 0:
            prompt_embeddings = []
            for prompt, response in tqdm(
                train.itertuples(index=False, name=None),
                desc="Encoding train set",
                total=len(train),
            ):
                embedding = encoder.encode(prompt)
                prompt_embeddings.append(embedding)
                response_db.append(response)
            train_embeddings = np.concatenate(prompt_embeddings)

            if args.embeddings_file:
                np.save(args.embeddings_file, train_embeddings)

        # 4. Add train embeddings and responses to the FAISS Index
        cache = Cache(args.cache, train_embeddings, response_db)
        convo_cache = ConvoCache(encoder, cache)
    else:
        cache = None

    # 5. Complete test dialogues using cache
    experiment_data = pd.DataFrame(  # This is the common structure for all experiments
        {
            "dialogue_history": test["dialogue_history"].tolist(),
            "reference_response": test["reference_response"].tolist(),
        }
    )

    if args.experiment == "baseline":
        experiment_data = baseline_experiment(experiment_data, convo_cache)
    elif args.experiment == "partial_utterance":
        experiment_data = one_response_partial_utterance_experiment(
            experiment_data, convo_cache
        )
    elif args.experiment == "prefetch":
        experiment_data = prefetch_experiment(
            experiment_data, convo_cache, args.prefetch_split
        )
    elif args.experiment == "random":
        experiment_data = random_experiment(experiment_data, convo_cache)
    elif args.experiment == "gpt_complete":
        experiment_data = gpt_benchmark_experiment(
            experiment_data, args.gpt_model, args.openai_key
        )
    elif args.experiment == "evaluate_reference":
        # No need to do anything
        if not args.evaluate:
            print(
                "Warning: You are running the evaluate_reference experiment without evaluating "
                "the results. Nothing will happen. Set --evaluate to True to evaluate the results."
            )
    else:
        raise ValueError(f"Unknown experiment: {args.experiment}")

    # Remove the encoder to free GPU memory
    try:
        encoder.destroy()
    except AttributeError:
        pass
    del encoder
    del cache

    # 6. Evaluate the experiment with UniEval
    if args.evaluate:
        print("Evaluating the experiment with UniEval")

        try:
            uni_eval = UniEvalDialog()
        except RuntimeError as e:
            print(
                "UniEval Failed to load. GPU memory error? Saving experiment data to file.\n"
                "See exception:\n",
                e,
            )
            save_experiment_results(experiment_data, args.output_file)
            return

        def eval_column(
            response_column: pd.Series, dialogue_column_name="dialogue_history"
        ) -> pd.Series:
            """Return a series of eval scores for a given series of responses."""
            scores = []
            for response, dialogue_history in tqdm(
                zip(response_column, experiment_data[dialogue_column_name]),
                total=len(response_column),
                desc=f"Evaluating {response_column.name}",
                leave=True,
            ):
                scores.append(
                    uni_eval.evaluate(dialogue_history, response, dims=["coherence"])
                )
            return pd.Series(scores, name=f"uni_eval_{response_column.name}")

        if args.experiment == "baseline":
            for column in tqdm(
                [
                    "response_1",
                    "response_2",
                    "response_3",
                    "response_4",
                    "response_5",
                ],
                desc="Evaluating",
            ):
                eval_scores: pd.Series = eval_column(experiment_data[column])
                experiment_data = pd.concat([experiment_data, eval_scores], axis=1)

        elif args.experiment == "prefetch":
            for column in tqdm(
                [
                    "response_1",
                    "response_2",
                    "response_3",
                    "response_4",
                    "response_5",
                ],
                desc="Evaluating",
            ):
                eval_scores: pd.Series = eval_column(
                    experiment_data[column], "split_prompt"
                )
                experiment_data = pd.concat([experiment_data, eval_scores], axis=1)
        elif args.experiment == "partial_utterance":
            for column in tqdm(
                [f"response_{split:.2f}" for split in PARTIAL_UTTERANCE_SPLITS],
                desc="Evaluating",
            ):
                eval_scores: pd.Series = eval_column(experiment_data[column])
                experiment_data = pd.concat([experiment_data, eval_scores], axis=1)

        elif args.experiment == "random":
            for column in tqdm(
                [
                    "random_1",
                    "random_2",
                    "random_3",
                    "random_4",
                    "random_5",
                ],
                desc="Evaluating",
            ):
                eval_scores: pd.Series = eval_column(experiment_data[column])
                experiment_data = pd.concat([experiment_data, eval_scores], axis=1)
        elif args.experiment == "evaluate_reference":
            eval_scores: pd.Series = eval_column(experiment_data["reference_response"])
            experiment_data = pd.concat([experiment_data, eval_scores], axis=1)
        elif args.experiment == "gpt_complete":
            eval_scores: pd.Series = eval_column(
                experiment_data[f"{args.gpt_model}_gpt_response"]
            )
            experiment_data = pd.concat([experiment_data, eval_scores], axis=1)
        else:
            raise ValueError(f"Unknown experiment for eval: {args.experiment}")

    # 7. Save the experiment results
    save_experiment_results(experiment_data, args.output_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="A script to reproduce our experiment given a "
        "file with conversations"
    )

    # Input dataset options are:
    # --hf_dataset:  Huggingface dataset in form of dailyDialog. Test and Train included
    parser.add_argument(
        "--hf_dataset",
        type=str,
        help="Huggingface dataset in form of daily_dialog. Test and Train included.",
        default="daily_dialog",
        # Can also be a local file loaded with hf datasets.load_dataset
    )

    parser.add_argument(
        "-o",
        "--output_file",
        type=str,
        help="output file with the results",
        required=True,
    )

    parser.add_argument(
        "--embeddings_file",
        type=str,
        help="file to load/save the train set embeddings. Avoids recomputing them",
    )

    # Options for different experiments
    parser.add_argument(
        "-e",
        "--encoder",
        type=str,
        help="encoder to use for the model",
        default="none",
        choices=["angle", "simcse", "none"],  # None can be used for experiments
        # not requiring an encoder
    )
    parser.add_argument(
        "-w",
        "--weights",
        type=float,
        nargs="+",
        help="weights to use for the weights encoder",
        default=None,
    )
    parser.add_argument(
        "-wr",
        "--weights_rate",
        type=float,
        help="Rate of exponential decay for the weights encoder. "
        "If None, no weights are used",
        default=None,
    )
    parser.add_argument(
        "-wn",
        "--num_weights",
        type=int,
        help="number of weights to generate for the weights encoder",
        default=10,
    )

    parser.add_argument(
        "-c",
        "--cache",
        type=str,
        help="cache to use for the model",
        default="IP",
        choices=["IP", "L2"],
    )
    parser.add_argument(
        "--experiment",
        type=str,
        help="experiment to run",
        default="baseline",
        choices=[
            "baseline",
            "partial_utterance",
            "prefetch",
            "random",
            "evaluate_reference",
            "gpt_complete",
        ],
    )
    parser.add_argument(
        "--prefetch_split",
        type=float,
        help="split to use for prefetch experiment",
        default=0.9,
    )
    parser.add_argument(
        "--gpt_model",
        type=str,
        help="model to use for gpt complete experiment",
        default="gpt-3.5-turbo-1106",
    )
    parser.add_argument("--openai_key", type=str, help="OpenAI key for gpt completions")

    parser.add_argument(
        "--evaluate",
        type=bool,
        help="evaluate the experiment with unieval",
        default=False,
    )

    args = parser.parse_args()

    if args.openai_key is None:
        args.openai_key = os.getenv("OPENAI_KEY", None)

    main(args)
