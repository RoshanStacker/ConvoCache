import openai
import backoff
import logging

def setup_openai_key(key):
    openai.api_key = key

@backoff.on_exception(
    backoff.expo,
    (
        openai.error.RateLimitError,
        openai.error.ServiceUnavailableError,
        openai.error.Timeout,
        openai.error.APIError
    ),
    max_time=60 * 60 * 24,
    max_tries=1000,
    logger=logging.getLogger(__name__),
    backoff_log_level=logging.WARNING,
)
def _openai_ChatCompletion_backoff(*args, **kwargs):
    return openai.ChatCompletion.create(*args, **kwargs)

def respond_to_dialogue(dialogue_history, model, temperature=0):
    """
    Respond to a dialogue using the OpenAI ChatCompletion API.
    """
    prompt = ("You are an advanced student of English conducting a practice conversation. "
              "Stay in character.")

    messages = []
    for i, message in enumerate(reversed(dialogue_history)):
        messages.append({"role": "user" if i % 2 == 0 else "assistant",
                         "content": message})
    messages = messages[::-1]  # reverse
    messages.insert(0, {"role": "system", "content": prompt})

    response = _openai_ChatCompletion_backoff(
        model=model,
        messages=messages,
        max_tokens=150,
        temperature=temperature,
    )
    return response.choices[0].message.content
