import os
import time
import openai
from .OPENAI_KEY import openai_key

openai.api_key = openai_key


def prompt_single(
    prompt,
    system_prompt=None,
    model="gpt-3.5-turbo",
    max_tokens=1024,
    temperature=0.7,
    top_p=1,
    frequency_penalty=0,
    presence_penalty=0,
):
    assert type(prompt) is str

    if system_prompt is not None:
        message = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]
    else:
        message = [{"role": "user", "content": prompt}]

    response = openai.ChatCompletion.create(
        n=1,
        model=model,
        messages=message,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        frequency_penalty=frequency_penalty,
        presence_penalty=presence_penalty,
        request_timeout=180,
    )
    return response.choices[0].message.content, response


def prompt_request(
    prompt,
    system_prompt=None,
    model="gpt-3.5-turbo",
    output_max_tokens=1024,
    auto_truncate=True,
    max_retries=5,
    temperature=0,
    top_p=1,
    frequency_penalty=0,
    presence_penalty=0,
    log=True,
    log_dir="prompt_logs/",
    log_name=None,
):
    """
    Robust version of prompt that handles errors by retrying.
    """

    assert type(prompt) is str
    assert type(system_prompt) is str or system_prompt is None

    if model == "gpt-4":
        max_allowed_tokens = 1024 * 8 - 1
    else:
        max_allowed_tokens = 1024 * 4 - 1

    tok_p_word = 1.4  # tokens per word

    if not auto_truncate:

 

        req_total_tokens = (len(prompt.split())) * tok_p_word + output_max_tokens

        # dynamic adjust max tokens (output only) based on prompt length up to a max
        if req_total_tokens > max_allowed_tokens:
            output_max_tokens = max_allowed_tokens - len(prompt.split()) * tok_p_word
        else:
            output_max_tokens = output_max_tokens
        output_max_tokens = int(output_max_tokens)
    else:
        # Truncate prompt
        prompt_tokens_allowed = max_allowed_tokens - output_max_tokens
        prompt_words_allowed = prompt_tokens_allowed // tok_p_word
        current_words = len(prompt.split())

        # Estimate character per word
        char_per_word = len(prompt) / current_words

        # Truncate prompt at word level if needed
        if current_words > prompt_words_allowed:
            char_limit = int(0.9 * prompt_words_allowed * char_per_word)
            prompt = prompt[:char_limit]

    ret = None
    i = 0
    resp = None

    while ret is None and i < max_retries:
        try:
            ret, resp = prompt_single(
                prompt=prompt,
                system_prompt=system_prompt,
                model=model,
                max_tokens=output_max_tokens,
                temperature=temperature,
                top_p=top_p,
                frequency_penalty=frequency_penalty,
                presence_penalty=presence_penalty,
            )
            assert type(ret) == str
        except KeyboardInterrupt:
            raise KeyboardInterrupt
        except Exception as e:
            print("Prompt failed. Retrying. Error: {}".format(e))
            i += 1

    if ret is None:
        raise Exception(
            "Prompt failed after {} retries.\n Response: {}".format(
                max_retries, str(resp)
            )
        )
    else:
        # log
        if log:
            if log_name is None:
                # generate timestamped log name
                log_name = time.strftime("%Y%m%d-%H%M%S") + ".txt"
            path = os.path.join(log_dir, log_name)
            # create log dir if it doesn't exist
            if not os.path.exists(log_dir):
                os.makedirs(log_dir)
            with open(path, "w") as f:
                # log prompt, system prompt, and response
                f.write("PROMPT:\n")
                f.write(prompt)
                f.write("\n\nSYSTEM PROMPT:\n")
                f.write(system_prompt)
                f.write("\n\nRESPONSE:\n")
                f.write(ret)

        return ret


if __name__ == "__main__":
    print(
        openai.ChatCompletion.create(
            n=1,
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "Hi!"}],
            max_tokens=1024,
            temperature=0,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
        )
        .choices[0]
        .message.content
    )

    print(
        prompt_request(model="gpt-4", prompt="Hi!", system_prompt="You are a pirate!")
    )
