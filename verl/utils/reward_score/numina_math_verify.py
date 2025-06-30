import math
import re

from latex2sympy2_extended import NormalizationConfig
from math_verify import LatexExtractionConfig, parse, verify

def compute_score(solution_str, ground_truth):
    retval = 0.
    try:
        #processed_str = solution_str.split("<|im_start|>assistant\n", 1)[1]
        processed_str = solution_str
        acc_score = accuracy_reward(processed_str,ground_truth)
        format_score = format_reward(processed_str)
        retval = acc_score + format_score

    except Exception as e:
        print("compute_score", e)

    return retval

'''

def compute_score(solution_str, ground_truth, gen_len):
    retval = 0.
    try:
        #processed_str = solution_str.split("<|im_start|>assistant\n", 1)[1]
        processed_str = solution_str
        acc_score = cosine_scaled_reward(processed_str, ground_truth, gen_len)
        format_score = format_reward(processed_str)
        repitition_penalty = repetition_penalty_reward(processed_str, -5, 20)
        retval = acc_score + format_score + repitition_penalty

    except Exception as e:
        print("compute_score", e)

    return retval
'''

def format_reward(solution):
    """Reward function that checks if the completion has a specific format."""
    pattern = r"^<think>.*?</think>\s*<answer>.*?</answer>$"
    match = re.match(pattern, solution, re.DOTALL | re.MULTILINE)
    think_count = solution.count("<think>")
    think_end_count = solution.count("</think>")
    answer_count = solution.count("<answer>")
    answer_end_count = solution.count("</answer>")
    if match and think_count == 1 and think_end_count == 1 and answer_count == 1 and answer_end_count == 1:
        return 0.7
    else:
        return 0.0



def accuracy_reward(content, solution):
    """Reward function that checks if the completion is the same as the ground truth."""
    gold_parsed = parse(
        solution,
        extraction_mode="first_match",
        extraction_config=[LatexExtractionConfig()],
    )
    if len(gold_parsed) != 0:
        # We require the answer to be provided in correct latex (no malformed operators)
        answer_parsed = parse(
            content,
            extraction_config=[
                LatexExtractionConfig(
                    normalization_config=NormalizationConfig(
                        nits=True,
                        malformed_operators=True,
                        basic_latex=True,
                        equations=False,
                        boxed="all",
                        units=True,
                    ),
                    # Ensures that boxed is tried first
                    boxed_match_priority=0,
                    #try_extract_without_anchor=False,
                )
            ],
            extraction_mode="first_match",
            )
        # Reward 1 if the content is the same as the ground truth, 0 otherwise
        reward = float(verify(answer_parsed, gold_parsed))
        if reward > 0:
            reward = 1.3
    else:
        # If the gold solution is not parseable, we reward 1 to skip this example
        #reward = 1.0
        reward = 1.3
        print("Failed to parse gold solution: ", solution)
    return reward


def zipngram(text: str, ngram_size: int):
    words = text.lower().split()
    return zip(*[words[i:] for i in range(ngram_size)])

def repetition_penalty_reward(completion, max_penalty, ngram_size, **kwargs):

    if max_penalty > 0:
        raise ValueError(f"max_penalty {max_penalty} should not be positive")
            
    if completion == "":
        return 0.0
    
    if len(completion.split()) < ngram_size:
        return 0.0

    ngrams = set()
    total = 0
    for ng in zipngram(completion, ngram_size):
        ngrams.add(ng)
        total += 1

    scaling = 1 - len(ngrams) / total
    reward = scaling * max_penalty

    return reward


def cosine_scaled_reward(completion, solution, gen_len, **kwargs):
    acc_score = accuracy_reward(completion, solution)
    max_len = 8192
    # Apply cosine scaling based on length
    progress = gen_len / max_len
    cosine = math.cos(progress * math.pi)

    if acc_score > 0:
        min_value = 1
        max_value = 1.5
    else:
        # Swap min/max for incorrect answers
        min_value = 0
        max_value = -2

    reward = float(min_value + 0.5 * (max_value - min_value) * (1.0 + cosine))
    return reward


