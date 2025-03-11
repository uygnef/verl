# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import re
from typing import Tuple


def build_rm_prompt(solution_str: str, ground_truth, question=None, tokenizer=None) -> str:
    # print(
    #     f'Building reward model prompt for question: {question}\nsolution: {solution_str}\nground truth: {ground_truth}'
    # )
    system = (
        'You are a helpful assistant. Your mission is to judge the correctness of an answer for a given problem. '
        + 'The question, ground truth and the answer to be judged will be enclosed within '
        + '<question> </question>, <truth> </truth> and <answer> </answer> tags, respectively. '
        + 'You should respond the answser is correct, or the answer is incorrect. '
        + "Let's think step by step and output the reasoning process within <think> </think> tags and the final judgement after that. "
        + 'i.e., <think> reasoning process here </think> the answer is incorrect.'
    )
    if question is not None:
        if isinstance(question, str):
            question = question
        elif len(question) > 0:
            question = question[0].get('content', '')
        else:
            raise ValueError(f'Unknow question type {type(question)}, {question}')
    if question is None or question == '':
        raise ValueError('Question is empty')

    messages = [
        {'role': 'system', 'content': system},
        {
            'role': 'user',
            'content': f'<question>{question}</question>\n\n<truth>{ground_truth}</truth>\n\n<answer>{solution_str}</answer>',
        },
    ]
    prompt_with_chat_template = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    ## force the model to think
    prompt_with_chat_template += '<think>'
    # print(f'Build reward model prompt: {prompt_with_chat_template}')
    return prompt_with_chat_template


def parse_reward_response(solution_str: str, reward_response: str) -> Tuple[bool, bool]:
    """
    Parse the reward response from the reward model.
    Returns if the solution is in the correct format and if the reward model judges the solution as correct.
    """
    ## check format
    solution = re.search('#### (\\-?[0-9\\.\\,]+)', solution_str)
    format_correct = solution is not None

    ## extract the final judgement from the reward response
    ## which is following the last </think> tag
    think_tag = '</think>'
    think_tag_idx = reward_response.rfind(think_tag)
    if think_tag_idx == -1:
        print(
            f'Cannot find the last {think_tag} tag in the response, solution: {solution_str}, response: {reward_response}'
        )
        return format_correct, False
    think_tag_idx += len(think_tag)
    final_judgement = reward_response[think_tag_idx:].strip().lower()
    if 'the answer is correct' in final_judgement:
        answer_correct = True
    elif 'the answer is incorrect' in final_judgement:
        answer_correct = False
    else:
        print(
            f'Cannot find the final judgement in the response, suppose incorrect, solution: {solution_str}, response: {reward_response}'
        )
        answer_correct = False

    return format_correct, answer_correct


def compute_score(
    solution_str, ground_truth, reward_response='', method='strict', score=1.0, format_score=0.0, error_score=0.0
) -> float:
    """The scoring function with reward model response.

    Args:
        solution_str: the solution text
        ground_truth: the ground truth
        reward_response: the response from the reward model
        method: the method to extract the solution, choices are 'strict' and 'flexible'
        score: the score for the correct answer and format
        format_score: the score for the correct format but incorrect answer
        error_score: the score for the incorrect format
    """
    print(
        f'Computing score with reward response for <solution> {solution_str} </solution>\n<truth> {ground_truth} <truth>\n[reward response start] {reward_response} [reward response end]'
    )
    correct_format, correct_answer = parse_reward_response(solution_str, reward_response)
    if correct_format:
        if correct_answer:
            return score
        else:
            return format_score
    else:
        return error_score
