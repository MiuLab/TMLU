import argparse
import configparser
import json
import logging
import os
from collections.abc import Callable
from pprint import pprint

from datasets import load_dataset

from model import Anthropic_LM, Google_LM, HFLM_transformers, HFLM_vLLM, OpenAI_LM
from template import anthropic_template, hf_template, openai_template
from utils import check_ans, check_ans_cot

config = configparser.ConfigParser()
config.read('config.ini')
logging.basicConfig(
    level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

SUBSETS = [
    'AST_chinese',
    'AST_mathematics',
    'AST_biology',
    'AST_chemistry',
    'AST_physics',
    'AST_civics',
    'AST_geography',
    'AST_history',
    'GSAT_chinese',
    'GSAT_chemistry',
    'GSAT_biology',
    'GSAT_physics',
    'GSAT_earth_science',
    'GSAT_mathematics',
    'GSAT_geography',
    'GSAT_history',
    'GSAT_civics',
    'CAP_chinese',
    'CAP_mathematics',
    'CAP_biology',
    'CAP_history',
    'CAP_civics',
    'CAP_geography',
    'CAP_physics',
    'CAP_chemistry',
    'CAP_earth_science',
    'driving_rule',
    'basic_traditional_chinese_medicine',
    'clinical_traditional_chinese_medicine',
    'lawyer_qualification',
    'nutritionist',
    'tour_guide',
    'tour_leader',
    'taiwan_tourist_resources',
    'clinical_psychologist',
    'teacher_qualification',
    'accountant',
]


def parse_args():
    parser = argparse.ArgumentParser(description='Run TMLU-Eval')
    parser.add_argument(
        '--backend',
        choices=['hf', 'vllm', 'openai', 'anthropic', 'google', 'custom_api'],
        required=True,
        help='The backend type. '
        "Options: ['hf', 'vllm', 'openai', 'anthropic', 'google', 'custom_api']",
    )
    parser.add_argument(
        '--model',
        type=str,
        required=True,
        help='Model name.',
    )
    parser.add_argument(
        '--cache_dir',
        type=str,
        default=None,
        help='The dir to store the pretrained models downloaded from huggingface.co.',
    )
    parser.add_argument(
        '--revision', type=str, default=None, help='The revision of the huggingface model.'
    )
    parser.add_argument('--dtype', type=str, default='bfloat16', help='The dtype of the model.')
    parser.add_argument('--temperature', type=float, default=0.0, help='Sampling temperature.')
    parser.add_argument(
        '--max_length',
        type=int,
        default=None,
        help='Max input length for the open source model. '
        'Use max_length in generation_config or max_position_embeddings in config '
        'if this arg is not provided',
    )
    parser.add_argument(
        '--max_tokens',
        type=int,
        default=128,
        help='Max new tokens to generate for generation based evalutaion.',
    )
    parser.add_argument(
        '--subsets',
        type=str,
        default='ALL',
        help="The subsets of TMLU (splited by comma). Default is 'ALL'.",
    )
    parser.add_argument(
        '--base_url', type=str, default=None, help='The base url for OpenAI python API library.'
    )
    parser.add_argument(
        '--tensor_parallel_size', type=int, default=1, help='Tensor parallel size for vllm.'
    )
    parser.add_argument(
        '--log_dir', type=str, default=None, help='Directory for saving evaluation log.'
    )
    parser.add_argument(
        '--overwrite_log_dir', action='store_true', help='Overwrite logs in the directory.'
    )
    parser.add_argument(
        '--few_shot_num',
        type=int,
        default=5,
        help='The number for few shot example. Range: [0, 5]. Default is 5.',
    )
    parser.add_argument(
        '--timeout', type=float, default=20.0, help='Timeout for API based backend.'
    )
    parser.add_argument(
        '--max_retries', type=int, default=100, help='Max retries for API based backend.'
    )
    parser.add_argument('--cot', action='store_true', help='Use CoT evaluation.')
    parser.add_argument(
        '--reduce_few_shot',
        action='store_true',
        help='Reduce few-shot example number when prompt exceed the model length limit.',
    )
    parser.add_argument(
        '--apply_chat_template', action='store_true', help='Apply chat template on the prompts.'
    )
    return parser.parse_args()


def format_problem(
    example: dict[str, str],
    model_template: Callable,
    topic_line: str = '以下選擇題為出自臺灣的考題，答案為其中一個選項。',
    few_shot_examples: list[dict[str, str]] = None,
    few_shot_num: int = 0,
    cot: bool = False,
    tokenizer=None,
    max_length=None,
    prefill=None,
    apply_chat_template=True,
) -> tuple[str, list[str]]:
    if tokenizer is None:
        prompt = topic_line + '\n\n'
        if few_shot_examples and few_shot_num:
            for i in range(few_shot_num):
                fs_ex = few_shot_examples[i]
                fs_ex_prompt = model_template(fs_ex, use_cot=cot, include_ans=True)
                prompt += fs_ex_prompt + '\n\n'
        example_prompt = model_template(example, use_cot=cot, include_ans=False)
        prompt += example_prompt
    else:
        prompt = topic_line + '\n\n'
        example_prompt = model_template(example, use_cot=cot, include_ans=False)
        if apply_chat_template:
            template = (
                tokenizer.apply_chat_template(
                    [{'role': 'user', 'content': ''}], tokenize=False, add_generation_prompt=True
                )
                + prefill
            )
        else:
            template = prefill
        left_length = max_length
        left_length -= len(tokenizer.encode(prompt, add_special_tokens=False))
        left_length -= len(tokenizer.encode(example_prompt, add_special_tokens=False))
        left_length -= len(tokenizer.encode(template, add_special_tokens=False))
        if few_shot_examples and few_shot_num:
            for i in range(few_shot_num):
                fs_ex = few_shot_examples[i]
                fs_ex_prompt = model_template(fs_ex, use_cot=cot, include_ans=True)
                fs_ex_prompt += '\n\n'
                fs_ex_prompt_length = len(tokenizer.encode(fs_ex_prompt, add_special_tokens=False))
                if left_length - fs_ex_prompt_length > 0:
                    prompt += fs_ex_prompt
                    left_length -= fs_ex_prompt_length

        prompt += example_prompt
    example['prompt'] = prompt
    return example


if __name__ == '__main__':
    args = parse_args()

    if args.backend == 'openai':
        api_key = os.environ.get('OPENAI_API_KEY')
        model = OpenAI_LM(
            model_name=args.model,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            api_key=api_key,
            base_url=args.base_url,
            timeout=args.timeout,
            max_retries=args.max_retries,
        )
    elif args.backend == 'anthropic':
        api_key = os.environ.get('ANTHROPIC_API_KEY')
        model = Anthropic_LM(
            model_name=args.model,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            api_key=api_key,
            timeout=args.timeout,
            max_retries=args.max_retries,
        )
    elif args.backend == 'google':
        api_key = os.environ.get('GOOGLE_API_KEY')
        model = Google_LM(
            model_name=args.model,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            api_key=api_key,
            timeout=args.timeout,
            max_retries=args.max_retries,
        )
    elif args.backend == 'custom_api':
        api_key = 'EMPTY'
        model = OpenAI_LM(
            model_name=args.model,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            api_key=api_key,
            base_url=args.base_url,
            timeout=args.timeout,
            max_retries=args.max_retries,
        )
    elif args.backend == 'hf':
        if args.cot:
            raise ValueError('CoT evaluation is not supported with HF backend now.')
        model = HFLM_transformers(
            model_name=args.model,
            max_tokens=args.max_tokens,
            max_length=args.max_length,
            temperature=args.temperature,
            revision=args.revision,
            dtype=args.dtype,
            cache_dir=args.cache_dir,
        )
    else:
        model = HFLM_vLLM(
            model_name=args.model,
            tensor_parallel_size=args.tensor_parallel_size,
            max_tokens=args.max_tokens,
            max_length=args.max_length,
            temperature=args.temperature,
            revision=args.revision,
            dtype=args.dtype,
            cache_dir=args.cache_dir,
        )

    if args.subsets == 'ALL':
        subsets = SUBSETS
    else:
        subsets = [x.strip() for x in args.subsets.split(',')]
        for subset in subsets:
            if subset not in SUBSETS:
                raise ValueError(f'{subset} is not an available subset of TMLU.')

    results = {}
    if args.log_dir:
        log_root = args.log_dir
    else:
        if args.backend == 'hf':
            log_root = os.path.join('log', f"{args.model.replace('/', '_')}_logits")
        elif args.cot:
            log_root = os.path.join('log', f"{args.model.replace('/', '_')}_cot")
        else:
            log_root = os.path.join('log', args.model.replace('/', '_'))
    os.makedirs(log_root, exist_ok=True)

    if args.cot:
        score_func = check_ans_cot
    else:
        score_func = check_ans

    for subset_name in subsets:
        logging.info(f'Evaluating {subset_name}')
        test_data = load_dataset(
            'miulab/tmlu',
            subset_name,
            split='test',
        )
        fs_data = load_dataset(
            'miulab/tmlu',
            subset_name,
            split='dev',
        )

        past_scores = []
        past_ids = set()
        if (
            os.path.isfile(os.path.join(log_root, f'{subset_name}_out.jsonl'))
            and not args.overwrite_log_dir
        ):
            with open(os.path.join(log_root, f'{subset_name}_out.jsonl')) as f:
                lines = f.readlines()
                for line in lines:
                    example = json.loads(line)
                    past_ids.add(example['id'])
                    score = 1 if score_func(example['full_response'], example['gold_answer']) else 0
                    past_scores.append(score)

            test_data = test_data.filter(lambda example: example['id'] not in past_ids)

        if args.backend == 'openai' or args.backend == 'custom_api':
            test_data = test_data.map(
                format_problem,
                load_from_cache_file=False,
                fn_kwargs={
                    'model_template': openai_template,
                    'few_shot_examples': fs_data,
                    'few_shot_num': args.few_shot_num,
                    'cot': args.cot,
                },
            )
            if args.cot:
                outputs = model.generate(test_data, prefill='')
            else:
                outputs = model.generate(test_data, prefill='')

        elif args.backend == 'anthropic':
            test_data = test_data.map(
                format_problem,
                load_from_cache_file=False,
                fn_kwargs={
                    'model_template': anthropic_template,
                    'few_shot_examples': fs_data,
                    'few_shot_num': args.few_shot_num,
                    'cot': args.cot,
                },
            )
            if args.cot:
                outputs = model.generate(test_data, prefill='\n讓我們一步一步思考。\n')
            else:
                outputs = model.generate(test_data, prefill='\n正確答案：(')

        elif args.backend == 'google':
            test_data = test_data.map(
                format_problem,
                load_from_cache_file=False,
                fn_kwargs={
                    'model_template': openai_template,
                    'few_shot_examples': fs_data,
                    'few_shot_num': args.few_shot_num,
                    'cot': args.cot,
                },
            )
            if args.cot:
                outputs = model.generate(test_data, prefill='')
            else:
                outputs = model.generate(test_data, prefill='')

        elif args.backend == 'hf':
            if args.reduce_few_shot:
                test_data = test_data.map(
                    format_problem,
                    load_from_cache_file=False,
                    fn_kwargs={
                        'model_template': hf_template,
                        'few_shot_examples': fs_data,
                        'few_shot_num': args.few_shot_num,
                        'cot': args.cot,
                        'tokenizer': model.tokenizer,
                        'max_length': model.model_max_length,
                        'prefill': '\n正確答案：(',
                        'apply_chat_template': args.apply_chat_template,
                    },
                )
            else:
                test_data = test_data.map(
                    format_problem,
                    load_from_cache_file=False,
                    fn_kwargs={
                        'model_template': hf_template,
                        'few_shot_examples': fs_data,
                        'few_shot_num': args.few_shot_num,
                        'cot': args.cot,
                    },
                )
            outputs = model.generate(
                test_data, prefill='\n正確答案：(', apply_chat_template=args.apply_chat_template
            )

        else:
            if args.cot:
                prefill = '\n讓我們一步一步思考。\n'
            else:
                prefill = '\n正確答案：('

            if args.reduce_few_shot:
                test_data = test_data.map(
                    format_problem,
                    load_from_cache_file=False,
                    fn_kwargs={
                        'model_template': hf_template,
                        'few_shot_examples': fs_data,
                        'few_shot_num': args.few_shot_num,
                        'cot': args.cot,
                        'tokenizer': model.tokenizer,
                        'max_length': model.model_max_length,
                        'prefill': prefill,
                        'apply_chat_template': args.apply_chat_template,
                    },
                )
            else:
                test_data = test_data.map(
                    format_problem,
                    load_from_cache_file=False,
                    fn_kwargs={
                        'model_template': hf_template,
                        'few_shot_examples': fs_data,
                        'few_shot_num': args.few_shot_num,
                        'cot': args.cot,
                    },
                )

            outputs = model.generate(
                test_data, prefill=prefill, apply_chat_template=args.apply_chat_template
            )

        if args.overwrite_log_dir:
            output_file_open_type = 'w'
        else:
            output_file_open_type = 'a'

        with open(os.path.join(log_root, f'{subset_name}_out.jsonl'), output_file_open_type) as f:
            for i in range(len(outputs)):
                line = {
                    'id': test_data[i]['id'],
                    'prompt': test_data[i]['prompt'],
                    'full_response': outputs[i],
                    'gold_answer': test_data[i]['answer'],
                }
                f.write(json.dumps(line, ensure_ascii=False) + '\n')

        if len(outputs) != len(test_data):
            raise ValueError(
                f'Error occurred when evaluating {subset_name}: output length mismatch'
            )
        scores = [
            1 if score_func(output, row['answer']) else 0 for output, row in zip(outputs, test_data)
        ]
        scores += past_scores
        avg_score = sum(scores) / len(scores)
        results[subset_name] = avg_score
    pprint(results)
