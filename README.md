# TMLU Eval

<p align="center">
🤗 <a href="https://huggingface.co/datasets/miulab/tmlu" target="_blank">Dataset</a>
• 📃 <a href="https://arxiv.org/pdf/2403.20180" target="_blank">Paper</a>
</p>

For open source models, we support

* Hugging-Face as backend
  * Use the probabilities of the option codes as prediction
* vLLM as backend
  * Parse the model generation to extract the prediction

For proprietary models, we support

* OpenAI API
  * Including custom API server which support **[openai-python](https://github.com/openai/openai-python)**
* Anthropic API
* Google API


## Environment Setup

To set up the environment for TMLU Eval, follow these steps:

1. Create a new conda environment:
   ```bash
   conda create -n tmlu python=3.10 -y
   ```

2. Activate the environment:
   ```bash
   conda activate tmlu
   ```

3. Install PyTorch with CUDA support:
   ```bash
   conda install pytorch pytorch-cuda=12.1 -c pytorch -c nvidia -y
   ```

4. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```



## Hugging Face

Use the probabilities of the option codes as prediction.

### Command:

```bash
$python3 tmlu_eval.py \
	--backend hf \
	--model [Model name on huggingface or path] \
	--dtype [torch type for the model, default will follow the model config] \
	--subsets [Choose subsets of TMLU (names splited by comma) for evalutaion. Default is 'ALL'] \
	--log_dir [Directory for saving evaluation log] \
	--few_shot_num [The number for few shot example. Range: [0, 5]. Default is 5.]
```

### Example:

```bash
$python3 tmlu_eval.py \
	--backend hf \
	--model yentinglin/Taiwan-LLM-7B-v2.1-chat \
	--dtype float16 \
	--temperature 0.0 \
	--subsets AST_chinese,AST_mathematics \
	--log_dir log/prob_based/Taiwan-LLM-7B-v2.1-chat \
	--few_shot_num 5
```

## vLLM

Parse the model generation to extract the prediction.

### Command:

```bash
$python3 tmlu_eval.py \
	--backend vllm \
	--model [Model name on huggingface or path] \
	--dtype [torch type for the model, default will follow the model config] \
	--temperature [temperature for generation] \
	--max_tokens [max new tokens to generate] \
	--subsets [Choose subsets of TMLU (names splited by comma) for evalutaion. Default is 'ALL']  \
	--tensor_parallel_size [Tensor parallel size for vLLM] \
	--log_dir [Directory for saving evaluation log] \
	--few_shot_num [The number for few shot example. Range: [0, 5]. Default is 5.] \
	--cot [Use CoT evaluation.]
```

### Example:

```bash
$python3 tmlu_eval.py \
	--backend vllm \
	--model yentinglin/Taiwan-LLM-7B-v2.1-chat \
	--dtype float16 \
	--temperature 0.0 \
	--max_tokens 128 \
	--subsets AST_chinese,AST_mathematics \
	--tensor_parallel_size 1 \
	--log_dir log/gen_based/Taiwan-LLM-7B-v2.1-chat \
	--few_shot_num 5 \
	--cot
```

## Custom API model (use **[openai-python](https://github.com/openai/openai-python)** for querying)

Parse the model generation to extract the prediction.

### Command:

```bash
$python3 tmlu_eval.py \
	--backend custom_api \
	--model [Model name] \
	--base_url [base url for the API]
	--temperature [temperature for generation] \
	--max_tokens [max new tokens to generate] \
	--subsets [Choose subsets of TMLU (names splited by comma) for evalutaion. Default is 'ALL'] \
	--log_dir [Directory for saving evaluation log] \
	--few_shot_num [The number for few shot example. Range: [0, 5]. Default is 5.] \
	--cot [Use CoT evaluation.]
```

### Example:

```bash
$python3 tmlu_eval.py \
	--backend custom_api \
	--model yentinglin/Taiwan-LLM-MoE-alpha \
	--base_url http://127.0.0.1:8888/v1
	--temperature 0.0 \
	--max_tokens 128 \
	--subsets AST_chinese,AST_mathematics \
	--log_dir log/gen_based/Taiwan-LLM-MoE-alpha
```

## OpenAI

Parse the model generation to extract the prediction.

#### before start

Set environment variable `OPENAI_API_KEY`.

### Command:

```bash
$python3 tmlu_eval.py \
	--backend openai \
	--model [Model name] \
	--temperature [temperature for generation] \
	--max_tokens [max new tokens to generate] \
	--subsets [Choose subsets of TMLU (names splited by comma) for evalutaion. Default is 'ALL'] \
	--log_dir [Directory for saving evaluation log] \
	--few_shot_num [The number for few shot example. Range: [0, 5]. Default is 5.] \
	--cot [Use CoT evaluation.]
```

### Example:

```bash
$python3 tmlu_eval.py \
	--backend openai \
	--model gpt-4-1106-preview \
	--temperature 0.0 \
	--max_tokens 128 \
	--subsets AST_chinese,AST_mathematics \
	--log_dir log/gen_based/gpt-4-1106-preview
```

## Anthropic

Parse the model generation to extract the prediction.

#### before start

Set environment variable`ANTHROPIC_API_KEY`.

### Command:

```bash
$python3 tmlu_eval.py \
	--backend anthropic \
	--model [Model name] \
	--temperature [temperature for generation] \
	--max_tokens [max new tokens to generate] \
	--subsets [Choose subsets of TMLU (names splited by comma) for evalutaion. Default is 'ALL'] \
	--log_dir [Directory for saving evaluation log] \
	--few_shot_num [The number for few shot example. Range: [0, 5]. Default is 5.] \
	--cot [Use CoT evaluation.]
```

### Example:

```bash
$python3 tmlu_eval.py \
	--backend anthropic \
	--model claude-2.0 \
	--temperature 0.0 \
	--max_tokens 128 \
	--subsets AST_chinese,AST_mathematics \
	--log_dir log/gen_based/claude-2.0
```

## Gemini-pro

Parse the model generation to extract the prediction.

#### before start

Set environment variable`GOOGLE_API_KEY`.

### Command:

```bash
$python3 tmlu_eval.py \
	--backend google \
	--model [Model name] \
	--temperature [temperature for generation] \
	--max_tokens [max new tokens to generate] \
	--subsets [Choose subsets of TMLU (names splited by comma) for evalutaion. Default is 'ALL'] \
	--log_dir [Directory for saving evaluation log] \
	--few_shot_num [The number for few shot example. Range: [0, 5]. Default is 5.] \
	--cot [Use CoT evaluation.]
```

### Example:

```bash
$python3 tmlu_eval.py \
	--backend google \
	--model gemini-pro \
	--temperature 0.0 \
	--max_tokens 128 \
	--subsets AST_chinese,AST_mathematics \
	--log_dir log/gen_based/gemini-pro
```


# Citation

If you use TMLU in your research, please cite the following paper:

```
@article{DBLP:journals/corr/abs-2403-20180,
  author       = {Po{-}Heng Chen and
                  Sijia Cheng and
                  Wei{-}Lin Chen and
                  Yen{-}Ting Lin and
                  Yun{-}Nung Chen},
  title        = {Measuring Taiwanese Mandarin Language Understanding},
  journal      = {CoRR},
  volume       = {abs/2403.20180},
  year         = {2024},
  url          = {https://doi.org/10.48550/arXiv.2403.20180},
  doi          = {10.48550/ARXIV.2403.20180},
  eprinttype    = {arXiv},
  eprint       = {2403.20180},
  timestamp    = {Wed, 10 Apr 2024 17:37:45 +0200},
  biburl       = {https://dblp.org/rec/journals/corr/abs-2403-20180.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
```
