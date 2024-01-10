# How to easily train your own Large Language Model (LLM)?

## Fine-tuning open-source LLMs like Llama2 or Mistral

### Model Selection

The first step involves choosing the right model architecture for your needs. Different large language models have different strengths and weaknesses based on the data they initially trained on. Also, factors to consider include the size of the model (number of parameters), its performance on tasks similar to yours, and the computational resources required for training and inference.

My recommendation for beginners is to use open-source LLMs like Llama2 or Mistral models. You can choose the parameter size according to your computational resources (Check out the table at the bottom).

### Dataset and Prompt Formats

This part is about preparing the data that will be used to train the model. It involves collecting a large and diverse dataset that's representative of the tasks the model will perform. The data needs to be cleaned and formatted correctly. Prompt formats refer to how you structure the inputs and outputs for the model, which is crucial for effective training and eventual usage of the model.

There are various prompt formats that can be used while training LLMs. I recommend checking online resources for which prompt format fits your base LLM. The library that we are using in the examples automatically adjusts the prompt formats but you can checkout these most known formats:

#### Alpaca:

```plaintext
{system_prompt}

### Instruction:
{query}

### Response:
{response}
```

#### Llama2:

```plaintext
<s><<SYS>>{system_prompt}<</SYS>> [INST] {query} [/INST] {response}<s/>
```

#### Mistral:

```plaintext
<s>{system} [INST] {query} [/INST] {response}<s/>
```

### SFT (Supervised Fine-Tuning)

Once you have a pre-trained model and a dataset, supervised fine-tuning (SFT) involves training the model further on your specific dataset. This helps the model learn the nuances of the domain or task it will be used for. It's called "supervised" because it typically involves training with labeled data, where the correct outputs are known.

SFT is integral to training models like LLAMA2 and Mistral, particularly for models with extensive parameters. Another common method is LoRA ( Low-Rank Adaptation of Large Language Models), a technique that allows only a small portion of the model to be trainable, thus reducing the number of learned parameters significantly. This allows for efficient training by modifying a fraction of the models, thus reducing the number of parameters and needed memory space significantly.

### Quantization

Your model's size and computational load can be further optimized using Quantized Low-Rank Adapters (QLoRA), which sit atop a 4-bit or 8-bit quantized, frozen model, preserving the base model's robustness. This fine-tuning process, which involves precise quantization, leads to a compact model without significantly affecting performance. The resulting model, which requires saving only the modifications, is compatible with various data types and retains the original model's integrity.

### Additional Methods

In enhancing the training process, techniques like Neptune noise and Flash Attention 2 can be incorporated to prevent overfitting and improve attention mechanism efficiency. By applying these methodologies to specific settings, your models can be fine-tuned to efficiently and accurately process your instructions, resulting in streamlined models.
Hardware Requirements (GPU VRAM)

| Method | Bits | 7B    | 13B   | 30B   | 65B    | 8x7B  |
| ------ | ---- | ----- | ----- | ----- | ------ | ----- |
| Full   | 16   | 160GB | 320GB | 600GB | 1200GB | 900GB |
| Freeze | 16   | 20GB  | 40GB  | 120GB | 240GB  | 200GB |
| LoRA   | 16   | 16GB  | 32GB  | 80GB  | 160GB  | 120GB |
| QLoRA  | 8    | 10GB  | 16GB  | 40GB  | 80GB   | 80GB  |
| QLoRA  | 4    | 6GB   | 12GB  | 24GB  | 48GB   | 32GB  |

---

### Example Training

I will show how you can easily start training your own LLaMA-2 7B/13B/70B and Mistral 7B/8x7B models with simple steps.

Let's consider using the ['LLaMA-Factory'](https://github.com/hiyouga/LLaMA-Factory) repository for our example training. However, it's important to note that there are several other frameworks available that share similarities in their functionalities and setup processes. The reason I prefer 'LLaMA-Factory' is due to its user-friendly nature, characterized by its simplicity in setup and ease of use.

#### Setup

Basically, you can just download the codes from the GitHub repo to setup:

```bash
git clone https://github.com/hiyouga/LLaMA-Factory.git
cd LLaMA-Factory
pip install -r requirements.txt
```

- If you are getting errors or having issues, please refer to the [GitHub repo](https://github.com/hiyouga/LLaMA-Factory).

If you have multiple GPUs and want to use all of them for the training, choose 'multi-gpu' with this command (you can use default/NO options for other questions):

```bash
# configure the environment:
accelerate config

# use this command while multi-gpu training:
accelerate launch src/train_bash.py ...

# use this command while single-gpu training:
CUDA_VISIBLE_DEVICES=0 python src/train_bash.py ...
```

#### SFT (Supervised Fine-Tuning)

I recommend fine-tuning the parameters according to your dataset and use case:

```bash
accelerate launch src/train_bash.py \
    --stage sft \
    --model_name_or_path meta-llama/Llama-2-13b-chat-hf \
    --do_train \
    --dataset alpaca_gpt4_en \
    --template llama2 \
    --finetuning_type lora \
    --quantization_bit 4 \
    --lora_target q_proj,v_proj \
    --output_dir trained_model_sft \
    --overwrite_cache \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 4 \
    --lr_scheduler_type cosine \
    --logging_steps 10 \
    --save_steps 100000 \
    --learning_rate 2e-5 \
    --num_train_epochs 2.0 \
    --neftune_noise_alpha 5 \
    --flash_attn \
    --bf16
```

```bash
accelerate launch src/train_bash.py \
    --stage sft \
    --model_name_or_path mistralai/Mixtral-8x7B-Instruct-v0.1 \
    --do_train \
    --dataset alpaca_gpt4_en \
    --template mistral \
    --finetuning_type lora \
    --quantization_bit 4 \
    --lora_target q_proj,v_proj \
    --output_dir trained_model_sft \
    --overwrite_cache \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 4 \
    --lr_scheduler_type cosine \
    --logging_steps 10 \
    --save_steps 100000 \
    --learning_rate 2e-5 \
    --num_train_epochs 2.0 \
    --neftune_noise_alpha 5 \
    --flash_attn \
    --bf16
```

#### Datasets

If you want to use your custom datasets, please update 'LLaMA-Factory/data/dataset_info.json'

For more information: https://github.com/hiyouga/LLaMA-Factory/blob/main/data/README.md

#### Prediction

```bash

CUDA_VISIBLE_DEVICES=0 python src/train_bash.py \
    --stage sft \
    --model_name_or_path meta-llama/Llama-2-13b-chat-hf \
    --do_predict \
    --dataset alpaca_gpt4_en \
    --template llama2 \
    --finetuning_type lora \
    --quantization_bit 4 \
    --checkpoint_dir trained_model_sft \
    --output_dir trained_model_output \
    --per_device_eval_batch_size 1 \
    --max_samples 10 \
    --predict_with_generate \
    --bf16
```

```bash

CUDA_VISIBLE_DEVICES=0 python src/train_bash.py \
    --stage sft \
    --model_name_or_path mistralai/Mixtral-8x7B-Instruct-v0.1 \
    --do_predict \
    --dataset alpaca_gpt4_en \
    --template mistral \
    --finetuning_type lora \
    --quantization_bit 4 \
    --checkpoint_dir trained_model_sft \
    --output_dir trained_model_output \
    --per_device_eval_batch_size 1 \
    --max_samples 10 \
    --predict_with_generate \
    --bf16
```

#### Merge and Export the Final Model

```bash
python src/export_model.py \
    --model_name_or_path meta-llama/Llama-2-13b-chat-hf \
    --template llama2 \
    --finetuning_type lora \
    --quantization_bit 4 \
    --checkpoint_dir trained_model_sft \
    --export_dir trained_model_output
```

```bash

python src/export_model.py \
    --model_name_or_path mistralai/Mixtral-8x7B-Instruct-v0.1 \
    --template mistral \
    --finetuning_type lora \
    --quantization_bit 4 \
    --checkpoint_dir trained_model_sft \
    --export_dir trained_model_output
```

### Final Words

The examples I provided can help you to easily start fine-tuning your LLM. However, if you optimize your training further, you can write your own python code, checkout: https://huggingface.co/docs/autotrain/llm_finetuning
