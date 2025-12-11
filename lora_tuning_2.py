import os
import json
import re
from typing import List, Dict

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
)
from datasets import load_dataset
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
)
from huggingface_hub import login, HfApi
# from dotenv import load_dotenv

# ===== 0. 환경 설정 및 로그인 =====
load_dotenv()
login(token="")

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("Device:", device)

# ===== 1. 모델 / 토크나이저 로드 (Qwen2.5-7B) =====
model_name = "Qwen/Qwen2.5-7B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    trust_remote_code=True,
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    trust_remote_code=True,
    load_in_4bit=True,       # QLoRA용 4bit 로딩
    torch_dtype="auto",
    device_map="auto",
)

# ===== 2. LoRA 설정 + QLoRA 준비 =====
lora_config = LoraConfig(
    r=32,
    lora_alpha=64,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

# 4bit 학습 준비
model.config.use_cache = False
model = prepare_model_for_kbit_training(model)

# LoRA 적용
model = get_peft_model(model, lora_config)

# gradient checkpointing
model.gradient_checkpointing_enable()
model.enable_input_require_grads()

model.print_trainable_parameters()

# ===== 3. 토크나이징 함수 =====
# 회의 전체 컨텍스트를 조금 더 살리기 위해 MAX_LEN 확대
MAX_LEN = 512      # 필요하면 1024까지 확장 가능
ANSWER_MAX = 256   # JSON 답변 최대 길이

def tokenize_function(example: Dict) -> Dict:
    messages = example["messages"]
    assert messages[-1]["role"] == "assistant"

    # system + user 까지만 프롬프트로 사용
    prompt_text = tokenizer.apply_chat_template(
        messages[:-1],
        tokenize=False,
        add_generation_prompt=True,
    )

    # 마지막 assistant 메시지를 정답으로 사용
    answer_text = messages[-1]["content"] + tokenizer.eos_token

    prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    answer_ids = tokenizer(answer_text, add_special_tokens=False)["input_ids"]

    # 1) 프롬프트 길이 제한
    max_prompt_len = MAX_LEN - ANSWER_MAX
    if len(prompt_ids) > max_prompt_len:
        # 프롬프트가 너무 길면 뒤쪽만 남김 (회의 후반부/문맥 유지용)
        prompt_ids = prompt_ids[-max_prompt_len:]

    # 2) 답변 길이 제한
    max_answer_len = MAX_LEN - len(prompt_ids)
    if max_answer_len <= 0:
        answer_ids = []
    else:
        answer_ids = answer_ids[:max_answer_len]

    input_ids = prompt_ids + answer_ids
    attention_mask = [1] * len(input_ids)

    # 3) 패딩
    pad_len = MAX_LEN - len(input_ids)
    if pad_len > 0:
        input_ids += [tokenizer.pad_token_id] * pad_len
        attention_mask += [0] * pad_len

    # 프롬프트 부분은 -100으로 마스킹, 답변 부분만 loss 계산
    labels = [-100] * len(prompt_ids) + answer_ids + [-100] * pad_len

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }

# ===== 4. 데이터셋 로드 =====
# syn_data.jsonl : 라인마다 {"messages": [...]} 구조라고 가정
dataset = load_dataset(
    "json",
    data_files={"train": ["syn_data.jsonl"]},
)

print("원본 예시 샘플:")
print(dataset["train"][0])

# train / eval 분리 (약 2500개 기준 → 2250 / 250 정도)
dataset = dataset["train"].train_test_split(
    test_size=0.1,
    shuffle=True,
    seed=42,
)

train_ds = dataset["train"]
eval_ds = dataset["test"]

# ===== 5. 토크나이즈된 데이터 생성 =====
tokenized_train = train_ds.map(
    tokenize_function,
    batched=False,
    remove_columns=train_ds.column_names,
)

tokenized_eval = eval_ds.map(
    tokenize_function,
    batched=False,
    remove_columns=eval_ds.column_names,
)

# 샘플 체크
sample = tokenized_train[0]
print("input_ids:", sample["input_ids"][:80])
print("labels   :", sample["labels"][:80])

valid_labels = sum(1 for x in sample["labels"] if x != -100)
print("유효 라벨 토큰 개수:", valid_labels)

from transformers import default_data_collator

# ===== 6. 학습 설정 (TrainingArguments) =====
training_args = TrainingArguments(
    output_dir="./results_qwen2.5_7b_meeting",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    num_train_epochs=5,               # 3~5 범위에서 조절 가능

    bf16=True,                        # A40 등 bf16 지원 GPU면 True, 아니면 False로
    fp16=False,

    learning_rate=1e-4,               # 7B + 2.5k 데이터에 조금 더 안정적인 값
    lr_scheduler_type="cosine",
    warmup_ratio=0.05,

    logging_dir="./logs_qwen2.5_7b_meeting",
    logging_steps=30,
    logging_strategy="steps",
    logging_first_step=True,

    evaluation_strategy="epoch",      # 매 epoch마다 eval
    save_strategy="epoch",
    save_total_limit=3,
    load_best_model_at_end=True,      # 가장 좋은 eval loss 모델 사용
    metric_for_best_model="loss",

    optim="adamw_torch",
    ddp_find_unused_parameters=False,

    remove_unused_columns=False,      # labels 컬럼 유지를 위해 False
    push_to_hub=False,
    report_to=["none"],               # 원하면 ["tensorboard"] 등으로 변경 가능
)

# ===== 7. Trainer 정의 및 학습 =====
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_eval,
    data_collator=default_data_collator,
    tokenizer=tokenizer,
)

trainer.train()

# ===== 8. 모델 / 토크나이저 로컬 저장 =====
save_dir = "./Qwen2.5_7B_meeting_json_sft_v1"
trainer.save_model(save_dir)
tokenizer.save_pretrained(save_dir)

print(f"모델이 {save_dir} 에 저장되었습니다.")

# ===== 9. Hugging Face Hub 업로드 (원하면 사용) =====
api = HfApi()
repo_id = "CHOROROK/Qwen2.5_7B_meeting_json_sft_v1"

api.create_repo(
    repo_id=repo_id,
    repo_type="model",
    private=False,
    exist_ok=True,
)

api.upload_folder(
    folder_path=save_dir,
    repo_id=repo_id,
    repo_type="model",
    commit_message="Upload Qwen2.5-7B meeting JSON SFT LoRA adapter v1",
)
print(f"HuggingFace Hub에 {repo_id} 로 업로드 완료")
