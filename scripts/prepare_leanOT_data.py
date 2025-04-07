import os
from datasets import load_dataset, Dataset


promt = f"""You are a helpful assistant who will solve every problem
,→ **WITHOUT** Long Chain-of-Thought
### Instruction:
@ Natural language theorem statement:
{NL_statement}
@ Lean4 theorem statement:
```lean4
{Lean_statement}
```&
@ Lean4 theorem statement and proof with explanatory comments preceding each
,→ line of code:"""
output = f"""
### Response:
=== Outpus Example ===
<Thought>
The user ask not to solve with long CoT, so I will directly write the answer.
</Thought>
<Output>
```lean4
{Lean_output}
```&
</Output>"""

def process_lean_proof(proof_text):
    """处理Lean证明文本,提取到:= by前的部分作为prompt"""
    lines = proof_text.split("\n")
    prompt_lines = []
    answer_lines = []
    is_answer = False

    for line in lines:
        if ":= by" in line:
            prompt_lines.append(line.split(":= by")[0])
            is_answer = True

        elif is_answer:
            answer_lines.append(line)
        else:
            prompt_lines.append(line)
    # 将prompt_lines和answer_lines转换为字符串
    prompt_lines = [line.strip() for line in prompt_lines if line.strip()]
    answer_lines = [line.strip() for line in answer_lines if line.strip()]

    prompt = f"""You are a helpful assistant who will solve every problem
    ,→ **WITHOUT** Long Chain-of-Thought
    ### Instruction:
    @ Lean4 theorem statement:
    ```lean4
    {prompt_lines}
    ```&
    @ Lean4 theorem statement and proof with explanatory comments preceding each
    ,→ line of code:"""
    output = f"""
    ### Response:
    === Outpus Example ===
    <Thought>
    The user ask not to solve with long CoT, so I will directly write the answer.
    </Thought>
    <Output>
    ```lean4
    {answer_lines}
    ```&
    </Output>"""

    return (prompt, output)

def prepare_lean_dataset():
    """准备Lean数据集"""
    # 直接加载parquet文件
    dataset = load_dataset(
        'parquet', 
        data_files='/app/qi/backup/data/RPROVER/Lean-workbook-proofs/data/train-00000-of-00001.parquet'
    )['train']
    

    # 处理proof提取prompt
    processed_data = {
        "context_messages": [],
        "formal_statement": []
    }

    for item in dataset:
        prompt,answer = process_lean_proof(item["full_proof"])
        processed_data["context_messages"].append(answer)
        processed_data["formal_statement"].append(prompt)

    # 创建新数据集
    processed_dataset = Dataset.from_dict(processed_data)
    return processed_dataset

def main():
    output_dir = "/app/qi/backup/data/RPROVER/lean_proofs_data"
    
    # 处理数据
    dataset = prepare_lean_dataset()
    
    # 保存数据集
    os.makedirs(output_dir, exist_ok=True)
    dataset.save_to_disk(output_dir)

if __name__ == "__main__":
    main()
