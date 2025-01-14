import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import json
import argparse


# 定义生成数学问题的提示
def generate_math_prompt(question: str) -> str:
    '''
    @param: question The math question to be solved
    
    @return: prompt The formatted prompt for the model
    '''
    prompt = (
        f"Please solve the following math problem step by step and provide the final answer. "
        "Do not add any extra output. The final answer should be clearly marked.\n\n"
        f"Question: {question}\n\n"
    )
    return prompt


# 推理函数
def solve_math_with_llm(data: list, temperature: float) -> list:
    '''
    @param: data A list of instances, each containing a math question.
    @param: temperature The temperature parameter for controlling randomness.
    
    @return: predictions A list of instances with the math question and predicted result.
    '''
    predictions = []
    for d in data:
        # question = d['question']        # gsm8k
        question = d['problem']     # MATH
        prompt = generate_math_prompt(question)
        
        # 手动格式化输入为消息
        messages = [
            {"role": "system", "content": "You are a math assistant who solves problems step by step."},
            {"role": "user", "content": prompt}
        ]
        
        # 格式化输入文本
        formatted_text = "\n".join([f"{msg['role']}: {msg['content']}" for msg in messages])
        
        # 对输入文本进行 tokenization
        model_inputs = tokenizer([formatted_text], return_tensors="pt").to(device)
        
        # 生成模型输出
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=512,
            temperature=temperature  # 设置生成温度
        )
        
        # 解码输出
        output = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        # 提取最终答案
        final_answer = output.replace(prompt, '').replace("You are a math assistant who solves problems step by step.", '')
        
        print("Question:", question)
        print("Model's Answer:", final_answer)
        
        # 保存结果
        predictions.append({
            'question': question,
            'prediction': final_answer, 
            'ground_truth': d['answer']
        })
    return predictions


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run math problem solving with LLM")
    # parser.add_argument('--model', type=str, required=True, help='Path to the model') 
    parser.add_argument('--result', type=str, required=True, help='Path to save the results') 
    parser.add_argument('--temperature', type=float, default=1.0, help='Temperature for generation (default: 1.0)') 
    args = parser.parse_args()

    # 加载测试数据集
    data = []
    with open("./dataset/MATH/test.jsonl", 'r') as f:
        for line in f:
            data.append(json.loads(line))
    print("Dataset Loaded!")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 加载微调后的模型和分词器
    model_path = './models/qwen2505_sft_finetuned_MATH'
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype="auto", device_map="auto", low_cpu_mem_usage=True)
    
    model.to(device)
    print("Model loaded successfully!")

    # 获取模型预测结果
    predicted_results = solve_math_with_llm(data, args.temperature)

    # 保存预测结果
    with open(args.result, 'w', encoding='utf-8') as f:
        json.dump(predicted_results, f, ensure_ascii=False, indent=4)

    print("Results saved successfully!")
