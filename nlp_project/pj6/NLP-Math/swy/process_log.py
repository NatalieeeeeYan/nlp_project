import json

# 定义日志文件和JSONL文件路径
log_file_path = "/Users/songwenyan/ME/Fudan/NLP/NLP-Math/swy/log/test_qwen2505_prefix_temp.log"
jsonl_file_path = "dataset/MATH/test.jsonl"
output_file_path = "results/qw2505_prefix_temp.json"
output_records = []

def load_ground_truth(jsonl_file):
    """
    Load ground-truth data from a JSONL file.
    
    @param jsonl_file: Path to the JSONL file.
    @return: List of questions (ground truth).
    """
    questions = []
    ground_truth = {}
    with open(jsonl_file, "r", encoding="utf-8") as file:
        for line in file:
            record = json.loads(line)
            question = record["problem"].strip()
            questions.append(question)
            ground_truth[question] = record["solution"].strip()
    return questions, ground_truth

def process_log_segment(segment, question):
    """
    Process a segment of the log file to extract the question and Model's Answer.
    
    @param segment: A segment of the log file (str).
    @param question: The question for this segment (str).
    @return: A dictionary with question and prediction.
    """
    lines = segment.splitlines()
    current_answer = []
    in_answer = False

    for line in lines:
        line = line.strip()

        # Identify Model's Answer
        if "Model's Answer:" in line:
            in_answer = True
            answer_start = line.find("Model's Answer:") + len("Model's Answer:")
            if answer_start < len(line):
                current_answer.append(line[answer_start:].strip())
            continue

        # Add subsequent lines to the current answer
        if in_answer:
            current_answer.append(line)

    return {
        "question": question,
        "prediction": " ".join(current_answer).strip()
    }

def extract_log_data(log_file, questions):
    """
    Extract log data by splitting the log file based on ground-truth questions.
    
    @param log_file: Path to the log file (str).
    @param questions: List of questions from ground truth (list).
    @return: List of extracted data with question and prediction.
    """
    extracted_data = []
    with open(log_file, "r", encoding="utf-8") as file:
        log_content = file.read()

    # Split log content based on questions
    for i, question in enumerate(questions):
        # Find the start and end positions of the current question in the log
        start_idx = log_content.find(question)
        if start_idx == -1:
            continue  # Skip if question not found

        # Determine the end of this question's segment
        end_idx = log_content.find(questions[i + 1]) if i + 1 < len(questions) else len(log_content)
        segment = log_content[start_idx:end_idx]

        # Process the segment to extract data
        extracted_data.append(process_log_segment(segment, question))

    return extracted_data

# 加载ground_truth数据
questions, ground_truth_data = load_ground_truth(jsonl_file_path)

# 从日志文件中提取数据
log_data = extract_log_data(log_file_path, questions)

# 合并日志数据和ground_truth数据
for record in log_data:
    question = record["question"]
    prediction = record["prediction"]
    answer = ground_truth_data.get(question, None)
    if answer is not None:
        output_records.append({
            "question": question,
            "ground_truth": answer,
            "prediction": prediction
        })

# 保存结果到JSON文件
with open(output_file_path, "w", encoding="utf-8") as output_file:
    json.dump(output_records, output_file, indent=4, ensure_ascii=False)

print(f"Extracted {len(output_records)} records and saved to {output_file_path}.")
