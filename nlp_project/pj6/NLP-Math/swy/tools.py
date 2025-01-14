import json

def merge_jsonl_files(input_file1, input_file2, output_file):
    """
    合并两个 JSONL 文件到一个新的 JSONL 文件中。
    
    :param input_file1: 第一个输入文件路径
    :param input_file2: 第二个输入文件路径
    :param output_file: 合并后的输出文件路径
    """
    # 打开文件并读取所有行
    data = []
    with open(input_file1, 'r', encoding='utf-8') as f1:
        for line in f1:
            data.append(json.loads(line.strip()))  # 将每行解析为 JSON 对象

    with open(input_file2, 'r', encoding='utf-8') as f2:
        for line in f2:
            data.append(json.loads(line.strip()))  # 将每行解析为 JSON 对象

    # 写入到新的 JSONL 文件
    with open(output_file, 'w', encoding='utf-8') as fout:
        for entry in data:
            fout.write(json.dumps(entry, ensure_ascii=False) + '\n')  # 写回每个 JSON 对象为一行

    print(f"合并完成，输出文件为：{output_file}")


# 使用示例
if __name__ == "__main__":
    input_file1 = "./dataset/MATH/AugMATH_part1.jsonl"
    input_file2 = "./dataset/MATH/AugMATH_part2.jsonl"
    output_file = "./dataset/MATH/AugMATH_merged.jsonl"

    merge_jsonl_files(input_file1, input_file2, output_file)
