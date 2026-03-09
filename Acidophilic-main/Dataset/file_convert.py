import pandas as pd

def process_acido_split_pandas(csv_file):
    # 读取CSV文件
    df = pd.read_csv(csv_file)
    
    # 为每个组合创建计数器
    counters = {}
    
    # 处理每个组合的数据
    for (label, set_type), group in df.groupby(['label', 'set']):
        if label == 1 and set_type == 'train':
            filename = 'Positive.txt'
            prefix = 'Positive'
        elif label == 0 and set_type == 'train':
            filename = 'Negative.txt'
            prefix = 'Negative'
        elif label == 1 and set_type == 'test':
            filename = 'Ind-positive.txt'
            prefix = 'Ind-positive'
        elif label == 0 and set_type == 'test':
            filename = 'Ind-negative.txt'
            prefix = 'Ind-negative'
        else:
            continue
        
        # 写入文件
        with open(filename, 'w', encoding='utf-8') as f:
            for i, (_, row) in enumerate(group.iterrows(), 1):
                f.write(f'>{prefix}_{i}\n')
                f.write(f'{row["sequence"]}\n')

# 使用示例
if __name__ == "__main__":
    process_acido_split_pandas('./acidoSplit.csv')
    print("文件处理完成！已生成四个txt文件。")