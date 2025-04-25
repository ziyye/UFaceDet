import os

def convert_labels(input_dir):
    """
    将指定目录下所有txt文件中的标签0转换为1
    :param input_dir: 包含标签文件的目录路径
    """
    # 遍历目录下的所有文件
    for filename in os.listdir(input_dir):
        if filename.endswith('.txt'):
            file_path = os.path.join(input_dir, filename)
            
            # 读取文件内容
            with open(file_path, 'r') as f:
                lines = f.readlines()
            
            # 修改标签
            new_lines = []
            for line in lines:
                parts = line.strip().split()
                if parts[0] == '0':  # 如果标签是0
                    parts[0] = '1'   # 改为1
                new_lines.append(' '.join(parts) + '\n')
            
            # 写回文件
            with open(file_path, 'w') as f:
                f.writelines(new_lines)

if __name__ == '__main__':
    # 指定包含标签文件的目录
    label_dir = '/data01/data/qrcode/synthetise/train/labels'  # 请替换为实际的标签文件目录
    convert_labels(label_dir)
