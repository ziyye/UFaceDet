import json

# 读取第一个 JSON 文件并解析为字典
with open('/home/tianyuan/workspace/sqyolov8/data_pre/label_20240402_2.json', 'r') as file:
    data_checked = json.load(file)

# 读取第二个 JSON 文件并解析为字典
with open('/home/tianyuan/workspace/sqyolov8/data_pre/label_20240402.json', 'r') as file:
    data_row = json.load(file)

# 合并两个字典
merged_data = []
img_paths = []
# for data2 in data_row:
#     img_paths.append(data2['img_path'])

for data1 in data_checked:
    img_path1 = data1["img_path"]
    data3 = {}
    for data2 in data_row:
        img_path2 = data2["img_path"]
        # 当找到元素
        if img_path1 == img_path2:
            data3['img_path'] = img_path1
            data3['chcl'] = data1['chcl']
            merged_data.append(data3)

save_json_path = "/home/tianyuan/workspace/sqyolov8/data_pre/merged_20240402.json"
with open(save_json_path, 'w', encoding='utf-8') as fw:
    json.dump(merged_data, fw, ensure_ascii=False, indent=4)