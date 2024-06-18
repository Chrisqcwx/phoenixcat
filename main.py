# 重新读取原始文件内容
with open('raw.txt', 'r') as file:
    # 读取全部内容
    data = file.read()

# 处理原始数据，将所有项目分割并独立成行
items = data.split('|')
formatted_content = '\n'.join(item.strip() for item in items)

# 将格式化后的内容保存到新文件中
with open('output.txt', 'w') as new_file:
    new_file.write(formatted_content)