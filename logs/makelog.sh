#!/bin/bash

# 检查是否提供了脚本路径参数
if [ -z "$1" ]; then
  echo "Usage: $0 <script_path>"
  exit 1
fi

# 获取脚本路径
script_path=$1

# 设置默认的日志文件夹
log_dir="/root/autodl-tmp/ESM/logs"

# 如果日志文件夹不存在，则创建它
mkdir -p "$log_dir"

# 获取当前日期时间
current_time=$(date +"%Y%m%d_%H%M%S")

# 获取脚本文件名（不带扩展名）
script_name=$(basename "$script_path" .py)

# 生成日志文件名
log_file="${log_dir}/${script_name}_${current_time}.log"

# 使用 nohup 运行 Python 脚本并将输出重定向到日志文件
nohup python "$script_path" > "$log_file" 2>&1 &

echo "Script is running in the background. Log file: $log_file"