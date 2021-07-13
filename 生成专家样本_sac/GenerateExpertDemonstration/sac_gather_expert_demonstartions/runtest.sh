#! /bin/bash
for file in `ls models/Hopper-v2` #注意此处这是两个反引号，表示运行系统命令
do
  #echo $file #在此处处理文件即可
  nohup python -u Test.py --file_name $file > test_dir/$file.log 2>&1 &
done
