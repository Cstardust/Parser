#### 数据构造步骤

```cmd
Step1: 分别运行databyllm_by_hanlp_test.ipynb，databyllm_by_hanlp_dev.ipynb使用hanlp构造数据
Step2: 分别运行databyllm_dev.ipynb，databyllm_test.ipynb使用gpt3.5构造数据
Step3: 运行format_gpt_result格式化GPT的结果
Step4: 运行merge_data将gpt句间的标签填充到hanlp的结果中
Step5: 运行split_train_test.ipynb分割train, test data[暂时不用]
Step6: 运行评估集构造构造真实的test data
```