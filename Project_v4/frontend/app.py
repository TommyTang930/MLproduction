import streamlit as st
import requests
import pandas as pd
import io
import json

st.title("端到端AutoML案例")
#endpoint = 'http://localhost:8080/predict'
endpoint = 'http://host.docker.internal:8080/predict'

test_csv = st.file_uploader(label="上传csv文件",
                            type=['csv'],
                            accept_multiple_files=False)

if test_csv:
    test_df = pd.read_csv(test_csv)
    st.subheader("上传数据集样本示例【前5行】")
    st.write(test_df.head())

    # 将 dataframe 转换为 BytesIO object
    test_bytes_obj = io.BytesIO()
    test_df.to_csv(test_bytes_obj, index=False)  # 写入到 BytesIO buffer
    test_bytes_obj.seek(0)

    files = {"file": ('test.csv', test_bytes_obj, "multipart/form-data")}

    # 点击预测按钮
    if st.button("开始预测"):
        if len(test_df)==0:
            st.write("请上传一个有效的测试集")
        else:
            with st.spinner("预测正在进行中，请等待..."):
                output = requests.post(endpoint,
                                       files=files,
                                       timeout=8000)
            st.success("成功！点击下方的 下载 按钮获取预测结果")
            st.download_button(
                label="下载",
                data=json.dumps(output.json()),
                file_name="AutoML_prediction_results.json"
            )