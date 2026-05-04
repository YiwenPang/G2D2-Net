# 此处是主代码，从这里开始即可。

import time

try:
    from data_preprocessing import preprocess_data
    from train_evaluate import main as train_and_evaluate
except ImportError:
    import importlib

    prep_module = importlib.import_module("data_preprocessing")
    train_module = importlib.import_module("train_evaluate")
    preprocess_data = prep_module.preprocess_data
    train_and_evaluate = train_module.main


def run_pipeline():
    start_time = time.time()

    # 第一步：数据预处理
    print("\n[步骤 1/2] 正在启动数据预处理...")
    print("-" * 30)
    try:
        preprocess_data()
        print("\n[OK] 数据预处理完成，特征文件已生成。")
    except Exception as e:
        print(f"\n[ERROR] 步骤 1 出错: {e}")
        return

    # 第二步：模型训练与评估
    print("\n" + "=" * 50)
    print("[步骤 2/2] 正在启动模型训练与评估...")
    print("-" * 30)
    try:
        train_and_evaluate()
        print("\n[OK] 模型训练与评估流程结束！")
    except Exception as e:
        print(f"\n[ERROR] 步骤 2 出错: {e}")
        return

    end_time = time.time()
    total_duration = end_time - start_time

    print("\n" + "=" * 50)
    print(f"全流程结束！总耗时: {total_duration:.2f} 秒")
    print("请查看 output/ 文件夹获取 AUC 曲线图和模型权重。")
    print("=" * 50)


if __name__ == "__main__":
    run_pipeline()
