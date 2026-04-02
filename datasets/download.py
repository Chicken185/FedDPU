import os
import ssl
import torchvision

# ================= 配置路径 =================
# 你指定的保存路径
SAVE_DIR = "/home/hdf/chicken/FL_PU/datasets"

# ================= 解决 SSL 证书报错 =================
# 在某些服务器或校园网环境下，Python 下载可能会报 SSL 错误
# 这行代码可以跳过证书验证，保证下载顺畅
ssl._create_default_https_context = ssl._create_unverified_context

def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"创建目录: {path}")
    else:
        print(f"目录已存在: {path}")

def download_datasets():
    create_dir(SAVE_DIR)
    
    print("="*30)
    print(f"开始下载数据到: {SAVE_DIR}")
    print("="*30)


    # 2. 下载 USPS
    print("\n[2/2] 正在下载 USPS...")
    try:
        torchvision.datasets.Imagenette(root=SAVE_DIR, train=True, download=True)
        torchvision.datasets.Imagenette(root=SAVE_DIR, train=False, download=True)
        print("✅ USPS 下载完成")
    except Exception as e:
        print(f"❌ USPS 下载失败: {e}")

if __name__ == "__main__":
    download_datasets()