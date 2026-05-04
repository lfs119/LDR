import torch

def check_cuda_environment():
    # 检查 CUDA 可用性
    cuda_available = torch.cuda.is_available()
    print("=" * 50)
    print(f"PyTorch CUDA 环境检测结果:")
    print(f"CUDA 可用性: {'✅ 可用' if cuda_available else '❌ 不可用'}")
    
    if cuda_available:
        # 获取 GPU 基础信息
        device_count = torch.cuda.device_count()
        print(f"\nGPU 总数: {device_count} 张")
        
        # 遍历所有 GPU 设备
        for i in range(device_count):
            device = torch.device(f"cuda:{i}")
            print(f"\nGPU {i}:")
            print(f"  ▶ 型号: {torch.cuda.get_device_name(i)}")
            print(f"  ▶ 计算能力: {torch.cuda.get_device_capability(i)}")
            
            # 获取显存信息（单位：GB）
            props = torch.cuda.get_device_properties(i)
            total_mem = props.total_memory / 1024**3
            allocated_mem = torch.cuda.memory_allocated(i) / 1024**3
            cached_mem = torch.cuda.memory_reserved(i) / 1024**3
            print(f"  ▶ 显存状态:")
            print(f"     总容量: {total_mem:.2f} GB")
            print(f"     已分配: {allocated_mem:.2f} GB")
            print(f"     缓存区: {cached_mem:.2f} GB")
            
            # 简单压力测试
            try:
                x = torch.randn(10000, 10000, device=device)
                y = x @ x
                del x, y
                print("  ▶ 基础运算测试: ✅ 通过")
            except Exception as e:
                print(f"  ▶ 基础运算测试: ❌ 失败 ({str(e)})")
    else:
        print("\n错误排查建议:")
        print("1. 检查 NVIDIA 驱动是否安装: `nvidia-smi`")
        print("2. 确认 CUDA 版本与 PyTorch 匹配")
        print("3. 查看 PyTorch 安装命令是否包含 CUDA 支持")

if __name__ == "__main__":
    check_cuda_environment()