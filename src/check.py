import torch

def check_cuda():
    if torch.cuda.is_available():
        print(f"GPU disponível: {torch.cuda.get_device_name(0)}")
        print(f"Quantidade de GPUs: {torch.cuda.device_count()}")
        print(f"Memória Total da GPU (GB): {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f}")
    else:
        print("Nenhuma GPU detectada. Usando CPU.")

if __name__ == "__main__":
    check_cuda()