import os
import torch
import torch.distributed as dist

world_size = int(os.environ.get("WORLD_SIZE", 1))
rank = int(os.environ.get("RANK", 0))
local_rank = int(os.environ.get("LOCAL_RANK", 0))

is_ddp = world_size > 1
print(f"is_ddp={is_ddp}, world_size={world_size}, rank={rank}, local_rank={local_rank}")

if torch.cuda.is_available():
    torch.cuda.set_device(local_rank)
    print("using", torch.cuda.current_device(), torch.cuda.get_device_name(torch.cuda.current_device()))

if is_ddp:
    dist.init_process_group(backend="nccl", device_id=local_rank)
    dist.barrier()
    dist.destroy_process_group()