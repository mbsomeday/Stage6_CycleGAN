
import os


is_ddp = "WORLD_SIZE" in os.environ and int(os.environ["WORLD_SIZE"]) > 1
if is_ddp:
    print('current is in ddp mode', os.environ["WORLD_SIZE"])
else:
    print('no ddp!')















