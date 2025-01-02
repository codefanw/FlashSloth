import os
# os.environ["https_proxy"] = "http://xxx.xxx.xxx.xxx:xx"  # in case you need proxy to access Huggingface Hub
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="microsoft/phi-2", 
    revision="ef382358ec9e382308935a992d908de099b64c23",
    local_dir='checkpoints/base/phi-2',
    local_dir_use_symlinks=False
)

snapshot_download(
    repo_id="google/siglip-so400m-patch14-384", 
    revision="7067f6db2baa594bab7c6d965fe488c7ac62f1c8",
    local_dir='checkpoints/base/siglip-so400m-patch14-384',
    local_dir_use_symlinks=False
)
