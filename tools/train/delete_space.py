from huggingface_hub import HfApi
import os
api = HfApi(token=os.environ.get('HF_TOKEN'))
api.delete_repo('Arkavo/autotrain-torg-ministral-8b-lora', repo_type='space')
print('Space deleted')
