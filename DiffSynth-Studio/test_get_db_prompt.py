import json,os
import pandas as pd

def get_train_prompt():
    '''
    1. 从coco里面挑选1w条prompt
    2. CARLos 中所有的prompt 560条
    '''
    all_datas = []
    global_id = 0
    # coco_path = "/shark/zhiwen/benchmark/EraseBenchmark/dataset/dataset/coco_30k.csv"
    # df = pd.read_csv(coco_path)
    # df_sampled = df.sample(n=10000, random_state=42)
    # for i, row in df_sampled.iterrows():
    #     image_id = row['image_id']
    #     prompt = row['prompt']
    #     all_datas.append({
    #         "iid":global_id,
    #         "source": "coco",
    #         "source_id": image_id,
    #         "prompt": prompt
    #     })
    #     global_id += 1
    
    # for category in prompts_by_category:
    #     for sub_category, prompts in prompts_by_category[category].items():
    #         for p in prompts:
    #             all_datas.append({
    #                 "iid":global_id,
    #                 "source": "CARLos",
    #                 "source_id": category,
    #                 "prompt": p
    #             })
    #             global_id += 1
    

    # diffusion_db_path = "/shark/zhiwen/lora_diff_length/diffusiondb_5000.csv"
    # df = pd.read_csv(diffusion_db_path)
    # # df_sampled = df.sample(n=5000, random_state=42)
    # for i, row in df.iterrows():
    #     prompt = row['prompt']
    #     all_datas.append({
    #         "iid":global_id,
    #         "source": "diffusiondb",
    #         "source_id": i,
    #         "prompt": prompt
    #     })
    #     global_id += 1
    
    diffusion_db_path = "rank_dataset/diffusiondb_test_200.csv"
    df = pd.read_csv(diffusion_db_path)
    # df_sampled = df.sample(n=5000, random_state=42)
    for i, row in df.iterrows():
        prompt = row['prompt']
        all_datas.append({
            "iid":global_id,
            "prompt": prompt
        })
        global_id += 1

    print(f"[Done] collected {len(all_datas)} prompts")
    with open("rank_dataset/diffusiondb_test_200.jsonl", "w", encoding="utf-8") as f:
        for data in all_datas:
            json_str = json.dumps(data, ensure_ascii=False)
            f.write(json_str + '\n')



if __name__ == "__main__":
    get_train_prompt()
    # data_path = '/shark/zhiwen/LoRA-fusion/dataset/prompt/metadata.parquet'
    # data = pd.read_parquet(data_path,engine='pyarrow')

    # subset = data[['prompt', 'seed']].sample(n=200)
    # subset['iid'] = range(len(subset))
    # subset.to_csv('rank_dataset/diffusiondb_test_200.csv',index=False)
