import os
import json





def read_reank_res(file_path='/shark/zhiwen/LoRAHunter/sd_encoder/test_data/250_clipemb-7_all_res_combinations_reank.jsonl',num=132):
    with open(file_path, 'r') as f:
        res = [json.loads(line) for line in f.readlines()]
    print(len(res))
    data = res[num]
    print(data['prompt'])
    # print(data.keys())
    combination_results = data['combination_results']
    print(combination_results.keys())
    print(combination_results['concept_names'])

    print(combination_results['diverse_combinations'][0])
    print(combination_results['diverse_combinations'][2])
    res = combination_results['diverse_combinations'][0]
    model_files = res['model_files']
    model_ids = [os.path.basename(one).split('/')[-1].split('.')[0] for one in model_files]
    return model_ids
    # return res

def read_metadata(file_path='/shark/zhiwen/LoRAHunter/SD_adapter_metadata/sd_lora_1/exist_file_adapters.jsonl', model_id='87277'):
    with open(file_path, 'r') as f:
        res = [json.loads(line) for line in f.readlines()]
    metadata_map = {str(one['adapter_id']): one for one in res}

    data = metadata_map[model_id]
    print(f"{data['model_file']},{data['title']}: {data['llm_description']}")

if __name__ == '__main__':
    model_ids = read_reank_res()
    for model_id in model_ids:
        read_metadata(model_id=model_id)
    # read_metadata(model_id='113747')