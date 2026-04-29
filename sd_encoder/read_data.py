import os
import json
import random




def read_reank_res(file_path='/shark/zhiwen/LoRAHunter/sd_encoder/test_data/data_500_clipemb_2-1248_diverse.jsonl',num=3):
    with open(file_path, 'r') as f:
        res = [json.loads(line) for line in f.readlines()]
    print(len(res))
    data = res[num]
    print(data['prompt'])
    print(data.keys())
    combination_results = data['combination_results']
    print(combination_results.keys())
    print(combination_results['concept_names'])

    print(len(combination_results['diverse_topk']))
    for i in range(10):
        print(combination_results['diverse_topk'][i])

    res = combination_results['diverse_topk'][0]

    # iid = random.randint(0, len(combination_results['candidate_pool'])-1)
    # res = combination_results['candidate_pool'][iid]
    # print(f"iid :{iid}, {res}")
    
    model_files = res['model_files']
    model_ids = [os.path.basename(one).split('/')[-1].split('.')[0] for one in model_files]
    return model_ids
    # return res


def read_llmreank_res(file_path='/shark/zhiwen/LoRAHunter/sd_encoder/test_data/data_500_qwenemb_4-68_reank_beam.jsonl',num=0):
    with open(file_path, 'r') as f:
        res = [json.loads(line) for line in f.readlines()]
    print(len(res))
    data = res[num]
    print(data['prompt'])
    # print(data.keys())
    rerank_results = data['rerank_results']

    for keyw, loras in rerank_results.items():
        print(keyw)

        lora_list = [x["lora"] for x in sorted(loras, key=lambda x: int(x["score"]), reverse=True)]
        for i, model_id in enumerate(lora_list[:5]):
            read_metadata(model_id=model_id)
            print('\n')



def read_metadata(file_path='/shark/zhiwen/LoRAHunter/SD_adapter_metadata/sd_lora_1/exist_file_adapters.jsonl', model_id='18377'):
    with open(file_path, 'r') as f:
        res = [json.loads(line) for line in f.readlines()]
    metadata_map = {str(one['adapter_id']): one for one in res}

    data = metadata_map[model_id]
    print('\n')
    print(f"{data['model_file']},{data['title']}: {data['llm_description']}")



def read_call_res(file_path = '/shark/zhiwen/LoRAHunter/sd_encoder/test_data/data_500_clipemb_2-1248.jsonl', num=1):
    
    with open(file_path, 'r') as f:
        res = [json.loads(line) for line in f.readlines()]
    
    data = res[num]
    retrieval_results = data['retrieval_results']
    print(len(retrieval_results))
    for keyword, item in retrieval_results.items():
        print("=="*50)
        print(f"{keyword}: {len(item)}")
        for i, one in enumerate(item[:10]):
            
            print(f"{i}: {one['model_file']}, score: {one['score']}")
            model_file = one['model_file']
            model_id = os.path.basename(model_file).split('/')[-1].split('.')[0]
            read_metadata(model_id=model_id)
            print('\n')

if __name__ == '__main__':
    model_ids = read_reank_res(num=0)
    for model_id in model_ids:
        read_metadata(model_id=model_id)
    # # read_metadata(model_id='18377')

    # read_call_res()
    # read_llmreank_res(num=3)