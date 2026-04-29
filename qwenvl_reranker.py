from Qwen3_VL_Embedding.src.models.qwen3_vl_reranker import Qwen3VLReranker
import json, os



def read_metadata(file_path='/shark/zhiwen/LoRAHunter/SD_adapter_metadata/sd_lora_1/exist_file_adapters.jsonl', model_id='18377'):
    with open(file_path, 'r') as f:
        res = [json.loads(line) for line in f.readlines()]
    metadata_map = {str(one['adapter_id']): one for one in res}

    data = metadata_map[model_id]
    # print('\n')
    # print(f"{data['model_file']},{data['title']}: {data['llm_description']}")
    return f"title:{data['title']}, description: {data['llm_description']}"

def read_call_res(file_path = '/shark/zhiwen/LoRAHunter/sd_encoder/test_data/data_500_qwenemb_5-199.jsonl', num=1):
    
    with open(file_path, 'r') as f:
        res = [json.loads(line) for line in f.readlines()]
    
    data = res[num]
    retrieval_results = data['retrieval_results']
    concepts = data['extract_concept']
    concepts_map = {concept['keyword']: concept for concept in concepts}
    # print(len(retrieval_results))
    infos = {}
    for keyword, item in retrieval_results.items():
        print("=="*50)
        # print(f"{keyword}: {len(item)}")
        adapters = []
        for i, one in enumerate(item[:40]):
            
            # print(f"{i}: {one['model_file']}, score: {one['score']}")
            model_file = one['model_file']
            model_id = os.path.basename(model_file).split('/')[-1].split('.')[0]
            adapter_info = read_metadata(model_id=model_id)
            adapters.append(adapter_info)
            # print('\n')
        infos[keyword] = {'adapters': adapters,'retrieval_des': concepts_map[keyword]['retrieval_description']}
    return infos

def reranker(query, docs):
    # Specify the model path
    model_name_or_path = "/shark/zhiwen/LoRAHunter/DiffSynth-Studio/models/Qwen/Qwen3-VL-Reranker-8B"

    # Initialize the Qwen3VLEmbedder model
    model = Qwen3VLReranker(model_name_or_path=model_name_or_path)
    # We recommend enabling flash_attention_2 for better acceleration and memory saving,
    # model = Qwen3VLReranker(model_name_or_path=model_name_or_path, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2")

    # Combine queries and documents into a single input list

    inputs = {
        "instruction": "Retrieve adapters related to the user query based on adapter information.",
        # "instruction": "Retrieve images or text relevant to the user's query.",
        "query": {"text": query},
        "documents": [{"text": doc} for doc in docs],
        "fps": 1.0
    }

    # inputs = {
    #     "instruction": "Retrieve images or text relevant to the user's query.",
    #     "query": {"text": "A woman playing with her dog on a beach at sunset."},
    #     "documents": [
    #         {"text": "A woman shares a joyful moment with her golden retriever on a sun-drenched beach at sunset, as the dog offers its paw in a heartwarming display of companionship and trust."},
    #         # {"image": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg"},
    #         # {"text": "A woman shares a joyful moment with her golden retriever on a sun-drenched beach at sunset, as the dog offers its paw in a heartwarming display of companionship and trust.", "image": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg"}
    #     ],
    #     "fps": 1.0
    # }

    scores = model.process(inputs)
    print(scores)
    return scores
    # [0.8613124489784241, 0.6757137179374695, 0.8125371336936951]

if __name__ == '__main__':
    infos = read_call_res(num=3)
    keywords = list(infos.keys())
    for keyword in keywords[:1]:
        item = infos[keyword]
        print(keyword, item['retrieval_des'])
        adapters = item['adapters']
        # for adapter in adapters[:5]:
        #     print(adapter)
        #     print('\n')
        retrieval_des = item['retrieval_des']
        lst = reranker(f"{keyword},{retrieval_des}", adapters)
        topk_idx = sorted(range(len(lst)), key=lambda i: lst[i], reverse=True)[:5]
        print(topk_idx)
        for idx in topk_idx:
            print(f"{idx}, {lst[idx]}: {adapters[idx]}")
        break
    
    # main()