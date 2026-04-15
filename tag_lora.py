import os, json, re
from pathlib import Path
from dashscope import Generation
import dashscope
import json
import time

def find_and_sort_safetensors(directory):
    """
    查找指定目录下所有 .safetensors 文件，并按文件名中的数字排序
    
    参数:
    directory (str): 要搜索的目录路径
    
    返回:
    list: 按数字排序的文件路径列表
    """
    directory_path = Path(directory)
    
    # 检查目录是否存在
    if not directory_path.exists() or not directory_path.is_dir():
        raise ValueError(f"错误：目录 '{directory}' 不存在或不是目录")
    
    # 查找所有 .safetensors 文件（递归搜索子目录）
    safetensor_files = list(directory_path.rglob('*.safetensors'))
    
    if not safetensor_files:
        print(f"警告：在 '{directory}' 中未找到任何 .safetensors 文件")
        return []
    
    # 按数字排序（提取文件名中的数字部分）
    def extract_number(file_path):
        # 获取文件名（不含后缀）
        filename = file_path.stem
        
        # 从文件名末尾提取连续数字
        # 例如: 'aa10' -> 提取 '10', 'aa10b' -> 提取 '10'（但根据您的格式，这种情况应不存在）
        match = re.search(r'(\d+)$', filename)
        if match:
            return int(match.group(1))
        else:
            # 如果没有数字，返回最大整数（这样会排在最后）
            return float('inf')
    
    # 按提取的数字排序
    sorted_files = sorted(safetensor_files, key=extract_number)
    
    return sorted_files

# 给每个 LoRA 选取合适的prompt用来生成图片
# 利用 CARLoS 提供的10种类别的prompt
# 不全部使用，把。llm_description 给 llm ，让他选出 3 个合适的类别

# 1、扫描文件夹找出可用LoRA

root_dir = '/shark/zhiwen/LoRAHunter/Qwen_LoRA'

def find_available_lora():
    top_4000_availble = "/shark/zhiwen/LoRAHunter/Available_LoRA_carlos_tags2.jsonl"
    with open(top_4000_availble, 'r') as f:
        datas_4k = [json.loads(line) for line in f.readlines()]
    datas_4k_map = {item['model_id']: item for item in datas_4k}
    
    owners_list = [item for item in os.listdir(root_dir)
                    if os.path.isdir(os.path.join(root_dir, item))]

    print(len(owners_list))

    datas = []

    for owner in owners_list[:]:
        owner_dir = os.path.join(root_dir, owner)
        models_list = [item for item in os.listdir(owner_dir)
                        if os.path.isdir(os.path.join(owner_dir, item))]
        for model_name in models_list:
            model_dir = os.path.join(owner_dir, model_name)
            metadata_path = os.path.join(model_dir, 'metadata.json')

            model_id = f'{owner}/{model_name}'
            

            if not os.path.exists(metadata_path) or model_id in datas_4k_map:
                continue
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            models_list = find_and_sort_safetensors(model_dir)
            if len(models_list) == 0:
                continue

            data = {'model_id': f'{owner}/{model_name}', 'model_file': f'{owner}/{model_name}/{models_list[-1].name.split("/")[-1]}', 'metadata': f'{owner}/{model_name}/metadata.json','llm_description': metadata['llm_description']}
            datas.append(data)

    print(f'Total {len(datas)} LoRA')

    with open('Available_LoRA_4k-13k.jsonl', 'w', encoding='utf-8') as f:
        for item in datas:
            # 将每个字典转换为JSON字符串并写入新行
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

def llm_calling(prompt):

    category = (
        '''Portraits
            --Realistic or Stylized Human Portraits
            --Fantasy and Sci-Fi Characters
            --Anime and Cartoon Characters
            --Celebrity Look-Alikes
        Landscapes
            --Natural Landscapes
            --Sci-Fi and Futuristic Worlds
            --Fantasy Worlds and Mythical Places
            --Post-Apocalyptic and Ruined Cities
        Artistic_Styles
            --Oil Painting, Watercolor, Digital Painting
            --Cyberpunk, Synthwave, Vaporwave
            --Ukiyo-e, Baroque, Gothic, Impressionist Styles
            --Minimalist and Abstract Art
        Conceptual_Arts
            --Optical Illusions
            --Dreamlike or Mind-Bending Scenes
            --Impossible Objects and Escher-Style Designs
            --Juxtaposition of Unrelated Elements
        Animals
            --Hyper-Realistic or Stylized Animals
            --Dragons, Unicorns, Phoenixes
            --Hybrid Creatures
        Fashion
            --Futuristic or Sci-Fi Fashion
            --High-Fashion Runway Outfits
            --Medieval, Fantasy, or Historical Clothing
            --Cyberpunk and Streetwear Designs
        Vehicles
            --Futuristic Cars, Bikes, Spaceships
            --Steampunk Machinery and Robots
            --Cybernetic Enhancements on Objects
        Food
            --Hyper-Realistic Food Photography
            --Fantasy Food (Floating Cakes, Glowing Drinks)
            --Still-Life Compositions with Unique Lighting
        Cinematic
            --Scenes That Look Like Movie Frames
            --Dynamic Action Shots
            --Noir, Horror, or Detective Themes
        Logos
            --Custom Branding and Typography
            --Stylized Icons and Minimalist Logos
            --Emblems with Fantasy or Futuristic Themes
        '''
    )

    dashscope.base_http_api_url = "https://dashscope.aliyuncs.com/api/v1"
    text0 = (
        f'现在有一个图像生成模型 LoRA Adapter 的用途描述：{prompt},\n'
        f'这是LoRA的用途分类，每个一级类别下面还包括二级类别，具体的类别如下：{category}'
        f'请根据LoRA Adapter 的用途描述，给出4个最合适的一类标签，每个标签用一个逗号隔开，不要添加任何前缀或后缀。'
        f'输出样例：Portraits,Fashion,Cinematic,Animals'
    )
    example_ = "This LoRA is designed to generate images of the character Cammy White from the Street Fighter series. The adapter accurately replicates Cammy's iconic look, including her green or black high-leg leotard, red beret, red gloves, and combat boots. The images consistently feature her long blonde hair styled in twin braids, and her distinctive red headwear. The LoRA also captures her athletic and combat-ready appearance, often depicted in dynamic poses such as squatting. The overall effect is a highly detailed and realistic portrayal of Cammy White, suitable for cosplay or character illustration purposes."
    text1 = (
        f'现在有一个图像生成模型 LoRA Adapter 的用途描述：{prompt},\n'
        f'对这个描述进行缩写。压缩后长度是clip模型输入的77个token。'
        f'这是一个输出样例：{example_}'
    )

    text = (
        f'现在有一个图像生成模型 LoRA Adapter 的描述: {prompt}'
        f'你要根据这些描述写5个合适的英文prompt，LoRA Adapter 应该能对这些prompt生效，提升用这些prompt生成图像效果。\n'
        f'每句prompt长度适中，保证5句prompt具有多样性，描述不同的场景或画面，但是要保证和LoRA Adapter对应，能通过prompt检索到这个LoRA。\n'
        f'只需要返回5行字符串，每行包含1条prompt'
        
    )
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": text},
    ]
    response = dashscope.MultiModalConversation.call(
        # 若没有配置环境变量，请用百炼API Key将下行替换为：api_key = "sk-xxx",
        # api_key="sk-1827f4c92fa849c59eb88fb122b0401b",
        api_key="sk-943d9e3de8394d85a639d4facc7b8bbf",
        model="qwen3.5-plus",
        messages=messages,
        result_format="message",
        # 开启深度思考
        enable_thinking=False,
    )

    if response.status_code == 200:
        # 打印思考过程
        print("=" * 20 + "思考过程" + "=" * 20)
        # print(response.output.choices[0].message.reasoning_content)
        
        # 打印回复
        print("=" * 20 + "完整回复" + "=" * 20)
        print(response.output.choices[0].message.content[0]['text'])
        
        return response.output.choices[0].message.content[0]['text']
    else:
        print(f"HTTP返回码：{response.status_code}")
        print(f"错误码：{response.code}")
        print(f"错误信息：{response.message}")
        return None

def use_llm_tag():
    filename = 'Available_LoRA_carlos_tags2.jsonl'
    with open('Available_LoRA_carlos_tags.jsonl', 'r', encoding='utf-8') as f:
        datas = [json.loads(line) for line in f.readlines()]
    
    for data in datas[:]:
        prompt = data['llm_description']
        model_id = data['model_id']
        print("\n")
        print(f"============ {model_id} ================")
        res = llm_calling(prompt)
        if res is None:
            continue
        data['short_description'] = res
        with open(filename, 'a', encoding='utf-8') as f:
            json_str = json.dumps(data, ensure_ascii=False)
            f.write(json_str + '\n')
        

def use_llm_prompt():
    filename = 'Available_LoRA_carlos_tags2_prompt.jsonl'

    with open(filename, 'r', encoding='utf-8') as f:
        exist_lora_list = [json.loads(line)['model_id'] for line in f.readlines()]
    

    source_file = 'source_files/LoRA_QWEN_IMAGE_20_B_top_4000.jsonl'
    with open(source_file, 'r', encoding='utf-8') as f:
        source_datas = [json.loads(line) for line in f.readlines()]
        
            

    sourcedata_by_id = {item["model_id"]: item for item in source_datas}


    with open('Available_LoRA_carlos_tags2.jsonl', 'r', encoding='utf-8') as f:
        datas = [json.loads(line) for line in f.readlines()]
    for data in datas[:]:
        llm_description = data['llm_description']
        model_id = data['model_id']
        if model_id in exist_lora_list:
            print(f"{model_id} 已经打标完")
            continue
        source_data = sourcedata_by_id[model_id]
        title = source_data['title']
        description = source_data['description']
        trigger_words = source_data['trigger_words']
        prompt = (
            f'title:{title} \nDescription: {description} \n'
            f'TriggerWords: {trigger_words} \nLLM Description: {llm_description}'
        )
        print("\n")
        print(f"============ {model_id} ================")
        res = llm_calling(prompt)
        if res is None:
            continue
        res = res.split('\n')
        print(len(res))
        data['prompts'] = res
        with open(filename, 'a', encoding='utf-8') as f:
            json_str = json.dumps(data, ensure_ascii=False)
            f.write(json_str + '\n')
        time.sleep(5)


if __name__ == '__main__':
    find_available_lora()
    # use_llm_tag()
    prompt = 'This LoRA adapter emphasizes the generation of highly defined male torsos with exaggerated abdominal musculature, rendered in a photorealistic 3D style. It consistently enhances the visibility and definition of abdominal muscles, pectorals, and shoulder contours, producing a hyper-realistic, sculpted appearance that mimics digital 3D modeling or bodybuilding photography. The subject is typically a shirtless man, often shown from the chest down to the waist, with a focus on anatomical precision and lighting that accentuates muscle relief through shadows and highlights. The visual style leans toward realism with smooth skin texture, natural skin tone, and subtle details like visible veins or slight sheen, suggesting a fitness or body-positive aesthetic. Backgrounds are usually mundane indoor settings—such as bathrooms or dressing areas—with minimal detail, serving only to frame the torso without drawing attention away. Clothing elements, when present, are limited to low-rise underwear or briefs, often featuring branded elastic bands, which further emphasize the exposed midsection. The adapter appears to prioritize physical form over facial features or full-body context, making it best suited for generating images focused on muscular physique and body structure. The overall mood is clean, direct, and physically idealized, aligning with fitness or body enhancement themes'
    # prompt = "This LoRA adapter induces a flat, abstract visual style characterized by simplified geometric forms, uniform color fields, and minimal depth cues. It tends to transform subjects into stylized, two-dimensional representations with clean edges and solid color blocks, often eliminating texture, shading, and realistic lighting. The resulting images resemble digital illustrations or graphic designs, emphasizing clarity and form over naturalism. While the original prompts are unknown, the consistent output suggests a focus on abstract composition rather than specific subjects, with figures or objects reduced to essential shapes and placed against plain or gradient backgrounds. The most affected elements include overall structure, color distribution, and spatial relationships, which are flattened into a cohesive, minimalist aesthetic. The adapter appears to prioritize design coherence and visual simplicity, producing outputs that feel like modern digital art or interface graphics. There is no strong emphasis on particular subjects, poses, or environments—instead, the core effect lies in the stylistic transformation toward abstraction and uniformity."
    # tags = llm_calling(prompt)
    # tags = tags.split('\n')

    # prompt_list = json.dumps(tags)
    # print(prompt_list[0])
    # print(tags)
    # print(tags.split(','))
    # use_llm_prompt()

# nohup python tag_lora.py > log/tag_prompt_Available_LoRA_carlos_tags2.log 2>&1 &