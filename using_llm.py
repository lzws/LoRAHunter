import os, json

from dashscope import Generation
import dashscope
from openai import OpenAI

dashscope.base_http_api_url = "https://dashscope.aliyuncs.com/api/v1"

def llm_calling(messages):


    dashscope.base_http_api_url = "https://dashscope.aliyuncs.com/api/v1"
    # dashscope.base_http_api_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    # messages = [
    #     {"role": "system", "content": "You are a helpful assistant."},
    #     {"role": "user", "content": text},
    # ]
    # MultiModalConversation、 qwen3.5-plus
    response = dashscope.MultiModalConversation.call(
        # 若没有配置环境变量，请用百炼API Key将下行替换为：api_key = "sk-xxx",
        # api_key="sk-1827f4c92fa849c59eb88fb122b0401b",
        api_key="sk-943d9e3de8394d85a639d4facc7b8bbf",
        model="qwen3.5-plus",
        messages=messages,
        result_format="message",
        # 开启深度思考
        enable_thinking=True,
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


def llm_openai(messages):
    zzz_url = "https://api.zhizengzeng.com/v1"
    zzz_api_key = "sk-zk20bb17720c0c83981439ad403ef1d714469dc899b91558"
    model = 'gpt-4o-mini'
    client = OpenAI(
        api_key=zzz_api_key,
        base_url=zzz_url,
    )

    response = client.chat.completions.create(model=model, messages=messages,)
    if response:
        print(f"response: {response}, \n\n")
        response_str = response.choices[0].message.content
        # print(response_str)
        return response_str
    else:
        print(f"HTTP返回码：{response.status_code}")
        print(f"错误码：{response.code}")
        print(f"错误信息：{response.message}")
        return None




# '5) Provide a search description: For each keyword, to retrieve a corresponding adapter, briefly describe what kind of adapter should be retrieved for this keyword. Directly stating the function the ideal adapter should perform will maximize the matching of adapter descriptions in the database. Below are examples of adapter descriptions from the adapter library：\n\n'
#           "- This LoRA generates realistic sci-fi humanoid robots with metallic bodies, articulated joints, and illuminated digital eyes. It features \"PiR\" branding on torsos within clean lab or exhibition settings. The style is photorealistic with sharp lighting and reflections, emphasizing the robot's mechanical design as the central subject against neutral backgrounds for a technical, futuristic mood. \n"
#           "- This LoRA generates individuals in traditional Chinese Hanfu: cross-collar, waist-level, two-piece with outer garment. It features layered V-collared robes with decorative trim, rich silk textures, and dark muted tones accented by gold. The aesthetic is realistic with soft lighting, natural skin, and classical elegance. Backgrounds are minimal or blurred with traditional elements like wood or red textiles. Hair includes elaborate updos and pins. Best for female subjects in static poses, emphasizing detailed costumes and historical authenticity. \n"


def extract_concept(text):

    # text = "A market square in a medieval village"
    # text = "A young man wearing a catchers mitt on top of a baseball field"

    extract_concept_prompt_en = (
        f"You are a keyword extractor specialized in identifying concepts and their descriptive adjectives from prompts used in text-to-image models. Your task is to detect key concepts (keywords with their adjectives) that significantly influence the content, style, or composition of the generated image. Ensure that the extracted concepts are relevant to the visual elements of the scene and describe distinct entities or attributes. Do not split related concepts that describe a single entity. Avoid extracting verbs describing actions. "
        f"\n\n"
        "Key Instructions: \n"
        "1) Extract descriptive concepts: Always extract keywords that include any relevant descriptive adjectives or qualifiers that modify them in the prompt. If adjectives or modifiers are provided, ensure they are part of the concept, as they provide crucial detail to the entity. For example: \n\n"
        "- For the prompt ”a large, ancient stone castle,” return the keyword ”large, ancient stone castle” rather than just ”castle.” The adjectives ”large,” ”ancient,” and ”stone” provide important context and description that shape the entity’s appearance and character. \n"
        "- For the prompt ”a vibrant painting of a tropical sunset,” return the keyword ”vibrant painting of a tropical sunset,” as ”vibrant” enhances the concept of the painting and ”tropical” adds a layer of specificity to the sunset. \n"
        '2) Focus on impactful concepts: Keywords should describe specific entities, concepts, styles, or attributes central to the generated image. Avoid overly general terms (e.g., ”thing” or ”place”). Ignore verbs describing an action.” \n\n'
        '3) Combine related concepts: If multiple keywords collectively describe a single entity, treat them as one unified concept. Do not split descriptive phrases or components that together define the same object, scene, or idea. If extracted keywords simply describe or qualify a specific entity, they must be merged into a single concept. For example: \n'
        ' - In the prompt ”a photo of San Francisco’s Golden Gate Bridge,” the entire phrase should be treated as one concept, ”a photo of San Francisco’s Golden Gate Bridge,” because all parts describe a single entity—the photo. \n'
        ' - In the prompt ”a picture of some food on the plate,” the entire phrase should be treated as one concept, ”a picture of some food on the plate,” because all parts collectively describe the picture and its content. \n\n' 
        '4) Provide explanations: For each keyword, provide a concise explanation of why it was chosen, focusing on its role in shaping the image’s appearance, theme, or composition. \n'
        '5) Provide search description: For each keyword, provide a description to retrieve the corresponding adapter. This description can retrieve a variety of adapters related to the keyword. \n'
        '6) Stick to the specified format: The output must follow the JSON format provided below within <output format> and </output format> tags. \n'
        f'\n\n'
        '''<output format> 
        [ 
            { "retrieval_description":"[Caption for retrieving the adapter]", ”explanation”: ”[Reason for selecting this keyword]”, ”keyword”: ”[Keyword identified]” }, {"retrieval_description":"[Caption for retrieving the adapter]",  ”explanation”: ”[Reason for selecting this keyword]”, ”keyword”: ”[Keyword identified]” }, ... 
        ]   
        </output format>
        '''
        '\n\n'
        f'Extract and explain keywords for the following prompt enclosed within <prompt> and </prompt> tags: \n'
        f'<prompt> {text} </prompt>'  
        f'\n\n'
        'Only Provide your response in the JSON format specified above without the <output format> and “json tags".'
    )

    extract_concept_prompt = (
        f"您是一名关键词提取器，专门负责从文本转图像模型中的提示信息中识别概念及其描述性形容词。您的任务是检测对生成图像的内容、风格或构图有显著影响的关键概念（关键词及其形容词）。请确保提取的概念与场景的视觉元素相关，并描述不同的实体或属性。请勿拆分描述同一实体的相关概念。避免提取描述动作的动词。"
        f"\n\n"
        "Key Instructions: \n"
        "1) 提取描述性概念：始终提取包含相关描述性形容词或限定词的关键词，这些关键词会在提示中修饰它们。如果提供了形容词或修饰词，请确保它们是概念的一部分，因为它们为实体提供了关键细节。例如: \n\n"
        "- For the prompt ”a large, ancient stone castle,” return the keyword ”large, ancient stone castle” rather than just ”castle.” The adjectives ”large,” ”ancient,” and ”stone” provide important context and description that shape the entity’s appearance and character. \n"
        "- For the prompt ”a vibrant painting of a tropical sunset,” return the keyword ”vibrant painting of a tropical sunset,” as ”vibrant” enhances the concept of the painting and ”tropical” adds a layer of specificity to the sunset. \n"
        '2) 聚焦于具有影响力的概念：关键词应描述与生成图像密切相关的特定实体、概念、风格或属性。避免使用过于笼统的词语（例如“事物”或“地点”）。忽略描述动作的动词。” \n\n'
        '3) 合并相关概念：如果多个关键词共同描述同一个实体，则应将它们视为一个统一的概念。不要拆分共同定义同一对象、场景或概念的描述性短语或组成部分。如果提取的关键词只是描述或限定某个特定实体，则必须将它们合并为一个概念。. For example: \n'
        ' - In the prompt ”a photo of San Francisco’s Golden Gate Bridge,” the entire phrase should be treated as one concept, ”a photo of San Francisco’s Golden Gate Bridge,” because all parts describe a single entity—the photo. \n'
        ' - In the prompt ”a picture of some food on the plate,” the entire phrase should be treated as one concept, ”a picture of some food on the plate,” because all parts collectively describe the picture and its content. \n\n' 
        '4) 提供解释：针对每个关键词，简要解释其选择原因，重点说明其在塑造图像外观、主题或构图方面的作用。 \n'
        '5) 提供一个检索描述：对于每个关键词，要检索一个对应的适配器，简要写几句话用来描述这个关键词应该检索什么样的适配器。直接写理想的适配器要发挥什么作用，可以最大程度匹配数据库里面适配器的描述，下面是适配器库里面的适配器的描述示例：\n\n'
          "- This LoRA generates realistic sci-fi humanoid robots with metallic bodies, articulated joints, and illuminated digital eyes. It features \"PiR\" branding on torsos within clean lab or exhibition settings. The style is photorealistic with sharp lighting and reflections, emphasizing the robot's mechanical design as the central subject against neutral backgrounds for a technical, futuristic mood. \n"
          "- This LoRA generates individuals in traditional Chinese Hanfu: cross-collar, waist-level, two-piece with outer garment. It features layered V-collared robes with decorative trim, rich silk textures, and dark muted tones accented by gold. The aesthetic is realistic with soft lighting, natural skin, and classical elegance. Backgrounds are minimal or blurred with traditional elements like wood or red textiles. Hair includes elaborate updos and pins. Best for female subjects in static poses, emphasizing detailed costumes and historical authenticity."
        f"\n\n"
        '6) Stick to the specified format: The output must follow the JSON format provided below within <output format> and </output format> tags. \n'
        '''<output format> 
        [ 
            { "retrieval_description":"[Caption for retrieving the adapter]", ”explanation”: ”[Reason for selecting this keyword]”, ”keyword”: ”[Keyword identified]” }, {"retrieval_description":"[Caption for retrieving the adapter]",  ”explanation”: ”[Reason for selecting this keyword]”, ”keyword”: ”[Keyword identified]” }, ... 
        ]  
        </output format>
        '''
        '\n\n'
        f'提取并解释以下提示中包含在 <prompt> 和 </prompt> 标签内的关键词。: \n'
        f'<prompt> {text} </prompt>'  
        f'\n\n'
        '请以上面指定的 JSON 格式提供您的回复，不要包含 <output format> 和“json” 标签。'
    )

    extract_concept_prompt_zh = (
        f"您是一名关键词提取器，专门负责从文本转图像模型中的提示信息中识别概念及其描述性形容词。您的任务是检测对生成图像的内容、风格或构图有显著影响的关键概念（关键词及其形容词）。请确保提取的概念与场景的视觉元素相关，并描述不同的实体或属性。请勿拆分描述同一实体的相关概念。避免提取描述动作的动词。"
        f"\n\n"
        "Key Instructions: \n"
        "1) 提取描述性概念,忽略位置关系。例如对于prompt 'table with cake', 提取'table'和'cake'两个概念"
        '2) 聚焦于具有影响力的概念：关键词应描述与生成图像密切相关的特定实体、概念、风格或属性。避免使用过于笼统的词语（例如“事物”或“地点”）。忽略描述动作的动词。” \n\n'
        '3) 合并相关概念：如果多个关键词共同描述同一个实体，则应将它们视为一个统一的概念。不要拆分共同定义同一对象、场景或概念的描述性短语或组成部分。如果提取的关键词只是描述或限定某个特定实体，则必须将它们合并为一个概念。. For example: \n'
        ' - In the prompt ”a photo of San Francisco’s Golden Gate Bridge,” the entire phrase should be treated as one concept, ”a photo of San Francisco’s Golden Gate Bridge,” because all parts describe a single entity—the photo. \n'
        ' - In the prompt ”a picture of some food on the plate,” the entire phrase should be treated as one concept, ”a picture of some food on the plate,” because all parts collectively describe the picture and its content. \n\n' 
        '4) 提供解释：针对每个关键词，简要解释其选择原因，重点说明其在塑造图像外观、主题或构图方面的作用。解释用英文 \n'
        '5) 提供一个检索描述：对于每个关键词，要检索一个对应的适配器，简要写几句英文用来描述这个关键词应该检索什么样的适配器。直接写理想的适配器要发挥什么作用，可以最大程度匹配数据库里面适配器的描述，要尽可能避免语义干扰，例如概念“火车”，理想的适配器应该是描述火车的主体，火车的外观。而不是火车站之类的适配器。下面是适配器库里面的适配器的描述示例：\n\n'
          "- This LoRA generates realistic sci-fi humanoid robots with metallic bodies, articulated joints, and illuminated digital eyes. It features \"PiR\" branding on torsos within clean lab or exhibition settings. The style is photorealistic with sharp lighting and reflections, emphasizing the robot's mechanical design as the central subject against neutral backgrounds for a technical, futuristic mood. \n"
          "- This LoRA generates individuals in traditional Chinese Hanfu: cross-collar, waist-level, two-piece with outer garment. It features layered V-collared robes with decorative trim, rich silk textures, and dark muted tones accented by gold. The aesthetic is realistic with soft lighting, natural skin, and classical elegance. Backgrounds are minimal or blurred with traditional elements like wood or red textiles. Hair includes elaborate updos and pins. Best for female subjects in static poses, emphasizing detailed costumes and historical authenticity."
        f"\n\n"
        '6) Stick to the specified format: The output must follow the JSON format provided below within <output format> and </output format> tags. \n'
        '''<output format> 
        [ 
            { "retrieval_description":"[Caption for retrieving the adapter]", ”explanation”: ”[Reason for selecting this keyword]”, ”keyword”: ”[Keyword identified]” }, {"retrieval_description":"[Caption for retrieving the adapter]",  ”explanation”: ”[Reason for selecting this keyword]”, ”keyword”: ”[Keyword identified]” }, ... 
        ]  
        </output format>
        '''
        '\n\n'
        f'提取并解释以下提示中包含在 <prompt> 和 </prompt> 标签内的关键词。: \n'
        f'<prompt> {text} </prompt>'  
        f'\n\n'
        '请以上面指定的 JSON 格式提供您的回复，不要包含 <output format> 和“json” 标签。'
    )

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": extract_concept_prompt_zh},
    ]

    res = llm_calling(messages)
    # res = llm_openai(messages)

    res = json.loads(res)
    return res


if __name__ == "__main__":
    prompt = "A fantastical witch in a forest in vibrant style"
    prompt = "A boat dock with two planes floating on the water."
    prompt = "A woman and child at table with a cake."
    # res = extract_concept(prompt)


    metadata_path = 'test_data/retrieval_testdata_500.jsonl'
    with open(metadata_path, 'r', encoding='utf-8') as f:
        datas = [json.loads(line) for line in f.readlines()]
    for data in datas[498:]:
        prompt = data['prompt']
        res = extract_concept(prompt)
        data['extract_concept'] = res
        with open('test_data/retrieval_testdata_500_extract_2-1.jsonl', 'a', encoding='utf-8') as f:
            f.write(json.dumps(data, ensure_ascii=False) + '\n')