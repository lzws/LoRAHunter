import json
import multiprocessing as mp
from queue import Empty


# 这里假设你已经有这个类
from rerank_beam import HybridCombinationPipeline


def gpu_worker(
    worker_id,
    gpu_id,
    task_queue,
    result_queue,
    lora_index_path,

    alpha,
    beta_compat,
    eta_redund,
    delta,

    lambda_concept,
    lambda_prompt,

    global_tau,
    route_text_ratio,
    task,

    redundancy_tau,
    compat_sim_suppress,
    global_mix_uniform,

    exhaustive_max_concepts,
    merge_pool_size,

    diversity_quality_weight,
    diversity_sim_weight,
    diversity_repeat_weight,

    sim_gamma,
    max_repeat_per_slot,

    per_concept_top_k,
    top_k,
    prefilter_top_n,
    repeat_penalty_mode,
):
    """
    每个 worker 绑定到一张固定 GPU
    """
    device = f"cuda:{gpu_id}"
    print(f"[Worker {worker_id}] start on {device}")

    pipeline = HybridCombinationPipeline(
        lora_index_path=lora_index_path,
        device=device,

        alpha=alpha,
        beta_compat=beta_compat,
        eta_redund=eta_redund,
        delta=delta,

        lambda_concept=lambda_concept,
        lambda_prompt=lambda_prompt,

        global_tau=global_tau,
        route_text_ratio=route_text_ratio,
        task=task,

        redundancy_tau=redundancy_tau,
        compat_sim_suppress=compat_sim_suppress,
        global_mix_uniform=global_mix_uniform,

        exhaustive_max_concepts=exhaustive_max_concepts,
        merge_pool_size=merge_pool_size,

        diversity_quality_weight=diversity_quality_weight,
        diversity_sim_weight=diversity_sim_weight,
        diversity_repeat_weight=diversity_repeat_weight,

        sim_gamma=sim_gamma,
        max_repeat_per_slot=max_repeat_per_slot,
    )

    while True:
        task_item = task_queue.get()
        if task_item is None:
            print(f"[Worker {worker_id}] received stop signal")
            break

        item_idx, data = task_item

        try:
            result = pipeline.optimize_one_data(
                data=data,
                per_concept_top_k=per_concept_top_k,
                top_k=top_k,
                prefilter_top_n=prefilter_top_n,
                repeat_penalty_mode=repeat_penalty_mode,
            )
            result_queue.put((item_idx, result, None))
        except Exception as e:
            data["combination_results"] = {
                "candidate_pool": [],
                "diverse_topk": [],
                "error": str(e),
            }
            result_queue.put((item_idx, data, str(e)))

    print(f"[Worker {worker_id}] exit")


def optimize_retrieval_results_file_hybrid_multi_gpu(
    retrieval_result_path: str,
    lora_index_path: str,
    output_path: str,

    gpu_ids,   # 例如 [0,1,2,3]

    per_concept_top_k: int = 40,
    top_k: int = 10,
    prefilter_top_n: int = 500,
    repeat_penalty_mode: str = "max",

    alpha: float = 1.0,
    beta_compat: float = 0.3,
    eta_redund: float = 0.4,
    delta: float = 0.5,

    lambda_concept: float = 1.0,
    lambda_prompt: float = 0.7,

    global_tau: float = 0.2,
    route_text_ratio: float = 0.8,
    task: str = "qwenemb",

    redundancy_tau: float = 0.3,
    compat_sim_suppress: float = 0.5,
    global_mix_uniform: float = 0.3,

    exhaustive_max_concepts: int = 3,
    merge_pool_size: int = 500,

    diversity_quality_weight: float = 0.7,
    diversity_sim_weight: float = 0.15,
    diversity_repeat_weight: float = 0.15,

    sim_gamma: float = 0.7,
    max_repeat_per_slot: int = 2,
):
    with open(retrieval_result_path, "r", encoding="utf-8") as f:
        datas = [json.loads(line) for line in f]

    num_items = len(datas)
    results = [None] * num_items

    ctx = mp.get_context("spawn")
    task_queue = ctx.Queue(maxsize=len(gpu_ids) * 4)
    result_queue = ctx.Queue()

    workers = []
    for worker_id, gpu_id in enumerate(gpu_ids):
        p = ctx.Process(
            target=gpu_worker,
            args=(
                worker_id,
                gpu_id,
                task_queue,
                result_queue,
                lora_index_path,

                alpha,
                beta_compat,
                eta_redund,
                delta,

                lambda_concept,
                lambda_prompt,

                global_tau,
                route_text_ratio,
                task,

                redundancy_tau,
                compat_sim_suppress,
                global_mix_uniform,

                exhaustive_max_concepts,
                merge_pool_size,

                diversity_quality_weight,
                diversity_sim_weight,
                diversity_repeat_weight,

                sim_gamma,
                max_repeat_per_slot,

                per_concept_top_k,
                top_k,
                prefilter_top_n,
                repeat_penalty_mode,
            ),
        )
        p.start()
        workers.append(p)

    # 投递任务
    for i, data in enumerate(datas):
        task_queue.put((i, data))

    # 给每个 worker 一个结束信号
    for _ in workers:
        task_queue.put(None)

    # 收集结果
    num_done = 0
    while num_done < num_items:
        item_idx, result_data, err = result_queue.get()
        results[item_idx] = result_data
        num_done += 1
        print(f"[Main] done {num_done}/{num_items} | idx={item_idx} | err={err}")

    # 等待 worker 退出
    for p in workers:
        p.join()

    with open(output_path, "w", encoding="utf-8") as fout:
        for data in results:
            fout.write(json.dumps(data, ensure_ascii=False) + "\n")

    print(f"[Done] saved optimized results to {output_path}")


if __name__ == "__main__":
    index_paths = ["lora_index/qwenemb_cosinsmoothl1_all_5-199.pt", "lora_index/clipemb_contastive_all_2-1248.pt"]

    optimize_retrieval_results_file_hybrid_multi_gpu(
        retrieval_result_path="test_data/data_500_qwenemb_5-199_2.jsonl",
        lora_index_path=index_paths[0],
        output_path="test_data/data_500_qwenemb_5-199_2-0_diverse.jsonl",

        gpu_ids=[1, 2, 3, 4, 5, 6],   # 你有哪些 GPU 就写哪些

        per_concept_top_k=30,
        top_k=10,
        prefilter_top_n=300,
        repeat_penalty_mode="max",

        alpha=1.0,
        beta_compat=0.3,
        eta_redund=0.4,
        delta=0.5,

        lambda_concept=1.0,
        lambda_prompt=0.7,

        global_tau=0.2,
        route_text_ratio=0.8,
        task="qwenemb",

        redundancy_tau=0.3,
        compat_sim_suppress=0.5,
        global_mix_uniform=0.3,

        exhaustive_max_concepts=3,
        merge_pool_size=200,

        diversity_quality_weight=0.7,
        diversity_sim_weight=0.15,
        diversity_repeat_weight=0.15,

        sim_gamma=0.7,
        max_repeat_per_slot=2,
    )
