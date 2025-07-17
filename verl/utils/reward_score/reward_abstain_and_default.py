import ray
from verl.utils.reward_score import default_compute_score
from verl.utils.reward_score.reward_abstain import compute_score_reward_abstain

def compute_score_reward_abstain_and_default(
        data_source, 
        solution_str, 
        ground_truth, 
        extra_info=None, 
        sandbox_fusion_url=None, 
        concurrent_semaphore=None, 
        memory_limit_mb=None
        ):

    rollout_info = extra_info.get("rollout_info", None)
    if rollout_info == 'abstain':
        return compute_score_reward_abstain(solution_str, ground_truth)
    elif rollout_info == "default":
        return default_compute_score(
            data_source, 
            solution_str, 
            ground_truth, 
            extra_info=extra_info, 
            sandbox_fusion_url=sandbox_fusion_url, 
            concurrent_semaphore=concurrent_semaphore, 
            memory_limit_mb=memory_limit_mb
            )
        
