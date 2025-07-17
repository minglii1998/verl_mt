import torch
from torch.utils.data import Sampler
import numpy as np
import ray
class CurriculumSampler(Sampler):
    """
    3-stage curriculum sampler controlled by ONE reward threshold.

    stage 0 → easy ∪ medium ∪ hard
    stage 1 → medium ∪ hard
    stage 2 → hard
    """
    def __init__(
        self,
        dataset,
        reward_threshold: float = 0.8,           # ← 唯一阈值
        generator: torch.Generator | None = None,
    ):
        super().__init__(dataset)
        self.dataset          = dataset
        self.generator        = generator or torch.Generator()
        self.reward_threshold = reward_threshold
        self.stage            = 0

        # ---------- 按 segment_count 分桶 ----------
        df = dataset.dataframe
        seg_cnt = np.array(
            [info.get("segment_count", 1) for info in df["extra_info"]],
            dtype=np.int32,
        )
        self.easy_idx   = np.where(seg_cnt <= 3)[0]
        self.medium_idx = np.where((seg_cnt > 3) & (seg_cnt <= 7))[0]
        self.hard_idx   = np.where(seg_cnt > 7)[0]

        self._bucket_map = {
            "easy":   self.easy_idx,
            "medium": self.medium_idx,
            "hard":   self.hard_idx,
        }
        self._refresh_allowed()

    # ---------------- Sampler API ----------------
    def __iter__(self):
        g = self.generator
        while True:
            ridx = torch.randint(len(self._allowed_idx), (1,), generator=g).item()
            yield int(self._allowed_idx[ridx])

    def __len__(self):
        return 100000                           # “无限”长度，供 DataLoader 循环

    # --------------- Curriculum control ----------
    def update(self, reward: float):
        """
        训练循环里调用：sampler.update(last_reward)
        只要 reward ≥ threshold 就升一级
        """
        if self.stage < 2 and reward >= self.reward_threshold:
            self.stage += 1
            self._refresh_allowed()

    # ----------------- helpers -------------------
    def _refresh_allowed(self):
        if self.stage == 0:
            buckets = ("easy", "medium", "hard")
        elif self.stage == 1:
            buckets = ("medium", "hard")
        else:
            buckets = ("hard",)

        self._allowed_idx = np.concatenate([self._bucket_map[b] for b in buckets])
        if len(self._allowed_idx) == 0:
            raise RuntimeError(
                f"No data for stage {self.stage}. Check segment_count or dataset."
            )
