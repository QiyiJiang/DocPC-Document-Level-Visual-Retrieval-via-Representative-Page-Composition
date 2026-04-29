import torch
import torch.nn.functional as F  # noqa: N812
from torch.nn import CrossEntropyLoss
from typing import List, Optional, Tuple
import math


class ColbertModule(torch.nn.Module):
    """
    Base module for ColBERT losses, handling shared utilities and hyperparameters.

    Args:
        max_batch_size (int): Maximum batch size for pre-allocating index buffer.
        tau (float): Temperature for smooth-max approximation.
        norm_tol (float): Tolerance for score normalization bounds.
        filter_threshold (float): Ratio threshold for pos-aware negative filtering.
        filter_factor (float): Multiplicative factor to down-weight high negatives.
    """

    def __init__(
        self,
        max_batch_size: int = 1024,
        tau: float = 0.1,
        norm_tol: float = 1e-3,
        filter_threshold: float = 0.95,
        filter_factor: float = 0.5,
    ):
        super().__init__()
        self.register_buffer("idx_buffer", torch.arange(max_batch_size), persistent=False)
        self.tau = tau
        self.norm_tol = norm_tol
        self.filter_threshold = filter_threshold
        self.filter_factor = filter_factor

    def _get_idx(self, batch_size: int, offset: int, device: torch.device):
        """
        Retrieve index and positive index tensors for in-batch losses.
        """
        idx = self.idx_buffer[:batch_size].to(device)
        return idx, idx + offset

    def _smooth_max(self, scores: torch.Tensor, dim: int) -> torch.Tensor:
        """
        Compute smooth max via log-sum-exp along a given dimension.
        """
        return self.tau * torch.logsumexp(scores / self.tau, dim=dim)

    def _apply_normalization(self, scores: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """
        Normalize scores by query lengths and enforce bounds.

        Args:
            scores (Tensor): Unnormalized score matrix [B, C].
            lengths (Tensor): Query lengths [B].

        Returns:
            Tensor: Normalized scores.

        Raises:
            ValueError: If normalized scores exceed tolerance.
        """
        normalized = scores / lengths.unsqueeze(1)
        mn, mx = torch.aminmax(normalized)
        if mn < -self.norm_tol or mx > 1 + self.norm_tol:
            raise ValueError(
                f"Scores out of bounds after normalization: "
                f"min={mn.item():.4f}, max={mx.item():.4f}, tol={self.norm_tol}"
            )
        return normalized

    def _aggregate(
        self,
        scores_raw: torch.Tensor,
        use_smooth_max: bool,
        dim_max: int,
        dim_sum: int,
    ) -> torch.Tensor:
        """
        Aggregate token-level scores into document-level.

        Args:
            scores_raw (Tensor): Raw scores tensor.
            use_smooth_max (bool): Use smooth-max if True.
            dim_max (int): Dimension to perform max/logsumexp.
            dim_sum (int): Dimension to sum over after max.
        """
        if use_smooth_max:
            return self._smooth_max(scores_raw, dim=dim_max).sum(dim=dim_sum)
        return scores_raw.amax(dim=dim_max).sum(dim=dim_sum)

    def _filter_high_negatives(self, scores: torch.Tensor, pos_idx: torch.Tensor) -> None:
        """
        Down-weight negatives whose score exceeds a fraction of the positive score.

        Args:
            scores (Tensor): In-batch score matrix [B, B].
            pos_idx (Tensor): Positive indices for each query in batch.
        """
        batch_size = scores.size(0)
        idx = self.idx_buffer[:batch_size].to(scores.device)
        pos_scores = scores[idx, pos_idx]
        thresh = self.filter_threshold * pos_scores.unsqueeze(1)
        mask = scores > thresh
        mask[idx, pos_idx] = False
        scores[mask] *= self.filter_factor


class ColbertLoss(ColbertModule):
    """
    InfoNCE loss for late interaction (ColBERT) without explicit negatives.

    Args:
        temperature (float): Scaling factor for logits.
        normalize_scores (bool): Normalize scores by query lengths.
        use_smooth_max (bool): Use log-sum-exp instead of amax.
        pos_aware_negative_filtering (bool): Apply pos-aware negative filtering.
    """

    def __init__(
        self,
        temperature: float = 0.02,
        normalize_scores: bool = True,
        use_smooth_max: bool = False,
        pos_aware_negative_filtering: bool = False,
        max_batch_size: int = 1024,
        tau: float = 0.1,
        norm_tol: float = 1e-3,
        filter_threshold: float = 0.95,
        filter_factor: float = 0.5,
    ):
        super().__init__(max_batch_size, tau, norm_tol, filter_threshold, filter_factor)
        self.temperature = temperature
        self.normalize_scores = normalize_scores
        self.use_smooth_max = use_smooth_max
        self.pos_aware_negative_filtering = pos_aware_negative_filtering
        self.ce_loss = CrossEntropyLoss()

    def forward(self, query_embeddings: torch.Tensor, doc_embeddings: torch.Tensor, offset: int = 0) -> torch.Tensor:
        """
        Compute ColBERT InfoNCE loss over a batch of queries and documents.

        Args:
            query_embeddings (Tensor): [B, Nq, D]
            doc_embeddings (Tensor): [B, Nd, D]
            offset (int): Offset for positive doc indices (multi-GPU).

        Returns:
            Tensor: Scalar loss value.
        """
        lengths = (query_embeddings[:, :, 0] != 0).sum(dim=1)
        raw = torch.einsum("bnd,csd->bcns", query_embeddings, doc_embeddings)
        scores = self._aggregate(raw, self.use_smooth_max, dim_max=3, dim_sum=2)

        if self.normalize_scores:
            try:
                scores = self._apply_normalization(scores, lengths)
            except ValueError as e:
                # Dynamically adjust tolerance and retry normalization.
                original_tol = self.norm_tol
                self.norm_tol = 0.1  # Temporarily loosen to 0.1.
                try:
                    scores = self._apply_normalization(scores, lengths)
                except ValueError:
                    # If still fails, use original scores (no normalization).
                    scores = scores  # Keep as is.
                finally:
                    self.norm_tol = original_tol  # Restore original tolerance.

        batch_size = scores.size(0)
        idx, pos_idx = self._get_idx(batch_size, offset, scores.device)

        if self.pos_aware_negative_filtering:
            self._filter_high_negatives(scores, pos_idx)

        # print(f"Scores shape: {scores.shape}, offset: {offset}")

        return self.ce_loss(scores / self.temperature, pos_idx)


class ColbertNegativeCELoss(ColbertModule):
    """
    InfoNCE loss with explicit negative documents.

    Args:
        temperature (float): Scaling for logits.
        normalize_scores (bool): Normalize scores by query lengths.
        use_smooth_max (bool): Use log-sum-exp instead of amax.
        pos_aware_negative_filtering (bool): Apply pos-aware negative filtering.
        in_batch_term_weight (float): Add in-batch CE term (between 0 and 1).
    """

    def __init__(
        self,
        temperature: float = 0.02,
        normalize_scores: bool = True,
        use_smooth_max: bool = False,
        pos_aware_negative_filtering: bool = False,
        in_batch_term_weight: float = 0.5,
        max_batch_size: int = 1024,
        tau: float = 0.1,
        norm_tol: float = 1e-3,
        filter_threshold: float = 0.95,
        filter_factor: float = 0.5,
    ):
        super().__init__(max_batch_size, tau, norm_tol, filter_threshold, filter_factor)
        self.temperature = temperature
        self.normalize_scores = normalize_scores
        self.use_smooth_max = use_smooth_max
        self.pos_aware_negative_filtering = pos_aware_negative_filtering
        self.in_batch_term_weight = in_batch_term_weight
        self.ce_loss = CrossEntropyLoss()

        assert in_batch_term_weight >= 0, "in_batch_term_weight must be non-negative"
        assert in_batch_term_weight <= 1, "in_batch_term_weight must be less than 1"

        self.inner_loss = ColbertLoss(
            temperature=temperature,
            normalize_scores=normalize_scores,
            use_smooth_max=use_smooth_max,
            pos_aware_negative_filtering=pos_aware_negative_filtering,
            max_batch_size=max_batch_size,
            tau=tau,
            norm_tol=norm_tol,
            filter_threshold=filter_threshold,
            filter_factor=filter_factor,
        )

    def forward(
        self,
        query_embeddings: torch.Tensor,
        doc_embeddings: torch.Tensor,
        neg_doc_embeddings: torch.Tensor,
        offset: int = 0,
    ) -> torch.Tensor:
        """
        Compute InfoNCE loss with explicit negatives and optional in-batch term.

        Args:
            query_embeddings (Tensor): [B, Nq, D]
            doc_embeddings (Tensor): [B, Nd, D] positive docs
            neg_doc_embeddings (Tensor): [B, Nneg, D] negative docs
            offset (int): Positional offset for in-batch CE.

        Returns:
            Tensor: Scalar loss.
        """
        lengths = (query_embeddings[:, :, 0] != 0).sum(dim=1)
        pos_raw = torch.einsum("bnd,bsd->bns", query_embeddings, doc_embeddings)
        neg_raw = torch.einsum("bnd,bsd->bns", query_embeddings, neg_doc_embeddings)
        pos_scores = self._aggregate(pos_raw, self.use_smooth_max, dim_max=2, dim_sum=1)
        neg_scores = self._aggregate(neg_raw, self.use_smooth_max, dim_max=2, dim_sum=1)

        if self.normalize_scores:
            pos_scores = self._apply_normalization(pos_scores, lengths)
            neg_scores = self._apply_normalization(neg_scores, lengths)

        loss = F.softplus((neg_scores - pos_scores) / self.temperature).mean()

        if self.in_batch_term_weight > 0:
            loss_ib = self.inner_loss(query_embeddings, doc_embeddings, offset)
            loss = loss * (1 - self.in_batch_term_weight) + loss_ib * self.in_batch_term_weight

        return loss


class ColbertPairwiseCELoss(ColbertModule):
    """
    Pairwise loss for ColBERT (no explicit negatives).

    Args:
        temperature (float): Scaling for logits.
        normalize_scores (bool): Normalize scores by query lengths.
        use_smooth_max (bool): Use log-sum-exp instead of amax.
        pos_aware_negative_filtering (bool): Apply pos-aware negative filtering.
    """

    def __init__(
        self,
        temperature: float = 1.0,
        normalize_scores: bool = True,
        use_smooth_max: bool = False,
        pos_aware_negative_filtering: bool = False,
        max_batch_size: int = 1024,
        tau: float = 0.1,
        norm_tol: float = 1e-3,
        filter_threshold: float = 0.95,
        filter_factor: float = 0.5,
    ):
        super().__init__(max_batch_size, tau, norm_tol, filter_threshold, filter_factor)
        self.temperature = temperature
        self.normalize_scores = normalize_scores
        self.use_smooth_max = use_smooth_max
        self.pos_aware_negative_filtering = pos_aware_negative_filtering

    def forward(self, query_embeddings: torch.Tensor, doc_embeddings: torch.Tensor, offset: int = 0) -> torch.Tensor:
        """
        Compute pairwise softplus loss over in-batch document pairs.

        Args:
            query_embeddings (Tensor): [B, Nq, D]
            doc_embeddings (Tensor): [B, Nd, D]
            offset (int): Positional offset for positives.

        Returns:
            Tensor: Scalar loss value.
        """
        lengths = (query_embeddings[:, :, 0] != 0).sum(dim=1)
        raw = torch.einsum("bnd,csd->bcns", query_embeddings, doc_embeddings)
        scores = self._aggregate(raw, self.use_smooth_max, dim_max=3, dim_sum=2)

        if self.normalize_scores:
            scores = self._apply_normalization(scores, lengths)

        batch_size = scores.size(0)
        idx, pos_idx = self._get_idx(batch_size, offset, scores.device)

        if self.pos_aware_negative_filtering:
            self._filter_high_negatives(scores, pos_idx)

        pos_scores = scores.diagonal(offset=offset)
        top2 = scores.topk(2, dim=1).values
        neg_scores = torch.where(top2[:, 0] == pos_scores, top2[:, 1], top2[:, 0])

        return F.softplus((neg_scores - pos_scores) / self.temperature).mean()


class ColbertPairwiseNegativeCELoss(ColbertModule):
    """
    Pairwise loss with explicit negatives and optional in-batch term.

    Args:
        temperature (float): Scaling for logits.
        normalize_scores (bool): Normalize scores by query lengths.
        use_smooth_max (bool): Use log-sum-exp instead of amax.
        pos_aware_negative_filtering (bool): Apply pos-aware negative filtering.
        in_batch_term_weight (float): Add in-batch CE term (between 0 and 1).
    """

    def __init__(
        self,
        temperature: float = 0.02,
        normalize_scores: bool = True,
        use_smooth_max: bool = False,
        pos_aware_negative_filtering: bool = False,
        in_batch_term_weight: float = 0.5,
        max_batch_size: int = 1024,
        tau: float = 0.1,
        norm_tol: float = 1e-3,
        filter_threshold: float = 0.95,
        filter_factor: float = 0.5,
    ):
        super().__init__(max_batch_size, tau, norm_tol, filter_threshold, filter_factor)
        self.temperature = temperature
        self.normalize_scores = normalize_scores
        self.use_smooth_max = use_smooth_max
        self.pos_aware_negative_filtering = pos_aware_negative_filtering
        self.in_batch_term_weight = in_batch_term_weight
        assert in_batch_term_weight >= 0, "in_batch_term_weight must be non-negative"
        assert in_batch_term_weight <= 1, "in_batch_term_weight must be less than 1"
        self.inner_pairwise = ColbertPairwiseCELoss(
            temperature=temperature,
            normalize_scores=normalize_scores,
            use_smooth_max=use_smooth_max,
            pos_aware_negative_filtering=pos_aware_negative_filtering,
            max_batch_size=max_batch_size,
            tau=tau,
            norm_tol=norm_tol,
            filter_threshold=filter_threshold,
            filter_factor=filter_factor,
        )

    def forward(
        self,
        query_embeddings: torch.Tensor,
        doc_embeddings: torch.Tensor,
        neg_doc_embeddings: torch.Tensor,
        offset: int = 0,
    ) -> torch.Tensor:
        """
        Compute pairwise softplus loss with explicit negatives and optional in-batch term.

        Args:
            query_embeddings (Tensor): [B, Nq, D]
            doc_embeddings (Tensor): [B, Nd, D] positive docs
            neg_doc_embeddings (Tensor): [B, Nneg, D] negative docs
            offset (int): Positional offset for positives.

        Returns:
            Tensor: Scalar loss value.
        """
        lengths = (query_embeddings[:, :, 0] != 0).sum(dim=1)
        pos_raw = torch.einsum("bnd,bsd->bns", query_embeddings, doc_embeddings)
        neg_raw = torch.einsum("bnd,bsd->bns", query_embeddings, neg_doc_embeddings)
        pos_scores = self._aggregate(pos_raw, self.use_smooth_max, dim_max=2, dim_sum=1)
        neg_scores = self._aggregate(neg_raw, self.use_smooth_max, dim_max=2, dim_sum=1)

        if self.normalize_scores:
            pos_scores = self._apply_normalization(pos_scores, lengths)
            neg_scores = self._apply_normalization(neg_scores, lengths)

        loss = F.softplus((neg_scores - pos_scores) / self.temperature).mean()

        if self.in_batch_term_weight > 0:
            loss_ib = self.inner_pairwise(query_embeddings, doc_embeddings, offset)
            loss = loss * (1 - self.in_batch_term_weight) + loss_ib * self.in_batch_term_weight

        return loss


class MultiPositiveInfoNCELoss(ColbertModule):
    """
    Multi-Positive InfoNCE loss function.
    
    Numerator: Sum of similarities between query and all positive examples.
    Denominator: Sum of similarities between query and all positive examples + negative examples.
    """

    def __init__(
        self,
        temperature: float = 0.02,
        normalize_scores: bool = True,
        use_smooth_max: bool = False,
        max_batch_size: int = 1024,
        tau: float = 0.1,
        norm_tol: float = 1e-3,
    ):
        super().__init__(max_batch_size, tau, norm_tol)
        self.temperature = temperature
        self.normalize_scores = normalize_scores
        self.use_smooth_max = use_smooth_max

    def forward(
        self, 
        query_embeddings: torch.Tensor, 
        doc_embeddings: torch.Tensor, 
        positive_mask: torch.Tensor,
        offset: int = 0
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            query_embeddings: [B_query, seq_len, dim]
            doc_embeddings: [B_doc, seq_len, dim] 
            positive_mask: [B_query, B_doc] 布尔掩码，True表示正例
            offset: 多GPU训练偏移量
            
        Returns:
            loss: 标量损失
            similarity_scores: [B_query, B_doc] 相似度矩阵，供后续使用
        """
        # Calculate query lengths.
        lengths = (query_embeddings[:, :, 0] != 0).sum(dim=1)
        
        # Calculate token-level similarity [B_query, B_doc, query_len, doc_len].
        raw_scores = torch.einsum("bnd,csd->bcns", query_embeddings, doc_embeddings)
        
        # Aggregate to document-level similarity [B_query, B_doc].
        similarity_scores = self._aggregate(raw_scores, self.use_smooth_max, dim_max=3, dim_sum=2)
        
        # Normalization.
        if self.normalize_scores:
            try:
                similarity_scores = self._apply_normalization(similarity_scores, lengths)
            except ValueError:
                # Dynamically adjust tolerance.
                original_tol = self.norm_tol
                self.norm_tol = 0.1
                try:
                    similarity_scores = self._apply_normalization(similarity_scores, lengths)
                except ValueError:
                    similarity_scores = similarity_scores
                finally:
                    self.norm_tol = original_tol
        
        # Apply temperature scaling.
        scaled_scores = similarity_scores / self.temperature
        
        batch_size_query = query_embeddings.size(0)
        total_loss = 0.0
        
        for i in range(batch_size_query):
            query_scores = scaled_scores[i]  # [B_doc]
            pos_mask = positive_mask[i]      # [B_doc]
            
            if not pos_mask.any():
                continue
                
            # Multi-Positive InfoNCE calculation.
            # Numerator: Sum of similarities between query and all positive examples (log domain).
            pos_scores = query_scores[pos_mask]
            numerator = torch.logsumexp(pos_scores, dim=0)
            
            # Denominator: Logsumexp of similarities between query and all documents.
            denominator = torch.logsumexp(query_scores, dim=0)
            
            # Negative log-likelihood.
            query_loss = denominator - numerator
            total_loss += query_loss
        
        return total_loss / batch_size_query, similarity_scores


class TopKListwiseLoss(torch.nn.Module):
    """
    Top-K Listwise loss, based on NDCG optimization.
    """
    
    def __init__(
        self,
        k: int = 10,
        temperature: float = 1.0,
        loss_type: str = "approx_ndcg",  # "approx_ndcg", "listmle"
    ):
        super().__init__()
        self.k = k
        self.temperature = temperature
        self.loss_type = loss_type
    
    def forward(
        self,
        similarity_scores: torch.Tensor,
        positive_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            similarity_scores: [B_query, B_doc] similarity scores
            positive_mask: [B_query, B_doc] positive mask
            
        Returns:
            loss: Scalar loss.
        """
        batch_size_query = similarity_scores.size(0)
        total_loss = 0.0
        
        for i in range(batch_size_query):
            query_scores = similarity_scores[i]  # [B_doc]
            pos_mask = positive_mask[i]         # [B_doc]
            
            if not pos_mask.any():
                continue
            
            # 选择Top-K候选
            top_k_values, top_k_indices = torch.topk(query_scores, min(self.k, len(query_scores)))
            
            # 获取Top-K中的相关性标签
            relevance_labels = pos_mask[top_k_indices].float()
            
            if self.loss_type == "approx_ndcg":
                query_loss = self._compute_approx_ndcg_loss(top_k_values, relevance_labels)
            elif self.loss_type == "listmle":
                query_loss = self._compute_listmle_loss(top_k_values, relevance_labels)
            else:
                raise ValueError(f"Unknown loss_type: {self.loss_type}")
            
            total_loss += query_loss
        
        return total_loss / batch_size_query
    
    def _compute_approx_ndcg_loss(self, scores: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """计算近似NDCG损失"""
        ideal_labels, _ = torch.sort(labels, descending=True)
        ideal_dcg = self._compute_dcg(ideal_labels)
        
        if ideal_dcg == 0:
            return torch.tensor(0.0, device=scores.device)
        
        actual_dcg = self._compute_dcg(labels)
        
        ndcg = actual_dcg / ideal_dcg
        return 1.0 - ndcg
    
    def _compute_dcg(self, labels: torch.Tensor) -> torch.Tensor:
        """计算DCG"""
        positions = torch.arange(1, len(labels) + 1, device=labels.device, dtype=torch.float)
        discounts = torch.log2(positions + 1)
        dcg = torch.sum(labels / discounts)
        return dcg
    
    def _compute_listmle_loss(self, scores: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Compute ListMLE loss."""
        relevant_indices = labels.bool()
        if not relevant_indices.any():
            return torch.tensor(0.0, device=scores.device)
        
        relevant_scores = scores[relevant_indices]
        scaled_scores = relevant_scores / self.temperature
        
        loss = 0.0
        for i in range(len(relevant_scores)):
            prob = F.softmax(scaled_scores[i:], dim=0)[0]
            loss += -torch.log(prob + 1e-8)
        
        return loss


class CombinedMultiPositiveLoss(torch.nn.Module):
    """
    Combined loss function: Multi-Positive InfoNCE + Top-K Listwise.
    """
    
    def __init__(
        self,
        infonce_weight: float = 1.0,
        listwise_weight: float = 0.1,
        listwise_freq: int = 5,  # Compute listwise loss every N steps.
        temperature: float = 0.02,
        normalize_scores: bool = True,
        use_smooth_max: bool = False,
        k: int = 10,
        listwise_loss_type: str = "approx_ndcg",
        max_batch_size: int = 1024,
        tau: float = 0.1,
        norm_tol: float = 1e-3,
    ):
        super().__init__()
        
        self.infonce_loss = MultiPositiveInfoNCELoss(
            temperature=temperature,
            normalize_scores=normalize_scores,
            use_smooth_max=use_smooth_max,
            max_batch_size=max_batch_size,
            tau=tau,
            norm_tol=norm_tol,
        )
        
        self.listwise_loss = TopKListwiseLoss(
            k=k,
            temperature=1.0, 
            loss_type=listwise_loss_type,
        )
        
        self.infonce_weight = infonce_weight
        self.listwise_weight = listwise_weight
        self.listwise_freq = listwise_freq
        self.step_count = 0
    
    def forward(
        self, 
        query_embeddings: torch.Tensor, 
        doc_embeddings: torch.Tensor, 
        positive_mask: torch.Tensor,
        offset: int = 0
    ) -> torch.Tensor:
        """
        Args:
            query_embeddings: [B_query, seq_len, dim]
            doc_embeddings: [B_doc, seq_len, dim]
            positive_mask: [B_query, B_doc] 正例掩码
            offset: 多GPU训练偏移量
            
        Returns:
            combined_loss: 标量损失
        """
        # 计算主损失
        infonce_loss, similarity_scores = self.infonce_loss(
            query_embeddings, doc_embeddings, positive_mask, offset
        )
        
        # 按频率计算辅助损失
        if self.step_count % self.listwise_freq == 0 and self.step_count > 200:
            listwise_loss = self.listwise_loss(similarity_scores.detach(), positive_mask)
            combined_loss = (
                self.infonce_weight * infonce_loss + 
                self.listwise_weight * listwise_loss
            )
        else:
            combined_loss = self.infonce_weight * infonce_loss
        
        self.step_count += 1
        return combined_loss
