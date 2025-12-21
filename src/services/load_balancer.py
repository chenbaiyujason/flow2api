"""Load balancing module for Flow2API"""
import random
from typing import Optional
from ..core.models import Token
from .concurrency_manager import ConcurrencyManager
from ..core.logger import debug_logger


class LoadBalancer:
    """Token load balancer with 'fill first' strategy for better batching"""

    def __init__(self, token_manager, concurrency_manager: Optional[ConcurrencyManager] = None):
        self.token_manager = token_manager
        self.concurrency_manager = concurrency_manager

    async def select_token(
        self,
        for_image_generation: bool = False,
        for_video_generation: bool = False,
        model: Optional[str] = None
    ) -> Optional[Token]:
        """
        Select a token using 'fill first' strategy - prioritize tokens with most remaining concurrency.
        This helps batch requests to the same token for better API call efficiency.

        Args:
            for_image_generation: If True, only select tokens with image_enabled=True
            for_video_generation: If True, only select tokens with video_enabled=True
            model: Model name (used to filter tokens for specific models)

        Returns:
            Selected token or None if no available tokens
        """
        debug_logger.log_info(f"[LOAD_BALANCER] 开始选择Token (图片生成={for_image_generation}, 视频生成={for_video_generation}, 模型={model})")

        active_tokens = await self.token_manager.get_active_tokens()
        debug_logger.log_info(f"[LOAD_BALANCER] 获取到 {len(active_tokens)} 个活跃Token")

        if not active_tokens:
            debug_logger.log_info(f"[LOAD_BALANCER] ❌ 没有活跃的Token")
            return None

        # Filter tokens based on generation type
        available_tokens = []
        filtered_reasons = {}  # 记录过滤原因

        for token in active_tokens:
            # Check if token has valid AT (not expired)
            if not await self.token_manager.is_at_valid(token.id):
                filtered_reasons[token.id] = "AT无效或已过期"
                continue

            # Filter for gemini-3.0 models (skip free tier tokens)
            if model and model in ["gemini-3.0-pro-image-landscape", "gemini-3.0-pro-image-portrait"]:
                if token.user_paygate_tier == "PAYGATE_TIER_NOT_PAID":
                    filtered_reasons[token.id] = "gemini-3.0模型不支持普通账号"
                    continue

            # Filter for image generation
            if for_image_generation:
                if not token.image_enabled:
                    filtered_reasons[token.id] = "图片生成已禁用"
                    continue

                # Check concurrency limit
                if self.concurrency_manager and not await self.concurrency_manager.can_use_image(token.id):
                    filtered_reasons[token.id] = "图片并发已满"
                    continue

            # Filter for video generation
            if for_video_generation:
                if not token.video_enabled:
                    filtered_reasons[token.id] = "视频生成已禁用"
                    continue

                # Check concurrency limit
                if self.concurrency_manager and not await self.concurrency_manager.can_use_video(token.id):
                    filtered_reasons[token.id] = "视频并发已满"
                    continue

            available_tokens.append(token)

        # 输出过滤信息
        if filtered_reasons:
            debug_logger.log_info(f"[LOAD_BALANCER] 已过滤Token:")
            for token_id, reason in filtered_reasons.items():
                debug_logger.log_info(f"[LOAD_BALANCER]   - Token {token_id}: {reason}")

        if not available_tokens:
            debug_logger.log_info(f"[LOAD_BALANCER] ❌ 没有可用的Token (图片生成={for_image_generation}, 视频生成={for_video_generation})")
            return None

        # 'Fill First' strategy: select token with most remaining concurrency
        # This helps batch requests to the same token
        # When remaining is equal, prefer smaller token ID for stable selection
        if self.concurrency_manager:
            best_token = None
            max_remaining = -1
            
            for token in available_tokens:
                if for_image_generation:
                    remaining = await self.concurrency_manager.get_image_remaining(token.id)
                elif for_video_generation:
                    remaining = await self.concurrency_manager.get_video_remaining(token.id)
                else:
                    remaining = None
                
                # None means no limit, treat as infinite
                if remaining is None:
                    remaining = float('inf')
                
                # Select if more remaining, or same remaining but smaller ID (stable selection)
                if remaining > max_remaining or (remaining == max_remaining and (best_token is None or token.id < best_token.id)):
                    max_remaining = remaining
                    best_token = token
            
            if best_token:
                debug_logger.log_info(
                    f"[LOAD_BALANCER] ✅ 已选择Token {best_token.id} ({best_token.email}) - "
                    f"余额: {best_token.credits}, 剩余并发: {max_remaining if max_remaining != float('inf') else '无限制'}"
                )
                return best_token

        
        # Fallback to random if no concurrency manager
        selected = random.choice(available_tokens)
        debug_logger.log_info(f"[LOAD_BALANCER] ✅ 已选择Token {selected.id} ({selected.email}) - 余额: {selected.credits}")
        return selected

