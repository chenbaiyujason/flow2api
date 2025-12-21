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

        # 'Fill First' strategy: select token with LEAST remaining concurrency (but > 0)
        # This prioritizes filling up one token to batch requests together
        # When remaining is equal, prefer smaller token ID for stable selection
        # IMPORTANT: Atomically acquire concurrency slot during selection to prevent race conditions
        if self.concurrency_manager:
            best_token = None
            min_remaining = float('inf')  # Start with infinity, find minimum
            
            for token in available_tokens:
                if for_image_generation:
                    remaining = await self.concurrency_manager.get_image_remaining(token.id)
                elif for_video_generation:
                    remaining = await self.concurrency_manager.get_video_remaining(token.id)
                else:
                    # 默认使用图片并发
                    remaining = await self.concurrency_manager.get_image_remaining(token.id)
                
                # None means no limit, treat as infinite (lowest priority for fill-first)
                if remaining is None:
                    debug_logger.log_info(f"[LOAD_BALANCER] Token {token.id} 没有并发限制，视为无限")
                    remaining = float('inf')
                elif remaining <= 0:
                    debug_logger.log_info(f"[LOAD_BALANCER] Token {token.id} 并发已满 (剩余: {remaining})，跳过")
                    continue  # Skip tokens with no remaining slots
                
                # Select if LESS remaining (to batch together), or same remaining but smaller ID
                if remaining < min_remaining or (remaining == min_remaining and (best_token is None or token.id < best_token.id)):
                    min_remaining = remaining
                    best_token = token

            
            if best_token:
                # ⚡ 关键修复: 原子性地获取并发槽位
                # 如果获取失败，递归重新选择 (排除当前 token)
                if for_image_generation:
                    acquired = await self.concurrency_manager.acquire_image(best_token.id)
                elif for_video_generation:
                    acquired = await self.concurrency_manager.acquire_video(best_token.id)
                else:
                    acquired = await self.concurrency_manager.acquire_image(best_token.id)
                
                if acquired:
                    debug_logger.log_info(
                        f"[LOAD_BALANCER] ✅ 已选择Token {best_token.id} ({best_token.email}) - "
                        f"余额: {best_token.credits}, 剩余并发: {min_remaining if min_remaining != float('inf') else '无限制'}"
                    )
                    return best_token
                else:
                    # 获取失败（其他请求抢先获取了），从列表中移除后重试
                    debug_logger.log_info(f"[LOAD_BALANCER] Token {best_token.id} 并发获取失败，重新选择")
                    available_tokens = [t for t in available_tokens if t.id != best_token.id]
                    if available_tokens:
                        # 递归重试
                        return await self.select_token(for_image_generation, for_video_generation, model)
                    else:
                        debug_logger.log_info(f"[LOAD_BALANCER] ❌ 所有Token并发已满")
                        return None
            else:
                debug_logger.log_info(f"[LOAD_BALANCER] ❌ 所有Token并发已满")
                return None

        
        # Fallback to random if no concurrency manager
        selected = random.choice(available_tokens)
        debug_logger.log_info(f"[LOAD_BALANCER] ✅ 已选择Token {selected.id} ({selected.email}) - 余额: {selected.credits}")
        return selected

