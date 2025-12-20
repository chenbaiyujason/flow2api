"""Batch Request Manager for merging multiple requests into batches"""
import asyncio
import uuid
import time
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple, Callable
from ..core.logger import debug_logger
from ..core.config import config


@dataclass
class PendingRequest:
    """单个待处理的请求"""
    request_id: str                    # 内部追踪ID
    scene_id: str                      # 用于匹配结果的 sceneId
    request_data: Dict                 # 请求数据 (不含clientContext)
    future: Optional[asyncio.Future] = None  # 在创建时设置
    created_at: float = field(default_factory=time.time)



@dataclass
class PendingBatch:
    """待处理的批次"""
    batch_id: str                      # 批次ID
    token_id: int                      # Token ID
    at_token: str                      # Access Token
    project_id: str                    # Project ID
    endpoint: str                      # API端点路径
    user_paygate_tier: str             # 用户等级
    requests: List[PendingRequest] = field(default_factory=list)
    timer_task: Optional[asyncio.Task] = None
    recaptcha_token: Optional[str] = None   # 共享的 recaptcha token
    session_id: Optional[str] = None        # 共享的 session id


class BatchRequestManager:
    """
    批量请求管理器
    
    功能:
    - 按 (token_id, project_id, endpoint) 分组收集请求
    - 最多4个请求合并为一个批次 (可通过配置调整)
    - 共享同一个 recaptcha_token
    - 收集窗口时间可配置 (默认300ms)
    - 视频异步轮询，图片同步返回
    """
    
    # 端点常量
    ENDPOINT_IMAGE = "batchGenerateImages"
    ENDPOINT_VIDEO_TEXT = "batchAsyncGenerateVideoText"
    ENDPOINT_VIDEO_START_IMAGE = "batchAsyncGenerateVideoStartImage"
    ENDPOINT_VIDEO_START_END = "batchAsyncGenerateVideoStartAndEndImage"
    ENDPOINT_VIDEO_REFERENCE = "batchAsyncGenerateVideoReferenceImages"
    
    @property
    def MAX_BATCH_SIZE(self) -> int:
        """从配置获取最大批次大小"""
        return config.batch_max_size
    
    @property
    def COLLECT_WINDOW_MS(self) -> int:
        """从配置获取收集窗口时间(毫秒)"""
        return config.batch_collect_window_ms
    
    def __init__(self, flow_client):
        """
        Args:
            flow_client: FlowClient 实例
        """
        self.flow_client = flow_client
        self._pending_batches: Dict[str, PendingBatch] = {}  # batch_key -> PendingBatch
        self._lock = asyncio.Lock()
    
    def _get_batch_key(self, token_id: int, project_id: str, endpoint: str) -> str:
        """生成批次分组键"""
        return f"{token_id}:{project_id}:{endpoint}"
    
    async def submit_video_request(
        self,
        token_id: int,
        at_token: str,
        project_id: str,
        endpoint: str,
        request_data: Dict,
        user_paygate_tier: str = "PAYGATE_TIER_ONE"
    ) -> asyncio.Future:
        """
        提交视频生成请求
        
        Args:
            token_id: Token ID
            at_token: Access Token
            project_id: 项目ID
            endpoint: API端点 (如 batchAsyncGenerateVideoText)
            request_data: 请求数据 (不含 clientContext, 包含 aspectRatio, seed, textInput 等)
            user_paygate_tier: 用户等级
            
        Returns:
            Future, 完成时包含该请求的操作结果
        """
        batch_key = self._get_batch_key(token_id, project_id, endpoint)
        scene_id = request_data.get("metadata", {}).get("sceneId") or str(uuid.uuid4())
        
        # 确保 request_data 有 metadata.sceneId
        if "metadata" not in request_data:
            request_data["metadata"] = {}
        request_data["metadata"]["sceneId"] = scene_id
        
        # 创建待处理请求
        loop = asyncio.get_running_loop()
        pending_request = PendingRequest(
            request_id=str(uuid.uuid4()),
            scene_id=scene_id,
            request_data=request_data,
            future=loop.create_future()
        )
        
        async with self._lock:
            # 检查是否有现有批次
            if batch_key not in self._pending_batches:
                # 创建新批次
                batch = PendingBatch(
                    batch_id=str(uuid.uuid4()),
                    token_id=token_id,
                    at_token=at_token,
                    project_id=project_id,
                    endpoint=endpoint,
                    user_paygate_tier=user_paygate_tier
                )
                self._pending_batches[batch_key] = batch
                debug_logger.log_info(f"[BATCH] 创建新批次: {batch_key}")
            
            batch = self._pending_batches[batch_key]
            batch.requests.append(pending_request)
            
            debug_logger.log_info(
                f"[BATCH] 请求加入批次: {batch_key}, "
                f"当前请求数: {len(batch.requests)}/{self.MAX_BATCH_SIZE}"
            )
            
            # 检查是否达到最大批次大小
            if len(batch.requests) >= self.MAX_BATCH_SIZE:
                # 取消计时器并立即发送
                if batch.timer_task and not batch.timer_task.done():
                    batch.timer_task.cancel()
                # 从 pending 中移除
                del self._pending_batches[batch_key]
                # 异步发送批次
                asyncio.create_task(self._send_video_batch(batch))
            else:
                # 启动或重置计时器
                if batch.timer_task is None or batch.timer_task.done():
                    batch.timer_task = asyncio.create_task(
                        self._batch_timer(batch_key, self.COLLECT_WINDOW_MS / 1000)
                    )
        
        return pending_request.future
    
    async def submit_image_request(
        self,
        token_id: int,
        at_token: str,
        project_id: str,
        request_data: Dict,
        user_paygate_tier: str = "PAYGATE_TIER_ONE"
    ) -> asyncio.Future:
        """
        提交图片生成请求
        
        Args:
            token_id: Token ID
            at_token: Access Token
            project_id: 项目ID
            request_data: 单个请求数据
            user_paygate_tier: 用户等级
            
        Returns:
            Future, 完成时包含该请求的生成结果
        """
        endpoint = self.ENDPOINT_IMAGE
        batch_key = self._get_batch_key(token_id, project_id, endpoint)
        
        # 图片请求使用索引作为匹配键
        request_index = str(uuid.uuid4())
        
        loop = asyncio.get_running_loop()
        pending_request = PendingRequest(
            request_id=request_index,
            scene_id=request_index,  # 图片用索引匹配
            request_data=request_data,
            future=loop.create_future()
        )
        
        async with self._lock:
            if batch_key not in self._pending_batches:
                batch = PendingBatch(
                    batch_id=str(uuid.uuid4()),
                    token_id=token_id,
                    at_token=at_token,
                    project_id=project_id,
                    endpoint=endpoint,
                    user_paygate_tier=user_paygate_tier
                )
                self._pending_batches[batch_key] = batch
                debug_logger.log_info(f"[BATCH] 创建新图片批次: {batch_key}")
            
            batch = self._pending_batches[batch_key]
            batch.requests.append(pending_request)
            
            debug_logger.log_info(
                f"[BATCH] 图片请求加入批次: {batch_key}, "
                f"当前请求数: {len(batch.requests)}/{self.MAX_BATCH_SIZE}"
            )
            
            if len(batch.requests) >= self.MAX_BATCH_SIZE:
                if batch.timer_task and not batch.timer_task.done():
                    batch.timer_task.cancel()
                del self._pending_batches[batch_key]
                asyncio.create_task(self._send_image_batch(batch))
            else:
                if batch.timer_task is None or batch.timer_task.done():
                    batch.timer_task = asyncio.create_task(
                        self._batch_timer_image(batch_key, self.COLLECT_WINDOW_MS / 1000)
                    )
        
        return pending_request.future
    
    async def _batch_timer(self, batch_key: str, delay: float):
        """批次计时器 (视频)"""
        try:
            await asyncio.sleep(delay)
            async with self._lock:
                if batch_key in self._pending_batches:
                    batch = self._pending_batches.pop(batch_key)
                    debug_logger.log_info(
                        f"[BATCH] 计时器触发，发送批次: {batch_key}, "
                        f"请求数: {len(batch.requests)}"
                    )
                    asyncio.create_task(self._send_video_batch(batch))
        except asyncio.CancelledError:
            pass
    
    async def _batch_timer_image(self, batch_key: str, delay: float):
        """批次计时器 (图片)"""
        try:
            await asyncio.sleep(delay)
            async with self._lock:
                if batch_key in self._pending_batches:
                    batch = self._pending_batches.pop(batch_key)
                    debug_logger.log_info(
                        f"[BATCH] 图片计时器触发，发送批次: {batch_key}, "
                        f"请求数: {len(batch.requests)}"
                    )
                    asyncio.create_task(self._send_image_batch(batch))
        except asyncio.CancelledError:
            pass
    
    async def _send_video_batch(self, batch: PendingBatch):
        """发送视频批量请求"""
        try:
            # 获取共享的 recaptcha_token
            recaptcha_token = await self.flow_client._get_recaptcha_token(batch.project_id) or ""
            session_id = self.flow_client._generate_session_id()
            
            debug_logger.log_info(
                f"[BATCH] 发送视频批次: {batch.batch_id}, "
                f"端点: {batch.endpoint}, 请求数: {len(batch.requests)}"
            )
            
            # 构建批量请求
            url = f"{self.flow_client.api_base_url}/video:{batch.endpoint}"
            
            json_data = {
                "clientContext": {
                    "recaptchaToken": recaptcha_token,
                    "sessionId": session_id,
                    "projectId": batch.project_id,
                    "tool": "PINHOLE",
                    "userPaygateTier": batch.user_paygate_tier
                },
                "requests": [req.request_data for req in batch.requests]
            }
            
            # 发送请求
            result = await self.flow_client._make_request(
                method="POST",
                url=url,
                json_data=json_data,
                use_at=True,
                at_token=batch.at_token
            )
            
            # 解析结果并分发给各个请求
            operations = result.get("operations", [])
            
            # 建立 sceneId -> operation 的映射
            scene_to_op = {op.get("sceneId"): op for op in operations}
            
            # 分发给各个请求的 future
            for req in batch.requests:
                if req.scene_id in scene_to_op:
                    req.future.set_result(scene_to_op[req.scene_id])
                else:
                    req.future.set_exception(
                        Exception(f"未找到 sceneId={req.scene_id} 的操作结果")
                    )
            
            debug_logger.log_info(
                f"[BATCH] 视频批次发送成功: {batch.batch_id}, "
                f"返回操作数: {len(operations)}"
            )
            
        except Exception as e:
            debug_logger.log_error(f"[BATCH] 视频批次发送失败: {str(e)}")
            # 所有请求都设置异常
            for req in batch.requests:
                if not req.future.done():
                    req.future.set_exception(e)
    
    async def _send_image_batch(self, batch: PendingBatch):
        """发送图片批量请求"""
        try:
            # 获取共享的 recaptcha_token
            recaptcha_token = await self.flow_client._get_recaptcha_token(batch.project_id) or ""
            session_id = self.flow_client._generate_session_id()
            
            debug_logger.log_info(
                f"[BATCH] 发送图片批次: {batch.batch_id}, "
                f"请求数: {len(batch.requests)}"
            )
            
            # 构建批量请求
            url = f"{self.flow_client.api_base_url}/projects/{batch.project_id}/flowMedia:batchGenerateImages"
            
            # 为每个请求添加 clientContext
            requests_data = []
            for req in batch.requests:
                req_data = req.request_data.copy()
                req_data["clientContext"] = {
                    "recaptchaToken": recaptcha_token,
                    "projectId": batch.project_id,
                    "sessionId": session_id,
                    "tool": "PINHOLE"
                }
                requests_data.append(req_data)
            
            json_data = {
                "clientContext": {
                    "recaptchaToken": recaptcha_token,
                    "sessionId": session_id
                },
                "requests": requests_data
            }
            
            # 发送请求
            result = await self.flow_client._make_request(
                method="POST",
                url=url,
                json_data=json_data,
                use_at=True,
                at_token=batch.at_token
            )
            
            # 图片是同步返回，按顺序分发结果
            media_list = result.get("media", [])
            
            for idx, req in enumerate(batch.requests):
                if idx < len(media_list):
                    req.future.set_result(media_list[idx])
                else:
                    req.future.set_exception(
                        Exception(f"图片结果数量不足，索引 {idx} 无结果")
                    )
            
            debug_logger.log_info(
                f"[BATCH] 图片批次发送成功: {batch.batch_id}, "
                f"返回媒体数: {len(media_list)}"
            )
            
        except Exception as e:
            debug_logger.log_error(f"[BATCH] 图片批次发送失败: {str(e)}")
            for req in batch.requests:
                if not req.future.done():
                    req.future.set_exception(e)
