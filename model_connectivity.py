#!/usr/bin/env python
"""
模型连通性测试脚本

逐一测试 .env 中配置的 LLM、VLM、Embedding、Rerank 模型
是否能成功建立连接并返回有效响应。

用法:
    python test_model_connectivity.py
    python test_model_connectivity.py --verbose   # 显示详细响应
"""

import argparse
import asyncio
import os
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# 确保优先使用本项目代码（而非已安装的 site-packages）
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

# ---------------------------------------------------------------------------
# 加载 .env
# ---------------------------------------------------------------------------
from dotenv import load_dotenv

load_dotenv(dotenv_path=PROJECT_ROOT / ".env", override=False)

from lightrag.llm.openai import openai_complete_if_cache, openai_embed

# ---------------------------------------------------------------------------
# 颜色输出（Windows GBK 兼容）
# ---------------------------------------------------------------------------
_GREEN = "\033[92m"
_RED = "\033[91m"
_YELLOW = "\033[93m"
_CYAN = "\033[96m"
_BOLD = "\033[1m"
_RESET = "\033[0m"
_PASS = "[PASS]"
_FAIL = "[FAIL]"
_WARN = "[WARN]"


def ok(msg: str):
    print(f"  {_GREEN}{_PASS}{_RESET} {msg}")


def fail(msg: str):
    print(f"  {_RED}{_FAIL}{_RESET} {msg}")


def warn(msg: str):
    print(f"  {_YELLOW}{_WARN}{_RESET} {msg}")


def header(title: str):
    print(f"\n{_BOLD}{_CYAN}{'=' * 60}{_RESET}")
    print(f"{_BOLD}{_CYAN}{title}{_RESET}")
    print(f"{_BOLD}{_CYAN}{'=' * 60}{_RESET}")


# ===================================================================
# 测试函数
# ===================================================================


async def test_llm(base_url: str, api_key: str, model: str, verbose: bool) -> bool:
    """测试 LLM 文本生成"""
    msg = "Hello, please respond with exactly one word: OK"
    try:
        t0 = time.time()
        resp = await openai_complete_if_cache(
            model=model,
            prompt=msg,
            api_key=api_key,
            base_url=base_url,
            max_tokens=10,
            temperature=0,
        )
        elapsed = time.time() - t0
        ok(f"响应成功 ({elapsed:.1f}s)")
        if verbose:
            print(f"     响应内容: {resp.strip()[:120]}")
        return True
    except Exception as e:
        fail(f"连接失败: {e}")
        return False


async def test_vlm(base_url: str, api_key: str, model: str, verbose: bool) -> bool:
    """测试 VLM 纯文本模式（不传图片，仅验证连通性）"""
    msg = "Hello, please respond with exactly one word: OK"
    try:
        t0 = time.time()
        resp = await openai_complete_if_cache(
            model=model,
            prompt=msg,
            api_key=api_key,
            base_url=base_url,
            max_tokens=10,
            temperature=0,
        )
        elapsed = time.time() - t0
        ok(f"响应成功 ({elapsed:.1f}s)")
        if verbose:
            print(f"     响应内容: {resp.strip()[:120]}")
        return True
    except Exception as e:
        fail(f"连接失败: {e}")
        return False


async def test_embedding(
    base_url: str, api_key: str, model: str, dim: int, verbose: bool
) -> bool:
    """测试 Embedding 向量化"""
    texts = ["Hello, world."]
    try:
        t0 = time.time()
        # Use .func to bypass @wrap_embedding_func_with_attrs (hard-coded
        # embedding_dim=1536 in the decorator).  The raw function accepts an
        # explicit embedding_dim that tells the API which dimension to return.
        result = await openai_embed.func(
            texts=texts,
            model=model,
            base_url=base_url,
            api_key=api_key,
            embedding_dim=dim,
        )
        elapsed = time.time() - t0
        actual_dim = len(result[0]) if len(result) > 0 else 0
        ok(f"响应成功 ({elapsed:.1f}s), 向量维度: {actual_dim}")
        if verbose:
            print(f"     前 5 维: {result[0][:5]}")
        return True
    except Exception as e:
        fail(f"连接失败: {e}")
        return False


async def test_rerank(
    base_url: str, api_key: str, model: str, verbose: bool
) -> bool:
    """测试 Rerank 重排序（仅做基础连通性测试）"""
    # Rerank API 没有标准化格式，这里仅尝试通用 POST 请求
    import json
    import httpx

    try:
        t0 = time.time()
        async with httpx.AsyncClient(timeout=30) as client:
            payload = {
                "model": model,
                "query": "Hello",
                "documents": ["World", "Hi there"],
            }
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            }
            resp = await client.post(
                f"{base_url.rstrip('/')}/rerank",
                json=payload,
                headers=headers,
            )
            elapsed = time.time() - t0
            if resp.status_code == 200:
                ok(f"响应成功 ({elapsed:.1f}s), 状态码: {resp.status_code}")
                if verbose:
                    print(f"     响应内容: {resp.text[:200]}")
                return True
            elif resp.status_code in (401, 403):
                warn(f"鉴权失败 (HTTP {resp.status_code})，请检查 API Key")
                return False
            elif resp.status_code == 404:
                warn(
                    f"未找到 Rerank 端点 (HTTP 404)，"
                    f"尝试了 {base_url.rstrip('/')}/rerank"
                )
                return False
            else:
                warn(f"异常状态码: HTTP {resp.status_code}")
                if verbose:
                    print(f"     响应内容: {resp.text[:200]}")
                return False
    except ImportError:
        warn("需要 httpx 库: pip install httpx")
        return False
    except Exception as e:
        warn(f"请求失败 (可能 API 格式不同): {e}")
        return False


# ===================================================================
# 主流程
# ===================================================================


async def main():
    parser = argparse.ArgumentParser(
        description="测试 .env 中配置的所有模型连通性"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="显示详细响应内容",
    )
    args = parser.parse_args()

    verbose = args.verbose

    print(f"{_BOLD}RAG-Anything 模型连通性测试{_RESET}")
    print(f"配置文件: {PROJECT_ROOT / '.env'}")
    print()

    # ------------------------------------------------------------------
    # 1. LLM
    # ------------------------------------------------------------------
    header("1. LLM 模型（文本生成）")
    llm_key = os.getenv("LLM_API_KEY", "")
    llm_url = os.getenv("LLM_BASE_URL", "")
    llm_model = os.getenv("LLM_MODEL", "")
    print(f"  Model:    {_BOLD}{llm_model}{_RESET}")
    print(f"  Base URL: {llm_url}")
    print(f"  API Key:  {llm_key[:8]}{'…' if llm_key else '<空>'}")

    if not llm_key or not llm_model:
        warn("未配置 LLM_API_KEY / LLM_MODEL，跳过")
        llm_ok = False
    else:
        llm_ok = await test_llm(llm_url, llm_key, llm_model, verbose)

    # ------------------------------------------------------------------
    # 2. VLM
    # ------------------------------------------------------------------
    header("2. VLM 视觉模型（图像理解）")
    vlm_key = os.getenv("VLM_API_KEY", "")
    vlm_url = os.getenv("VLM_BASE_URL", "")
    vlm_model = os.getenv("VLM_MODEL", "")
    print(f"  Model:    {_BOLD}{vlm_model}{_RESET}")
    print(f"  Base URL: {vlm_url}")
    print(f"  API Key:  {vlm_key[:8]}{'…' if vlm_key else '<空>'}")

    if not vlm_key or not vlm_model:
        warn("未配置 VLM_API_KEY / VLM_MODEL，跳过")
        vlm_ok = False
    else:
        vlm_ok = await test_vlm(vlm_url, vlm_key, vlm_model, verbose)

    # ------------------------------------------------------------------
    # 3. Embedding
    # ------------------------------------------------------------------
    header("3. Embedding 模型（文本向量化）")
    emb_key = os.getenv("EMBEDDING_API_KEY", "")
    emb_url = os.getenv("EMBEDDING_BASE_URL", "")
    emb_model = os.getenv("EMBEDDING_MODEL", "")
    emb_dim = int(os.getenv("EMBEDDING_DIM", "1024"))
    print(f"  Model:    {_BOLD}{emb_model}{_RESET}")
    print(f"  Base URL: {emb_url}")
    print(f"  API Key:  {emb_key[:8]}{'…' if emb_key else '<空>'}")

    if not emb_key or not emb_model:
        warn("未配置 EMBEDDING_API_KEY / EMBEDDING_MODEL，跳过")
        emb_ok = False
    else:
        emb_ok = await test_embedding(emb_url, emb_key, emb_model, emb_dim, verbose)

    # ------------------------------------------------------------------
    # 4. Rerank
    # ------------------------------------------------------------------
    header("4. Rerank 重排序模型（可选）")
    rerank_key = os.getenv("RERANK_API_KEY", "")
    rerank_url = os.getenv("RERANK_BASE_URL", "")
    rerank_model = os.getenv("RERANK_MODEL", "")
    print(f"  Model:    {_BOLD}{rerank_model}{_RESET}")
    print(f"  Base URL: {rerank_url}")
    print(f"  API Key:  {rerank_key[:8]}{'…' if rerank_key else '<空>'}")

    if not rerank_key or not rerank_model:
        warn("未配置 RERANK_API_KEY / RERANK_MODEL，跳过")
        rerank_ok = True
    else:
        rerank_ok = await test_rerank(rerank_url, rerank_key, rerank_model, verbose)

    # ------------------------------------------------------------------
    # 汇总
    # ------------------------------------------------------------------
    print()
    header("汇总")
    status = {
        "LLM (文本生成)": llm_ok,
        "VLM (图像理解)": vlm_ok,
        "Embedding (向量化)": emb_ok,
        "Rerank (重排序)": rerank_ok,
    }
    all_pass = True
    for name, ok_flag in status.items():
        tag = _PASS if ok_flag else _FAIL
        color = _GREEN if ok_flag else _RED
        print(f"  {color}{tag}{_RESET} {name}")
        if not ok_flag:
            all_pass = False

    print()
    if all_pass:
        print(f"  {_GREEN}{_BOLD}所有模型测试通过！{_RESET}")
    else:
        print(f"  {_YELLOW}部分模型测试未通过，请检查上方的错误信息。{_RESET}")

    return 0 if all_pass else 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)