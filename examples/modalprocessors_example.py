"""
Example of directly using modal processors

This example demonstrates how to use RAG-Anything's modal processors directly
without going through MinerU.

Model configuration is read from .env (see LLM_*, VLM_*, EMBEDDING_* env vars).
"""

import asyncio
import argparse
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Use the local checkout and its root .env when this file is run directly.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
load_dotenv(dotenv_path=PROJECT_ROOT / ".env", override=False)

from lightrag import LightRAG
from lightrag.kg.shared_storage import initialize_pipeline_status
from raganything import create_llm_model_func, create_vlm_model_func, create_embedding_func
from raganything.modalprocessors import (
    ImageModalProcessor,
    TableModalProcessor,
    EquationModalProcessor,
)

WORKING_DIR = "./rag_storage"


async def process_image_example(lightrag: LightRAG, vision_model_func):
    """Example of processing an image"""
    # Create image processor
    image_processor = ImageModalProcessor(
        lightrag=lightrag, modal_caption_func=vision_model_func
    )

    # Prepare image content
    image_content = {
        "img_path": "image.jpg",
        "image_caption": ["Example image caption"],
        "image_footnote": ["Example image footnote"],
    }

    # Process image
    (description, entity_info, _) = await image_processor.process_multimodal_content(
        modal_content=image_content,
        content_type="image",
        file_path="image_example.jpg",
        entity_name="Example Image",
    )

    print("Image Processing Results:")
    print(f"Description: {description}")
    print(f"Entity Info: {entity_info}")


async def process_table_example(lightrag: LightRAG, llm_model_func):
    """Example of processing a table"""
    # Create table processor
    table_processor = TableModalProcessor(
        lightrag=lightrag, modal_caption_func=llm_model_func
    )

    # Prepare table content
    table_content = {
        "table_body": """
        | Name | Age | Occupation |
        |------|-----|------------|
        | John | 25  | Engineer   |
        | Mary | 30  | Designer   |
        """,
        "table_caption": ["Employee Information Table"],
        "table_footnote": ["Data updated as of 2024"],
    }

    # Process table
    (description, entity_info, _) = await table_processor.process_multimodal_content(
        modal_content=table_content,
        content_type="table",
        file_path="table_example.md",
        entity_name="Employee Table",
    )

    print("\nTable Processing Results:")
    print(f"Description: {description}")
    print(f"Entity Info: {entity_info}")


async def process_equation_example(lightrag: LightRAG, llm_model_func):
    """Example of processing a mathematical equation"""
    # Create equation processor
    equation_processor = EquationModalProcessor(
        lightrag=lightrag, modal_caption_func=llm_model_func
    )

    # Prepare equation content
    equation_content = {"text": "E = mc^2", "text_format": "LaTeX"}

    # Process equation
    (description, entity_info, _) = await equation_processor.process_multimodal_content(
        modal_content=equation_content,
        content_type="equation",
        file_path="equation_example.txt",
        entity_name="Mass-Energy Equivalence",
    )

    print("\nEquation Processing Results:")
    print(f"Description: {description}")
    print(f"Entity Info: {entity_info}")


async def initialize_rag():
    """Initialize LightRAG with model functions from .env."""
    llm_model_func = create_llm_model_func()
    embedding_func = create_embedding_func()

    rag = LightRAG(
        working_dir=WORKING_DIR,
        embedding_func=embedding_func,
        llm_model_func=llm_model_func,
    )

    await rag.initialize_storages()
    await initialize_pipeline_status()

    return rag


def main():
    """Main function to run the example"""
    parser = argparse.ArgumentParser(description="Modal Processors Example")
    parser.add_argument(
        "--working-dir", "-w", default=WORKING_DIR, help="Working directory path"
    )
    args = parser.parse_args()

    # Run examples
    asyncio.run(main_async())


async def main_async():
    # 使用 model_config 工厂函数自动从 .env 读取配置
    # 每类模型可独立配置: LLM_*, VLM_*, EMBEDDING_* 环境变量
    llm_model_func = create_llm_model_func()
    vision_model_func = create_vlm_model_func()

    # Initialize LightRAG
    lightrag = await initialize_rag()

    # Run examples
    await process_image_example(lightrag, vision_model_func)
    await process_table_example(lightrag, llm_model_func)
    await process_equation_example(lightrag, llm_model_func)


if __name__ == "__main__":
    main()