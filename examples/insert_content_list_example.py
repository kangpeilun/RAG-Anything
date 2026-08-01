#!/usr/bin/env python
"""
Example script demonstrating direct content list insertion with RAGAnything

This example shows how to:
1. Create a simple content list with different content types
2. Insert content list directly without document parsing using insert_content_list() method
3. Perform pure text queries using aquery() method
4. Perform multimodal queries with specific multimodal content using aquery_with_multimodal() method
5. Handle different types of multimodal content in the inserted knowledge base
"""

import os
import argparse
import asyncio
import logging
import logging.config
from pathlib import Path

# Add project root directory to Python path before site-packages.
# This ensures examples use the local checkout, not an installed raganything package.
import sys

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
load_dotenv(dotenv_path=PROJECT_ROOT / ".env", override=False)

from lightrag.utils import EmbeddingFunc, logger, set_verbose_debug
from raganything import RAGAnything, RAGAnythingConfig
from raganything.model_config import (
    create_llm_model_func,
    create_vlm_model_func,
    create_embedding_func,
)


def configure_logging():
    """Configure logging for the application"""
    # Get log directory path from environment variable or use current directory
    log_dir = os.getenv("LOG_DIR", os.getcwd())
    log_file_path = os.path.abspath(
        os.path.join(log_dir, "insert_content_list_example.log")
    )

    print(f"\nInsert Content List example log file: {log_file_path}\n")
    os.makedirs(os.path.dirname(log_dir), exist_ok=True)

    # Get log file max size and backup count from environment variables
    log_max_bytes = int(os.getenv("LOG_MAX_BYTES", 10485760))  # Default 10MB
    log_backup_count = int(os.getenv("LOG_BACKUP_COUNT", 5))  # Default 5 backups

    logging.config.dictConfig(
        {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "default": {
                    "format": "%(levelname)s: %(message)s",
                },
                "detailed": {
                    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                },
            },
            "handlers": {
                "console": {
                    "formatter": "default",
                    "class": "logging.StreamHandler",
                    "stream": "ext://sys.stderr",
                },
                "file": {
                    "formatter": "detailed",
                    "class": "logging.handlers.RotatingFileHandler",
                    "filename": log_file_path,
                    "maxBytes": log_max_bytes,
                    "backupCount": log_backup_count,
                    "encoding": "utf-8",
                },
            },
            "loggers": {
                "lightrag": {
                    "handlers": ["console", "file"],
                    "level": "INFO",
                    "propagate": False,
                },
            },
        }
    )

    # Set the logger level to INFO
    logger.setLevel(logging.INFO)
    # Enable verbose debug if needed
    set_verbose_debug(os.getenv("VERBOSE", "false").lower() == "true")


def create_sample_content_list():
    """
    Create a simple content list for testing insert_content_list functionality

    Returns:
        List[Dict]: Sample content list with various content types

    Note:
        - img_path should be absolute path to the image file
        - page_idx represents the page number where the content appears (0-based)
    """
    content_list = [
        # Introduction text
        {
            "type": "text",
            "text": "Welcome to the RAGAnything System Documentation. This guide covers the advanced multimodal document processing capabilities and features of our comprehensive RAG system.",
            "page_idx": 0,  # Page number where this content appears
        },
        # System architecture image
        {
            "type": "image",
            "img_path": "/absolute/path/to/system_architecture.jpg",  # IMPORTANT: Use absolute path to image file
            "image_caption": ["Figure 1: RAGAnything System Architecture"],
            "image_footnote": [
                "The architecture shows the complete pipeline from document parsing to multimodal query processing"
            ],
            "page_idx": 1,  # Page number where this image appears
        },
        # Performance comparison table
        {
            "type": "table",
            "table_body": """| System | Accuracy | Processing Speed | Memory Usage |
                            |--------|----------|------------------|--------------|
                            | RAGAnything | 95.2% | 120ms | 2.1GB |
                            | Traditional RAG | 87.3% | 180ms | 3.2GB |
                            | Baseline System | 82.1% | 220ms | 4.1GB |
                            | Simple Retrieval | 76.5% | 95ms | 1.8GB |""",
            "table_caption": [
                "Table 1: Performance Comparison of Different RAG Systems"
            ],
            "table_footnote": [
                "All tests conducted on the same hardware with identical test datasets"
            ],
            "page_idx": 2,  # Page number where this table appears
        },
        # Mathematical formula
        {
            "type": "equation",
            "latex": "Relevance(d, q) = \\sum_{i=1}^{n} w_i \\cdot sim(t_i^d, t_i^q) \\cdot \\alpha_i",
            "text": "Document relevance scoring formula where w_i are term weights, sim() is similarity function, and α_i are modality importance factors",
            "page_idx": 3,  # Page number where this equation appears
        },
        # Feature description
        {
            "type": "text",
            "text": "The system supports multiple content modalities including text, images, tables, and mathematical equations. Each modality is processed using specialized processors optimized for that content type.",
            "page_idx": 4,  # Page number where this content appears
        },
        # Technical specifications table
        {
            "type": "table",
            "table_body": """| Feature | Specification |
                            |---------|---------------|
                            | Supported Formats | PDF, DOCX, PPTX, XLSX, Images |
                            | Max Document Size | 100MB |
                            | Concurrent Processing | Up to 8 documents |
                            | Query Response Time | <200ms average |
                            | Knowledge Graph Nodes | Up to 1M entities |""",
            "table_caption": ["Table 2: Technical Specifications"],
            "table_footnote": [
                "Specifications may vary based on hardware configuration"
            ],
            "page_idx": 5,  # Page number where this table appears
        },
        # Conclusion
        {
            "type": "text",
            "text": "RAGAnything represents a significant advancement in multimodal document processing, providing comprehensive solutions for complex knowledge extraction and retrieval tasks.",
            "page_idx": 6,  # Page number where this content appears
        },
    ]

    return content_list


async def demo_insert_content_list(
    working_dir: str = None,
):
    """
    Demonstrate content list insertion and querying with RAGAnything

    Model configuration is read from .env (see LLM_*, VLM_*, EMBEDDING_* env vars).

    Args:
        working_dir: Working directory for RAG storage
    """
    try:
        # Create RAGAnything configuration
        config = RAGAnythingConfig(
            working_dir=working_dir or "./rag_storage",
            enable_image_processing=True,
            enable_table_processing=True,
            enable_equation_processing=True,
            display_content_stats=True,  # Show content statistics
        )

        # 使用 model_config 工厂函数自动从 .env 读取配置
        # 每类模型可独立配置: LLM_*, VLM_*, EMBEDDING_* 环境变量
        llm_model_func = create_llm_model_func()
        vision_model_func = create_vlm_model_func()
        embedding_func = create_embedding_func()

        # 如需自定义覆盖，也可手动传入 OpenAIModelConfig:
        # from raganything.model_config import OpenAIModelConfig
        # llm_model_func = create_llm_model_func(
        #     OpenAIModelConfig(api_key="xxx", base_url="xxx", model="deepseek-chat")
        # )

        # Initialize RAGAnything
        rag = RAGAnything(
            config=config,
            llm_model_func=llm_model_func,
            vision_model_func=vision_model_func,
            embedding_func=embedding_func,
        )

        # Create sample content list
        logger.info("Creating sample content list...")
        content_list = create_sample_content_list()
        logger.info(f"Created content list with {len(content_list)} items")

        # Insert content list directly
        logger.info("\nInserting content list into RAGAnything...")
        await rag.insert_content_list(
            content_list=content_list,
            file_path="raganything_documentation.pdf",  # Reference file name for citation
            split_by_character=None,  # Optional text splitting
            split_by_character_only=False,  # Optional text splitting mode
            doc_id="demo-doc-001",  # Custom document ID
            display_stats=True,  # Show content statistics
        )
        logger.info("Content list insertion completed!")

        # Example queries - demonstrating different query approaches
        logger.info("\nQuerying inserted content:")

        # 1. Pure text queries using aquery()
        text_queries = [
            "What is RAGAnything and what are its main features?",
            "How does RAGAnything compare to traditional RAG systems?",
            "What are the technical specifications of the system?",
        ]

        for query in text_queries:
            logger.info(f"\n[Text Query]: {query}")
            result = await rag.aquery(query, mode="hybrid")
            logger.info(f"Answer: {result}")

        # 2. Multimodal query with specific multimodal content using aquery_with_multimodal()
        logger.info(
            "\n[Multimodal Query]: Analyzing new performance data against existing benchmarks"
        )
        multimodal_result = await rag.aquery_with_multimodal(
            "Compare this new performance data with the existing benchmark results in the documentation",
            multimodal_content=[
                {
                    "type": "table",
                    "table_data": """Method,Accuracy,Speed,Memory
                                New_Approach,97.1%,110ms,1.9GB
                                Enhanced_RAG,91.4%,140ms,2.5GB""",
                    "table_caption": "Latest experimental results",
                }
            ],
            mode="hybrid",
        )
        logger.info(f"Answer: {multimodal_result}")

        # 3. Another multimodal query with equation content
        logger.info("\n[Multimodal Query]: Mathematical formula analysis")
        equation_result = await rag.aquery_with_multimodal(
            "How does this similarity formula relate to the relevance scoring mentioned in the documentation?",
            multimodal_content=[
                {
                    "type": "equation",
                    "latex": "sim(a, b) = \\frac{a \\cdot b}{||a|| \\times ||b||} + \\beta \\cdot context\\_weight",
                    "equation_caption": "Enhanced cosine similarity with context weighting",
                }
            ],
            mode="hybrid",
        )
        logger.info(f"Answer: {equation_result}")

        # 4. Insert another content list with different document ID
        logger.info("\nInserting additional content list...")
        additional_content = [
            {
                "type": "text",
                "text": "This is additional documentation about advanced features and configuration options.",
                "page_idx": 0,  # Page number where this content appears
            },
            {
                "type": "table",
                "table_body": """| Configuration | Default Value | Range |
                                    |---------------|---------------|-------|
                                    | Chunk Size | 512 tokens | 128-2048 |
                                    | Context Window | 4096 tokens | 1024-8192 |
                                    | Batch Size | 32 | 1-128 |""",
                "table_caption": ["Advanced Configuration Parameters"],
                "page_idx": 1,  # Page number where this table appears
            },
        ]

        await rag.insert_content_list(
            content_list=additional_content,
            file_path="advanced_configuration.pdf",
            doc_id="demo-doc-002",  # Different document ID
        )

        # Query combined knowledge base
        logger.info("\n[Combined Query]: What configuration options are available?")
        combined_result = await rag.aquery(
            "What configuration options are available and what are their default values?",
            mode="hybrid",
        )
        logger.info(f"Answer: {combined_result}")

    except Exception as e:
        logger.error(f"Error in content list insertion demo: {str(e)}")
        import traceback

        logger.error(traceback.format_exc())


def main():
    """Main function to run the example"""
    parser = argparse.ArgumentParser(description="Insert Content List Example")
    parser.add_argument(
        "--working_dir", "-w", default="./rag_storage", help="Working directory path"
    )

    args = parser.parse_args()

    # Run the demo
    asyncio.run(
        demo_insert_content_list(
            working_dir=args.working_dir,
        )
    )


if __name__ == "__main__":
    # Configure logging first
    configure_logging()

    print("RAGAnything Insert Content List Example")
    print("=" * 45)
    print("Demonstrating direct content list insertion without document parsing")
    print("=" * 45)

    main()
