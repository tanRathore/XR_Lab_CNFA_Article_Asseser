
import argparse
from cna_rag_agent.utils.logging_config import logger
from cna_rag_agent import config # Ensures config is loaded

def run_ingestion_pipeline_main(args):
    logger.info("Attempting to start data ingestion pipeline via main.py...")
    try:
        from cna_rag_agent.pipeline.rag_pipeline import run_full_ingestion_pipeline
        run_full_ingestion_pipeline(
            clear_vector_store=args.clear_store,
            force_reprocess=args.force_reprocess # Pass the new flag
        )
    except ImportError as e:
        logger.error(f"Could not import 'run_full_ingestion_pipeline': {e}", exc_info=True)
    except Exception as e:
        logger.error(f"An error occurred during the ingestion pipeline execution from main.py: {e}", exc_info=True)

def start_streamlit_app_main():
    logger.info("Attempting to start Streamlit application via main.py...")
    try:
        import subprocess
        import sys
        streamlit_app_path = config.BASE_DIR / "src" / "streamlit_app.py"

        if not streamlit_app_path.exists():
            logger.error(f"Streamlit app file not found at: {streamlit_app_path}")
            logger.error("Please ensure 'streamlit_app.py' exists in the 'src/' directory.")
            return

        logger.info(f"Launching Streamlit from: {streamlit_app_path}")
        process = subprocess.Popen([sys.executable, "-m", "streamlit", "run", str(streamlit_app_path)])
        process.wait()
        logger.info("Streamlit application process finished.")
    except ImportError as e:
        logger.error(f"Import error while trying to start Streamlit: {e}", exc_info=True)
    except FileNotFoundError:
        logger.error("Streamlit command not found. Is Streamlit installed in the environment and in PATH?")
    except Exception as e:
        logger.error(f"An error occurred while trying to start the Streamlit app: {e}", exc_info=True)


def main():
    parser = argparse.ArgumentParser(description="CNfA RAG Agent Command Line Interface")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    subparsers.required = True

    ingest_parser = subparsers.add_parser("ingest", help="Run the data ingestion pipeline.")
    ingest_parser.add_argument(
        "--clear-store",
        action="store_true",
        help="Clear the existing vector store before ingesting new documents."
    )
    ingest_parser.add_argument(
        "--force-reprocess",
        action="store_true",
        help="Force reprocessing of documents, bypassing any existing cache for chunks/embeddings."
    )
    ingest_parser.set_defaults(func=run_ingestion_pipeline_main)

    streamlit_parser = subparsers.add_parser("streamlit_app", help="Start the Streamlit UI.")
    streamlit_parser.set_defaults(func=start_streamlit_app_main)

    args = parser.parse_args()
    
    logger.info(f"Executing command: {args.command} with arguments: {vars(args)}")
    if hasattr(args, 'func'):
        if args.command == "ingest":
            args.func(args)
        else:
            args.func()
    else:
        parser.print_help()

if __name__ == "__main__":
    logger.info(f"CNfA RAG Agent starting. Project base directory: {config.BASE_DIR}")
    main()