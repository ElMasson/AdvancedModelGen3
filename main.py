# main.py
import streamlit as st
from interface.main_page import MainPage
import logging
from core.config import Config


def setup_logging():
    """Configure le système de logging."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def main():
    """Point d'entrée principal de l'application."""
    # Configuration de la page Streamlit (doit être la première commande Streamlit)
    st.set_page_config(
        page_title="Database Schema Analyzer",
        page_icon="🔍",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    setup_logging()
    logger = logging.getLogger(__name__)

    try:
        main_page = MainPage()
        main_page.render()
    except Exception as e:
        logger.error(f"Application error: {str(e)}", exc_info=True)
        st.error("An unexpected error occurred. Please check the logs for details.")


if __name__ == "__main__":
    main()
