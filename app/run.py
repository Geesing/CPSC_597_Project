import multiprocessing
from streamlit_desktop_app import start_desktop_app

if __name__ == "__main__":
    multiprocessing.freeze_support()
    
    start_desktop_app("app.py", title="Document Searcher")