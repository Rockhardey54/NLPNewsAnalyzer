@echo off

:: Set up the current directory as the project directory
SET PROJECT_DIR=%cd%

:: Create the virtual environment if it doesn't exist
IF NOT EXIST "%PROJECT_DIR%\venv" (
    echo Creating virtual environment...
    python -m venv venv
)

:: Activate the virtual environment
call venv\Scripts\activate

:: Install dependencies (assuming you have a requirements.txt in the repo)
echo Installing dependencies...
pip install -r requirements.txt

:: Install the spaCy model (if not already installed)
echo Installing spaCy model...
python -m spacy download en_core_web_md

:: Run the Jupyter notebook
echo Running Jupyter notebook...
jupyter notebook --no-browser --port=8888 "%PROJECT_DIR%\headlines.ipynb"
