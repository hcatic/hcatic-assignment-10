PYTHON = python
FLASK_APP = app.py
VENV_DIR = venv

install:
	$(PYTHON) -m venv $(VENV_DIR)
	$(VENV_DIR)/bin/pip install --upgrade pip
	$(VENV_DIR)/bin/pip install Flask open-clip-torch torch pandas pillow

run:
	FLASK_APP=$(FLASK_APP) $(VENV_DIR)/bin/flask run