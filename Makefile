
gui_dev:
	jurigged ./gtk_ui.py

gui_dependencies:
	uv sync --locked --with gtk
