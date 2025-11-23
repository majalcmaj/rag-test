run:
	cd src && poetry run python main.py

lint:
	poetry run black src/

test:
	poetry run pytest src/

