.PHONY: install index test eval eval-claude

install:
	pip install -e ".[dev]"

index:
	payrag build-index

test:
	pytest -q

eval:
	payrag eval --agent offline

eval-claude:
	payrag eval --agent claude
