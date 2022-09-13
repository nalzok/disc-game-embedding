.PHONY: test

test:
	pipenv run python3 -m test.game1
	pipenv run python3 -m test.game2
	pipenv run python3 -m test.game3
