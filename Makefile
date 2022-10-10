.PHONY: test serve

test:
	pipenv run python3 -m test.test_empr
	pipenv run python3 -m test.test_poly
	pipenv run python3 -m test.test_trig
	pipenv run python3 -m test.test_3d

serve:
	FLASK_APP=visualizer.py pipenv run flask run
