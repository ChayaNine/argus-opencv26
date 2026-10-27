.PHONY: install test run eval bench-local inputs

install:
	pip install -r requirements-dev.txt

test:
	python -m pytest -q tests

run:
	python -m uvicorn argus.server.app:app --host 127.0.0.1 --port 8080

eval:
	python -m eval.run_eval --scenes 30 --gallery

inputs:
	python -m bench.make_inputs

bench-local:
	python -m bench.bench --label local --seconds 10 && python -m bench.compare
