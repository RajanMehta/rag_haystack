help:
	@echo "available commands"
	@echo " - install        	: installs production dependencies locally with uv"
	@echo " - install-dev    	: installs dev + production dependencies locally with uv"
	@echo " - lock           	: refreshes uv.lock"
	@echo " - build          	: builds haystack_api image (production, no dev deps) and qdrant image"
	@echo " - build-dev      	: builds haystack_api:dev image (with dev deps, used for tests)"
	@echo " - build-haystack 	: builds haystack_api image (production)"
	@echo " - build-haystack-dev	: builds haystack_api:dev image (with dev deps)"
	@echo " - build-qdrant   	: builds qdrant image"
	@echo " - up             	: stands up the stack"
	@echo " - dev             	: stands up the stack in dev mode"
	@echo " - logs           	: shows logs"
	@echo " - down           	: stops containers"
	@echo " - requirements		: exports requirements.txt file from uv (useful for ad-hoc image builds)"
	@echo " - isort          	: sort imports alphabetically, by sections and type"
	@echo " - black          	: python code formatter"
	@echo " - lint           	: checks linting, fixes automatically if possible"
	@echo " - clean           	: cleans the cache folders created by pytest and ruff"
	@echo " - format           	: runs isort + black + lint + clean"
	@echo " - stress           	: run load testing on haystack_api apis"
	@echo " - test           	: runs unit-tests (requires haystack_api:dev image)"
	@echo " - coverage        	: shows unit-test coverage (requires haystack_api:dev image)"
	@echo " - e2e            	: brings up the stack and runs end-to-end tests against it"
	@echo " - up-gpu           	: stands up the stack with gpu (more info in readme)"
	@echo " - dev-gpu        	: stands up the stack in dev mode with gpu (more info in readme)"
	@echo " - exp-list         	: lists registered experiments (see experiments/_registry.py)"
	@echo " - exp-up EXP=NNN_<slug>	: stands up the stack with PIPELINE_CONFIG=exp_<EXP>"

.PHONY: help init-persist install install-dev lock build build-dev build-haystack build-haystack-dev build-qdrant up dev logs down requirements lint isort black clean format stress test coverage e2e qdrant-cluster up-gpu dev-gpu standup-test exp-list exp-up

init-persist:
	./scripts/init-persist.sh

install:
	uv sync --locked --no-default-groups

install-dev:
	uv sync --locked --group dev

lock:
	uv lock

build: build-haystack build-qdrant

build-dev: build-haystack-dev build-qdrant

build-haystack:
	docker build --build-arg INSTALL_DEV=false -t haystack_api .

build-haystack-dev:
	docker build --build-arg INSTALL_DEV=true -t haystack_api:dev .

build-qdrant:
	docker build -t qdrant -f Dockerfile.qdrant .

up: init-persist
	docker-compose up -d

dev: init-persist
	docker-compose -f docker-compose.dev.yml up -d

logs:
	docker-compose logs -f

down:
	docker-compose down --remove-orphans

requirements:
	uv export --no-hashes --format requirements-txt --group dev --output-file requirements.txt

lint:
	uv run --group dev python -m ruff haystack_api tests --fix

isort:
	uv run --group dev python -m isort haystack_api tests

black:
	uv run --group dev python -m black haystack_api tests

clean:
	sudo rm -rf __pycache__ .pytest_cache .ruff_cache

format: isort black lint clean

stress: init-persist
	docker-compose -f docker-compose.stress.yml up

test: init-persist
	docker-compose -f docker-compose.test.yml up -d && docker-compose logs
	docker-compose exec -T haystack-api "./scripts/wait_for_haystack.sh"
	docker-compose exec -T haystack-api bash -c "python -m pytest tests/ --ignore=tests/e2e -vv -s -p no:cacheprovider"
	docker-compose down

coverage: init-persist
	docker-compose -f docker-compose.test.yml up -d
	docker-compose exec -T haystack-api "./scripts/wait_for_haystack.sh"
	docker-compose exec -u 0:0 -T haystack-api bash -c "python -m pytest -p no:cacheprovider --cov-report term-missing --cov=haystack_api --cov-fail-under=100 tests/ --ignore=tests/e2e"
	docker-compose down

e2e: init-persist
	docker-compose up -d
	@echo "Waiting for haystack-api to become healthy (up to 3 minutes)..."
	@for i in $$(seq 1 90); do \
		if curl -sf -m 3 http://localhost:31415/health >/dev/null 2>&1; then \
			echo "haystack-api is healthy."; break; \
		fi; \
		sleep 2; \
		if [ $$i -eq 90 ]; then echo "haystack-api never became healthy"; docker-compose logs haystack-api | tail -50; docker-compose down --remove-orphans; exit 1; fi; \
	done
	@set -e; \
		uv run --group dev pytest tests/e2e/ -vv -s -p no:cacheprovider; \
		status=$$?; \
		docker-compose down --remove-orphans; \
		exit $$status

qdrant-cluster: init-persist
	sudo mkdir -p qdrant_storage/0 qdrant_storage/1 qdrant_storage/2
	sudo chown -R 1000:1000 qdrant_storage/
	docker-compose -f docker-compose.yml -f docker-compose.qdrant-cluster.yml up -d

# GPU commands

up-gpu: init-persist
	docker-compose -f docker-compose.yml -f docker-compose.gpu.yml up -d

dev-gpu: init-persist
	docker-compose -f docker-compose.dev.yml -f docker-compose.gpu.yml up -d

standup-test: init-persist
	docker-compose up -d && docker-compose logs
	docker-compose exec -T haystack-api "./scripts/wait_for_haystack.sh"
	docker-compose exec -T haystack-api "./scripts/requests.sh"
	docker-compose down

# Experiments

exp-list:
	@uv run --group dev python -c "from experiments._registry import EXPERIMENT_CONFIGS; [print(k) for k in EXPERIMENT_CONFIGS]"

exp-up: init-persist
ifndef EXP
	$(error EXP is required, e.g. `make exp-up EXP=001_smart_document_splitter`)
endif
	PIPELINE_CONFIG=exp_$(EXP) docker-compose up -d
