# Smart Traffic Command Center — common workflows
.PHONY: help install pipeline api dashboard test lint docker-build docker-up clean

help:           ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
	 awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

install:        ## Install Python dependencies
	pip install -r requirements.txt

pipeline:       ## Run the full data + ML pipeline
	python -m src.run_pipeline

api:            ## Serve the REST API (http://localhost:8000/docs)
	uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload

dashboard:      ## Launch the Streamlit command center (http://localhost:8501)
	streamlit run dashboard/app.py

test:           ## Run the test suite
	pytest tests/ -q

docker-build:   ## Build the Docker image
	docker compose build

docker-up:      ## Build artifacts then serve api + dashboard
	docker compose run --rm pipeline
	docker compose up api dashboard

clean:          ## Remove generated artifacts
	rm -rf outputs/* models/*/ logs/* data/interim/* data/processed/* \
	       **/__pycache__ .pytest_cache
