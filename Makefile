run_preprocess:
	docker compose run dev poetry run jupyter nbconvert --to notebook --execute /workspace/notebook/preprocess/00_NEWS_EMBEDDING.ipynb
	docker compose run dev poetry run jupyter nbconvert --to notebook --execute /workspace/notebook/preprocess/01_NEWS_ENTITY.ipynb
	docker compose run dev poetry run jupyter nbconvert --to notebook --execute /workspace/notebook/preprocess/02_NEWS_TIME.ipynb
	docker compose run dev poetry run jupyter nbconvert --to notebook --execute /workspace/notebook/preprocess/03_FEATURE.ipynb

run_baseline:
	docker compose run -d dev poetry run jupyter nbconvert --to notebook --execute /workspace/notebook/experiment/00_BASELINE.ipynb

run_proposal:
	docker compose run -d dev poetry run jupyter nbconvert --to notebook --execute /workspace/notebook/experiment/01_PROPOSAL.ipynb
